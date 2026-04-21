#!/usr/bin/env python3
"""
Model Calibration — Formal Cost Model
======================================
Predictive equations for T_prefill, T_decode, and total latency.

Two-component latency model (derived from calibration + Phase 1 baseline):

    T_TTFT = T_vision_encoder  +  T_lm_prefill(N)  +  T_decode

Where:
  T_vision_encoder  ≈ 5991ms  (SigLIP crops; FIXED regardless of N_lm tokens)
  T_lm_prefill(N)   = γ·N² + β·N + α  (calibrated; scales with kept tokens)
  T_decode          ≈ roofline memory-bandwidth bound

Calibration results (model_calibration/calibration_results.json):
  γ = 2.095744e-05  ms/token²   R² = 0.9978
  β = 1.590525      ms/token    Scaling regime: LINEAR (N < 8192 crossover)
  α = -20.08        ms

Key finding: at N ≤ 1548, MLP layers (O(N)) dominate over attention (O(N²)).
The practical scaling is almost purely linear (quadratic term is <2% at N=1548).

Validation discrepancy (70.6% error) is EXPECTED:
  Phase 1 T_prefill = T_vision_encoder + T_lm_prefill = 5991 + 2498 = 8489ms
  Calibration only measures T_lm_prefill (vision encoder bypassed).
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

@dataclass
class CostModelConfig:
    """Configuration for the cost model."""

    # ── LM transformer calibration coefficients ───────────────────────────────
    # Source: model_calibration/calibration_results.json  (R² = 0.9978)
    gamma: float = 2.095744e-05   # ms/token²
    beta:  float = 1.590525       # ms/token
    alpha: float = -20.08         # ms constant

    # ── Vision encoder fixed cost ─────────────────────────────────────────────
    # SigLIP processes all image crops regardless of how many LM tokens are kept.
    # Empirically derived: Phase1_T_prefill(1548) − LM_T_prefill(1548)
    #   = 8489ms − 2498ms = 5991ms
    vision_encoder_ms: float = 5991.0  # ms (fixed; independent of N_lm)

    # ── Roofline constants (Apple M3, 8GB) ───────────────────────────────────
    m3_compute_peak_tflops: float = 3.6       # TFLOPS FP16
    m3_bandwidth_gbs: float = 100.0           # GB/s unified memory
    ridge_point_flops_per_byte: float = 36.0  # compute_peak / bandwidth

    # ── LM architecture (SmolVLM's language model, from calibration) ─────────
    num_layers:  int = 24
    hidden_size: int = 2048   # LM hidden size (NOT vision encoder's 1152)
    num_heads:   int = 32
    vocab_size:  int = 49152

    # ── Quantization savings ──────────────────────────────────────────────────
    quantization_savings: Dict[int, float] = None

    # ── ParVTS migration depth ────────────────────────────────────────────────
    migration_depth_default: int = 3

    def __post_init__(self):
        if self.quantization_savings is None:
            self.quantization_savings = {
                32: 1.0,
                16: 0.5,
                8:  0.25,
                4:  0.125,
            }


class CostModel:
    """
    Unified cost model for latency prediction.
    
    Core equations:
    - T_prefill(N) ≈ γ·N² + β·N + α             [from calibration]
    - T_decode(seq) ≈ (A + K) / bandwidth       [roofline]
    - T_migration(n, N) ≈ cost of pruning to n layers
    - T_total(res, prune, quant) = T_prefill + T_decode + migration cost
    """
    
    def __init__(self, config: CostModelConfig = None, calibration_path: Path = None):
        """
        Initialize cost model.  Optionally load coefficients from a
        calibration_results.json produced by calibrate_cost_model.py.
        """
        self.config = config or CostModelConfig()
        self.calibration_r2: Optional[float] = None
        self.calibration_hidden_size: Optional[int] = None

        if calibration_path is None:
            # Auto-discover calibration file next to this script
            default = Path(__file__).parent / "calibration_results.json"
            if default.exists():
                calibration_path = default

        if calibration_path and calibration_path.exists():
            with open(calibration_path) as f:
                calib = json.load(f)
            self.config.gamma = calib["gamma"]
            self.config.beta  = calib["beta"]
            self.config.alpha = calib["alpha"]
            self.calibration_r2 = calib.get("r2")
            self.calibration_hidden_size = calib.get("hidden_size")
    
    def predict_t_lm_prefill(self, n_tokens: int) -> float:
        """
        Predict LM-transformer prefill time for N tokens.

        T_lm_prefill(N) = γ·N² + β·N + α

        NOTE: At N ≤ 1548 the linear term dominates (MLP layers >> attention).
        The quadratic correction is <2% of total.
        Does NOT include vision encoder time (see vision_encoder_ms).
        """
        return max(0.0,
            self.config.gamma * n_tokens**2 +
            self.config.beta  * n_tokens +
            self.config.alpha
        )

    # Keep the old name as an alias so existing callers don't break
    def predict_t_prefill(self, n_tokens: int) -> float:
        return self.predict_t_lm_prefill(n_tokens)
    
    def predict_t_decode_roofline(
        self,
        seq_len: int,
        token_budget: int = 60,
        quantization_bits: int = 16,
    ) -> float:
        """
        Predict T_decode using Roofline model.
        
        For decode: mostly reading KV cache and weights, minimal compute.
        Bytes = KV_cache(seq_len) + Weights(quantization_bits)
        T_decode ≈ Bytes / Bandwidth
        
        Args:
            seq_len: Sequence length for KV cache
            token_budget: How many tokens to decode (for estimating load)
            quantization_bits: Precision for weights (4, 8, 16, 32)
            
        Returns:
            Predicted average TBT in milliseconds per token
        """
        # KV cache bytes: 2 (K+V) × seq_len × n_layers × hidden_size × 2 (FP16)
        kv_bytes = 2 * seq_len * self.config.num_layers * self.config.hidden_size * 2

        # Weight bytes: SmolVLM LM ~500M params (256M LM, ~250M vision) at given quant
        # 64MB reported for 4-bit; scale to 16-bit baseline = 64 × (16/4) = 256MB
        weight_bytes_fp16 = 256 * 1e6
        quant_factor = self.config.quantization_savings.get(quantization_bits, 1.0)
        weight_bytes = weight_bytes_fp16 * quant_factor * 2  # *2 =bytes from half-factors
        
        # Roofline: T = (KV + W) / Bandwidth
        total_bytes = (kv_bytes + weight_bytes) / 1e9  # Convert to GB
        bandwidth_bytes_per_s = self.config.m3_bandwidth_gbs * 1e9
        
        t_per_token_s = total_bytes / bandwidth_bytes_per_s
        t_per_token_ms = t_per_token_s * 1e3
        
        return t_per_token_ms
    
    def predict_migration_cost(
        self,
        n_full_tokens: int,
        n_pruned_tokens: int,
        migration_depth: int = 3,
    ) -> float:
        """
        ParVTS-style migration cost for token pruning.
        
        FLOPs_saved ≈ n_pruned × (num_layers - migration_depth) × L²
        Where L = context length.
        
        For now: rough estimate based on token reduction ratio.
        
        Args:
            n_full_tokens: Original token count
            n_pruned_tokens: After pruning
            migration_depth: Number of layers that still process all tokens
            
        Returns:
            Migration latency cost in milliseconds (overhead of pruning decision)
        """
        prune_ratio = (n_full_tokens - n_pruned_tokens) / n_full_tokens
        
        # Rough estimate: pruning overhead ≈ 2-5% of prefill time for shallow pruning
        # Deeper pruning (higher migration_depth) has less overhead
        overhead_ratio = 0.03 * (1.0 - migration_depth / self.config.num_layers)
        
        t_prefill_baseline = self.predict_t_prefill(n_full_tokens)
        migration_cost_ms = t_prefill_baseline * overhead_ratio * prune_ratio
        
        return migration_cost_ms
    
    def predict_latency(
        self,
        n_visual_tokens: int,
        n_decode_tokens: int = 60,
        quantization_bits: int = 16,
        pruning_ratio: float = 1.0,
        migration_depth: int = 3,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """
        Full latency prediction.
        
        Args:
            n_visual_tokens: Number of visual tokens (before pruning)
            n_decode_tokens: Target number of tokens to generate
            quantization_bits: Weight precision (4, 8, 16, 32)
            pruning_ratio: Keep ratio (1.0 = no pruning, 0.25 = prune to 25%)
            migration_depth: ParVTS depth parameter
            batch_size: Batch size (for KV cache and batching effects)
            
        Returns:
            Dict with t_prefill, t_decode, t_total, and component breakdown
        """
        # Apply pruning
        n_tokens_after_prune = max(int(n_visual_tokens * pruning_ratio), 32)

        # ── Component 1: Vision encoder (FIXED — independent of N_lm) ─────────
        t_vision_encoder = self.config.vision_encoder_ms

        # ── Component 2: LM transformer prefill ──────────────────────────────
        t_lm_prefill = self.predict_t_lm_prefill(n_tokens_after_prune)

        # ── Component 3: Token pruning migration overhead ─────────────────────
        t_migration = self.predict_migration_cost(
            n_visual_tokens, n_tokens_after_prune, migration_depth
        )

        # ── Component 4: Decode (roofline) ────────────────────────────────────
        avg_seq_len = (n_tokens_after_prune + n_decode_tokens) // 2
        t_decode_per_tok = self.predict_t_decode_roofline(
            avg_seq_len, n_decode_tokens, quantization_bits
        )
        t_decode_total = t_decode_per_tok * n_decode_tokens

        batch_overhead = 1.0 + 0.1 * (batch_size - 1)
        t_prefill_total = (t_vision_encoder + t_lm_prefill + t_migration) * batch_overhead
        t_total = t_prefill_total + t_decode_total * batch_overhead

        return {
            "t_vision_encoder_ms":   round(t_vision_encoder, 1),
            "t_lm_prefill_ms":       round(t_lm_prefill, 1),
            "t_migration_ms":        round(t_migration, 1),
            "t_decode_total_ms":     round(t_decode_total, 1),
            "t_decode_per_token_ms": round(t_decode_per_tok, 2),
            "t_total_ms":            round(t_total, 1),
            "n_tokens_effective":    n_tokens_after_prune,
            "pruning_ratio":         round(pruning_ratio, 3),
            "batch_overhead":        round(batch_overhead, 2),
            "sla_pass":              t_total <= 500.0,
            "note": ("Vision encoder (5991ms) dominates; token pruning reduces LM "
                     "component only."),
        }
    
    def find_sla_pruning_target(
        self,
        n_visual_tokens: int,
        sla_budget_ms: float = 500.0,
        quantization_bits: int = 16,
    ) -> Dict:
        """
        Find the minimum token count needed to meet SLA.
        
        Binary search for N such that T_prefill(N) ≤ SLA.
        
        Args:
            n_visual_tokens: Current token count
            sla_budget_ms: Latency budget (500ms standard)
            quantization_bits: Weight precision to assume
            
        Returns:
            Dict with target_tokens, pruning_ratio, and estimated latency
        """
        # First check if vision encoder alone exceeds SLA budget
        if self.config.vision_encoder_ms >= sla_budget_ms:
            return {
                "target_tokens": None,
                "pruning_ratio": None,
                "compression_ratio": None,
                "predicted_t_total_ms": None,
                "sla_budget_ms": sla_budget_ms,
                "sla_met": False,
                "blocking_component": "vision_encoder",
                "note": (
                    f"Vision encoder ({self.config.vision_encoder_ms:.0f}ms) alone "
                    f"exceeds SLA ({sla_budget_ms:.0f}ms). LM token pruning cannot "
                    "compensate; vision encoder optimization required."
                ),
            }

        # Binary search: find max N_lm such that full TTFT ≤ SLA
        lm_budget = sla_budget_ms - self.config.vision_encoder_ms
        low, high = 32, n_visual_tokens
        target_tokens = 32

        while low <= high:
            mid = (low + high) // 2
            t_lm = self.predict_t_lm_prefill(mid)
            if t_lm <= lm_budget:
                target_tokens = mid
                low = mid + 1
            else:
                high = mid - 1

        result_pred = self.predict_latency(
            n_visual_tokens=n_visual_tokens,
            n_decode_tokens=60,
            quantization_bits=quantization_bits,
            pruning_ratio=target_tokens / n_visual_tokens,
        )

        return {
            "target_tokens": target_tokens,
            "pruning_ratio": round(target_tokens / n_visual_tokens, 3),
            "compression_ratio": round(n_visual_tokens / target_tokens, 2),
            "predicted_t_total_ms": result_pred["t_total_ms"],
            "lm_budget_ms": round(lm_budget, 1),
            "sla_budget_ms": sla_budget_ms,
            "sla_met": result_pred["sla_pass"],
        }
    
    def find_resolution_wall(self, sla_budget_ms: float = 500.0) -> Dict:
        """
        Identify the resolution at which even aggressive pruning can't meet SLA.
        
        For SmolVLM's tiling, assumes ~1548 tokens at all resolutions up to 1024px.
        Beyond that (hypothetically), token count grows quadratically.
        
        Returns:
            Dict with wall_resolution (approx) and supporting analysis
        """
        # SmolVLM clamps to 1548 tokens for resolutions up to 1024px
        # Hypothetical: beyond 1024px, assume tiles scale as (res/patch_size)²
        
        # LM budget after subtracting fixed vision encoder cost
        lm_budget = sla_budget_ms - self.config.vision_encoder_ms

        if lm_budget <= 0:
            return {
                "sla_achievable_with_lm_pruning": False,
                "min_lm_tokens_for_sla": None,
                "baseline_tokens": 1548,
                "required_pruning_ratio": None,
                "note": "Vision encoder cost alone exceeds SLA budget.",
            }

        # Solve γ·N² + β·N + (α - lm_budget) = 0 for N
        a = self.config.gamma
        b = self.config.beta
        c = self.config.alpha - lm_budget
        discriminant = b**2 - 4*a*c

        if discriminant < 0 or a == 0:
            # Linear case: N = (lm_budget - alpha) / beta
            n_sla = (lm_budget - self.config.alpha) / self.config.beta if self.config.beta > 0 else None
        else:
            sqrt_disc = float(np.sqrt(discriminant))
            roots = [(-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a)]
            pos_roots = [r for r in roots if r > 0]
            n_sla = min(pos_roots) if pos_roots else None

        sla_achievable = n_sla is not None and n_sla > 0

        return {
            "sla_achievable_with_lm_pruning": sla_achievable,
            "min_lm_tokens_for_sla": int(n_sla) if sla_achievable else None,
            "baseline_tokens": 1548,
            "required_pruning_ratio": round(n_sla / 1548, 3) if sla_achievable else None,
            "lm_budget_ms": round(lm_budget, 1),
            "vision_encoder_ms": self.config.vision_encoder_ms,
            "note": ("LM token pruning CAN reduce LM cost to budget, but end-to-end "
                     "SLA requires also optimising the vision encoder."),
        }
    
    def find_batching_wall(self, memory_budget_mb: float = 8000.0) -> Dict:
        """
        Identify batch size at which KV cache growth hits memory ceiling.
        
        Returns:
            Dict with wall_batch_size and memory breakdown
        """
        # KV cache: Batch × SeqLen × 2 × Layers × HiddenSize × 2 bytes
        # Weights: ~64 MB for SmolVLM
        # Assume seq_len ≈ 1548 + 60 (input + decode)
        
        model_weights_mb = 64.0
        seq_len = 1608
        # KV: 2 (K+V) × seq_len × n_layers × hidden_size × 2 bytes (FP16)
        kv_per_batch_mb = (
            seq_len * 2 * self.config.num_layers * self.config.hidden_size * 2 / 1e6
        )
        
        available_for_kv = memory_budget_mb - model_weights_mb - 200  # reserve OS
        max_batch = max(1, int(available_for_kv / kv_per_batch_mb))
        
        return {
            "memory_budget_mb": memory_budget_mb,
            "model_weights_mb": model_weights_mb,
            "seq_len_assumed": seq_len,
            "kv_cache_per_batch_mb": round(kv_per_batch_mb, 1),
            "estimated_max_batch_size": max_batch,
            "headroom_mb": round(available_for_kv - max_batch * kv_per_batch_mb, 1),
        }


if __name__ == "__main__":
    ROOT = Path(__file__).parent
    model = CostModel(calibration_path=ROOT / "calibration_results.json")

    print("=" * 70)
    print("COST MODEL — CALIBRATED PREDICTIONS")
    print("=" * 70)
    print(f"  γ = {model.config.gamma:.6e}, β = {model.config.beta:.4f}, α = {model.config.alpha:.2f}")
    if model.calibration_r2:
        print(f"  Calibration R² = {model.calibration_r2:.4f}")
    print(f"  Vision encoder (fixed) = {model.config.vision_encoder_ms:.0f}ms")

    # 1. Baseline — no pruning
    r = model.predict_latency(n_visual_tokens=1548)
    print("\n1. Baseline (1548 LM tokens, no pruning):")
    print(f"   T_vision_encoder : {r['t_vision_encoder_ms']:.0f}ms  (fixed — SigLIP crops)")
    print(f"   T_lm_prefill     : {r['t_lm_prefill_ms']:.0f}ms  (LM transformer only)")
    print(f"   T_decode         : {r['t_decode_total_ms']:.0f}ms")
    print(f"   T_total          : {r['t_total_ms']:.0f}ms  (Phase1 measured: 8489ms)")
    print(f"   SLA pass         : {r['sla_pass']}")

    # 2. Aggressive LM pruning (21% of tokens, calibration-derived SLA target)
    r2 = model.predict_latency(n_visual_tokens=1548, pruning_ratio=326/1548)
    print(f"\n2. LM pruned to 326 tokens ({326/1548:.0%} of baseline):")
    print(f"   T_vision_encoder : {r2['t_vision_encoder_ms']:.0f}ms  (unchanged — still needs full image)")
    print(f"   T_lm_prefill     : {r2['t_lm_prefill_ms']:.0f}ms  (was {r['t_lm_prefill_ms']:.0f}ms)")
    print(f"   T_total          : {r2['t_total_ms']:.0f}ms")
    print(f"   SLA pass         : {r2['sla_pass']}")

    # 3. SLA target analysis
    sla = model.find_sla_pruning_target(1548)
    print("\n3. SLA target (T_total ≤ 500ms):")
    if sla.get("blocking_component") == "vision_encoder":
        print(f"   {sla['note']}")
    else:
        print(f"   LM tokens needed  : {sla['target_tokens']}")
        print(f"   Pruning ratio     : {sla['pruning_ratio']} ({sla['compression_ratio']}× compression)")
        print(f"   SLA met           : {sla['sla_met']}")

    # 4. Resolution / pruning wall
    wall = model.find_resolution_wall()
    print("\n4. Resolution wall:")
    print(f"   LM SLA achievable   : {wall['sla_achievable_with_lm_pruning']}")
    print(f"   Min LM tokens       : {wall['min_lm_tokens_for_sla']}")
    if wall.get("note"):
        print(f"   Note: {wall['note']}")

    # 5. Batching wall
    bw = model.find_batching_wall()
    print("\n5. Batching wall (8GB memory):")
    print(f"   Max batch size  : {bw['estimated_max_batch_size']}")
    print(f"   KV cache/batch  : {bw['kv_cache_per_batch_mb']:.1f}MB")
