#!/usr/bin/env python3
"""
Model Calibration — Formal Cost Model
======================================
Predictive equations for T_prefill, T_decode, and total latency.

Two-component latency model (MEASURED, baseline/results_v2.json 2026-07-28):

    T_TTFT = T_vision(c)  +  T_lm_prefill(N)  +  T_decode

Where:
  T_vision(c)       = 553.5·c + 27.4 ms  (MEASURED, near-perfectly linear in
                    crop count c; direct stage isolation on the M3-8GB target)
  T_lm_prefill(N)   = γ·N² + β·N + α  (MEASURED fit; γ=1.170e-3, β=1.2073,
                    α=244.60; domain N ∈ (100, 1560))
  T_decode          ≈ roofline memory-bandwidth bound (modeled)

SUPERSEDED: the old 5991 ms vision "residual" (end-to-end 8489 ms minus a
synthetic 2498 ms LM fit, attributed to a "24 crops" setting that does not
exist — MAX_CROPS is 17) and the synthetic prefill fit (γ=2.096e-5,
β=1.591, α=−20.08, in-sample R²=0.9978 but ~2x low at N=1560).  Both live
in amio_constants for provenance only.

Source of truth for coefficients: amio_constants (MEASURED section).
model_calibration/calibration_results.json holds the SUPERSEDED synthetic
fit and is used only as a fallback if the measured constants are missing.

Measured anchors: TTFT stage sum 897 ms at 1 crop, 14,395 ms at 17 crops.
"""

import json
import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import amio_constants as C

@dataclass
class CostModelConfig:
    """Configuration for the cost model."""

    # ── LM transformer calibration coefficients ───────────────────────────────
    # Source: amio_constants MEASURED fit (baseline/results_v2.json).
    # Valid domain: N ∈ amio_constants.PREFILL_DOMAIN = (100, 1560).
    gamma: float = C.PREFILL_GAMMA   # 1.170e-3 ms/token²  (MEASURED)
    beta:  float = C.PREFILL_BETA    # 1.2073   ms/token
    alpha: float = C.PREFILL_ALPHA   # 244.60   ms (positive intercept)

    # ── Vision stage cost — MEASURED linear model ─────────────────────────────
    # T_vision(c) = vision_ms_per_crop·c + vision_fixed_ms  (full GPU, no
    # contention; per-crop ratios 540–566 across 1–17 crops).  The old 5991 ms
    # residual (amio_constants.VISION_BASE_MS_UNVALIDATED) is superseded and
    # survives only as provenance.
    vision_ms_per_crop: float = C.VISION_MS_PER_CROP  # 553.5 ms/crop (MEASURED)
    vision_fixed_ms:    float = C.VISION_FIXED_MS     # 27.4 ms

    # ── Roofline constants (Apple M3, 8GB) ───────────────────────────────────
    m3_compute_peak_tflops: float = 3.6       # TFLOPS FP16
    m3_bandwidth_gbs: float = C.M3_MEMORY_BW_GBPS  # GB/s unified memory
    ridge_point_flops_per_byte: float = 36.0  # compute_peak / bandwidth

    # ── LM architecture (from checkpoint config.json via amio_constants) ─────
    # hidden 2048 / 24 layers = SmolLM2-1.7B-class LM (the old 1152 figure used
    # elsewhere was the VISION encoder's hidden size and gave a wrong KV size).
    num_layers:  int = C.LM_NUM_LAYERS
    hidden_size: int = C.LM_HIDDEN_SIZE
    num_heads:   int = C.LM_NUM_HEADS
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
        Initialize cost model.

        Coefficient precedence (documented, deliberate):
          1. amio_constants MEASURED fit (already the CostModelConfig defaults)
             — the source of truth.
          2. calibration_results.json — the SUPERSEDED synthetic-embedding fit
             — used ONLY as a fallback when the measured constants are absent
             from amio_constants (e.g. an old checkout).  It is never allowed
             to silently override the measured coefficients.
        The JSON's r2/hidden_size metadata is still recorded for reporting.
        """
        self.config = config or CostModelConfig()
        self.calibration_r2: Optional[float] = None
        self.calibration_hidden_size: Optional[int] = None
        self.coefficients_source: str = "amio_constants (MEASURED)"

        if calibration_path is None:
            # Auto-discover calibration file next to this script
            default = Path(__file__).parent / "calibration_results.json"
            if default.exists():
                calibration_path = default

        if calibration_path and calibration_path.exists():
            with open(calibration_path) as f:
                calib = json.load(f)
            # Metadata only — coefficients stay measured unless missing.
            self.calibration_r2 = calib.get("r2")
            self.calibration_hidden_size = calib.get("hidden_size")
            measured_available = all(
                hasattr(C, name)
                for name in ("PREFILL_GAMMA", "PREFILL_BETA", "PREFILL_ALPHA")
            )
            if not measured_available:
                # Fallback path (superseded synthetic fit) — documented above.
                self.config.gamma = calib["gamma"]
                self.config.beta  = calib["beta"]
                self.config.alpha = calib["alpha"]
                self.coefficients_source = (
                    "calibration_results.json (SUPERSEDED synthetic fallback)"
                )
    
    def predict_t_vision(self, n_crops: int) -> float:
        """
        MEASURED vision tower + connector latency (full GPU, no contention):

            T_vision(c) = 553.5·c + 27.4 ms   (baseline/results_v2.json)

        Crop count is clamped to the processor's valid range [1, MAX_CROPS=17].
        """
        c = max(1, min(int(n_crops), C.MAX_CROPS))
        return self.config.vision_ms_per_crop * c + self.config.vision_fixed_ms

    def crops_for_visual_tokens(self, n_visual_tokens: int) -> int:
        """Crop count implied by an image-token budget (81 tokens/crop),
        clamped to the processor's [1, 17] range."""
        c = math.ceil(max(1, n_visual_tokens) / C.TOKENS_PER_CROP)
        return max(1, min(c, C.MAX_CROPS))

    def predict_t_lm_prefill(self, n_tokens: int) -> float:
        """
        Predict LM-transformer prefill time for N tokens.

        T_lm_prefill(N) = max(0, γ·N² + β·N + α)   [MEASURED fit]

        Valid domain: N ∈ (100, 1560) (amio_constants.PREFILL_DOMAIN);
        values outside are extrapolations.  The measured α = +244.6 ms is
        positive, so the clamp below is a harmless safety net.

        Does NOT include vision stage time (see predict_t_vision — the
        measured 553.5·c + 27.4 ms linear model).
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

        This is a MODELED lower bound (pure bandwidth roofline, no framework
        overhead) — no decode calibration data exists to validate it.

        Args:
            seq_len: Sequence length for KV cache
            token_budget: How many tokens to decode (for estimating load)
            quantization_bits: Precision for weights (4, 8, 16, 32)

        Returns:
            Predicted average TBT in milliseconds per token (clamped ≥ 0)
        """
        # KV cache bytes (FP16 KV): 196,608 B/token from amio_constants
        # (= 2 × 24 layers × 32 KV heads × 64 head_dim × 2 B; the old figure
        # of 110,592 used the vision encoder's hidden size and was wrong).
        kv_bytes = seq_len * C.KV_BYTES_PER_TOKEN_FP16

        # Weight bytes: SmolLM2-1.7B-class LM (see amio_constants — the
        # "500M-param LM" in earlier docs was wrong).  MODELED ASSUMPTION:
        # ~1.7e9 params × 4 B (FP32 reference), scaled by quantization.
        weight_bytes_fp32 = 1.7e9 * 4
        quant_factor = self.config.quantization_savings.get(quantization_bits, 1.0)
        weight_bytes = weight_bytes_fp32 * quant_factor

        # Roofline: T = (KV + W) / Bandwidth
        total_bytes = kv_bytes + weight_bytes
        bandwidth_bytes_per_s = self.config.m3_bandwidth_gbs * 1e9

        t_per_token_s = total_bytes / bandwidth_bytes_per_s
        t_per_token_ms = t_per_token_s * 1e3

        return max(0.0, t_per_token_ms)
    
    def predict_migration_cost(
        self,
        n_full_tokens: int,
        n_pruned_tokens: int,
        migration_depth: int = 3,
    ) -> float:
        """
        ParVTS-style migration cost for token pruning.

        MODELED ASSUMPTION (no calibration data): the first `migration_depth`
        LM layers still process ALL n_full_tokens before the pruned tokens
        are discarded, so the extra cost relative to prefilling only the
        kept tokens is the depth fraction of the full-vs-kept prefill gap,
        plus a small constant token-selection overhead:

            T_migration ≈ (migration_depth / num_layers)
                          × (T_prefill(n_full) − T_prefill(n_kept))
                          + SELECTION_OVERHEAD

        Cost INCREASES with migration depth (the previous formula was
        inverted — deeper migration looked cheaper, making pruning near-free).

        Args:
            n_full_tokens: Original token count
            n_pruned_tokens: After pruning (kept tokens)
            migration_depth: Number of layers that still process all tokens

        Returns:
            Migration latency cost in milliseconds (≥ 0)
        """
        SELECTION_OVERHEAD_MS = 2.0   # modeled constant: saliency scoring/gather

        if n_full_tokens <= 0 or n_pruned_tokens >= n_full_tokens:
            return 0.0

        depth_frac = min(1.0, max(0.0, migration_depth / self.config.num_layers))
        prefill_gap_ms = max(
            0.0,
            self.predict_t_lm_prefill(n_full_tokens)
            - self.predict_t_lm_prefill(n_pruned_tokens),
        )
        return depth_frac * prefill_gap_ms + SELECTION_OVERHEAD_MS
    
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
            Dict with t_prefill, t_decode, t_total, and component breakdown.
            All component latencies are clamped ≥ 0.

        NOTE: the batch overhead factor 1 + 0.1×(B−1) below is an
        UNCALIBRATED ASSUMPTION — no measurement backs the 10%-per-request
        figure; treat batched predictions as illustrative only.
        """
        # Apply pruning
        n_tokens_after_prune = max(int(n_visual_tokens * pruning_ratio), 32)

        # ── Component 1: Vision encoder (MEASURED linear in crop count) ──────
        # Pruning happens AFTER the encoder, so vision cost is set by the
        # UNpruned visual-token budget (crops = ceil(tokens / 81), ≤ 17).
        n_crops = self.crops_for_visual_tokens(n_visual_tokens)
        t_vision_encoder = self.predict_t_vision(n_crops)

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

        # UNCALIBRATED ASSUMPTION: +10% cost per extra request in the batch.
        batch_overhead = 1.0 + 0.1 * (batch_size - 1)
        t_prefill_total = (t_vision_encoder + t_lm_prefill + t_migration) * batch_overhead
        t_total = max(0.0, t_prefill_total + t_decode_total * batch_overhead)

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
            "n_crops":               n_crops,
            "sla_pass":              t_total <= C.TTFT_SLA_MS,
            "note": ("Vision stage (measured 553.5·c + 27.4 ms) dominates; "
                     "token pruning reduces the LM component only. Batch "
                     "overhead factor is an uncalibrated assumption."),
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
        # First check if the vision encoder alone exceeds the SLA budget.
        # The request's crop count is fixed by its visual-token budget, so the
        # relevant vision cost is T_vision(crops(n_visual_tokens)); even the
        # MINIMUM 1-crop cost (≈581 ms measured) exceeds the 500 ms SLA.
        n_crops = self.crops_for_visual_tokens(n_visual_tokens)
        t_vision = self.predict_t_vision(n_crops)
        if t_vision >= sla_budget_ms:
            return {
                "target_tokens": None,
                "pruning_ratio": None,
                "compression_ratio": None,
                "predicted_t_total_ms": None,
                "sla_budget_ms": sla_budget_ms,
                "sla_met": False,
                "blocking_component": "vision_encoder",
                "note": (
                    f"Vision encoder ({t_vision:.0f}ms at {n_crops} crops; "
                    f"measured minimum {self.predict_t_vision(1):.0f}ms at 1 crop) "
                    f"alone exceeds SLA ({sla_budget_ms:.0f}ms). LM token pruning "
                    "cannot compensate; vision encoder optimization required."
                ),
            }

        # Binary search: find max N_lm such that full TTFT ≤ SLA
        lm_budget = sla_budget_ms - t_vision
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
        Identify whether ANY crop setting can meet the SLA.

        Processor crop settings are {1, 5, 10, 17} (size.longest_edge
        {384, 768, 1152, 1536}); total input tokens per setting are
        amio_constants.TOKENS_PER_CONFIG ({1:100, 5:466, 10:922, 17:1560}).

        Returns:
            Dict with feasibility analysis.  Under the measured numbers even
            the 1-crop vision stage (≈581 ms) exceeds a 500 ms budget, so the
            honest answer is "not achievable at any crop setting".
        """
        # LM budget after subtracting the MINIMUM (1-crop) vision cost
        min_vision_ms = self.predict_t_vision(1)
        lm_budget = sla_budget_ms - min_vision_ms

        if lm_budget <= 0:
            return {
                "sla_achievable_with_lm_pruning": False,
                "min_lm_tokens_for_sla": None,
                "baseline_tokens": C.TOKENS_PER_CONFIG[C.MAX_CROPS],  # 1560
                "required_pruning_ratio": None,
                "min_vision_ms": round(min_vision_ms, 1),
                "note": (
                    f"Vision encoder cost alone ({min_vision_ms:.0f}ms at the "
                    f"minimum 1-crop setting) exceeds the {sla_budget_ms:.0f}ms "
                    "SLA budget — infeasible at every crop setting."
                ),
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
        baseline_tokens = C.TOKENS_PER_CONFIG[C.MAX_CROPS]   # 1560

        return {
            "sla_achievable_with_lm_pruning": sla_achievable,
            "min_lm_tokens_for_sla": int(n_sla) if sla_achievable else None,
            "baseline_tokens": baseline_tokens,
            "required_pruning_ratio": round(n_sla / baseline_tokens, 3) if sla_achievable else None,
            "lm_budget_ms": round(lm_budget, 1),
            "min_vision_ms": round(min_vision_ms, 1),
            "note": ("LM token pruning CAN reduce LM cost to budget, but end-to-end "
                     "SLA requires also optimising the vision encoder."),
        }
    
    def find_batching_wall(self, memory_budget_mb: float = 8000.0) -> Dict:
        """
        Identify batch size at which KV cache growth hits memory ceiling.
        
        Returns:
            Dict with wall_batch_size and memory breakdown
        """
        # KV per sequence from amio_constants (196,608 B/token FP16)
        # Weights: measured checkpoint size (amio_constants.MODEL_WEIGHTS_MB)
        # Assume seq_len ≈ 1560 + 60 (measured 17-crop input total + decode)

        model_weights_mb = C.MODEL_WEIGHTS_MB
        seq_len = C.TOKENS_PER_CONFIG[C.MAX_CROPS] + 60   # 1620
        kv_per_batch_mb = seq_len * C.KV_BYTES_PER_TOKEN_FP16 / (1024 ** 2)

        available_for_kv = memory_budget_mb - model_weights_mb - C.OS_RESERVE_MB
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
    print("COST MODEL — MEASURED PREDICTIONS (baseline/results_v2.json)")
    print("=" * 70)
    print(f"  γ = {model.config.gamma:.6e}, β = {model.config.beta:.4f}, α = {model.config.alpha:.2f}")
    print(f"  Coefficients source: {model.coefficients_source}")
    print(f"  Vision stage (MEASURED): T_vision(c) = "
          f"{model.config.vision_ms_per_crop:.1f}·c + {model.config.vision_fixed_ms:.1f} ms")

    N_MAX = C.TOKENS_PER_CONFIG[C.MAX_CROPS]   # 1560 total input tokens @ 17 crops
    IMG_MAX = C.TOKENS_PER_CROP * C.MAX_CROPS  # 1377 image tokens @ 17 crops

    # 1. Baseline — 17-crop default config, no pruning
    r = model.predict_latency(n_visual_tokens=N_MAX)
    print(f"\n1. Baseline (17 crops, {N_MAX} total input tokens, no pruning):")
    print(f"   T_vision_encoder : {r['t_vision_encoder_ms']:.0f}ms  "
          f"(measured mean 9412ms at 17 crops)")
    print(f"   T_lm_prefill     : {r['t_lm_prefill_ms']:.0f}ms  "
          f"(measured mean 4983ms at N=1560)")
    print(f"   T_decode         : {r['t_decode_total_ms']:.0f}ms")
    print(f"   T_total          : {r['t_total_ms']:.0f}ms  "
          f"(measured TTFT stage sum at 17 crops: {C.TTFT_MS_MEASURED_17_CROP:.0f}ms)")
    print(f"   SLA pass         : {r['sla_pass']}")

    # 2. Aggressive LM pruning (~21% of tokens kept)
    r2 = model.predict_latency(n_visual_tokens=N_MAX, pruning_ratio=326/N_MAX)
    print(f"\n2. LM pruned to 326 tokens ({326/N_MAX:.0%} of baseline):")
    print(f"   T_vision_encoder : {r2['t_vision_encoder_ms']:.0f}ms  (unchanged — still needs full image)")
    print(f"   T_lm_prefill     : {r2['t_lm_prefill_ms']:.0f}ms  (was {r['t_lm_prefill_ms']:.0f}ms)")
    print(f"   T_total          : {r2['t_total_ms']:.0f}ms")
    print(f"   SLA pass         : {r2['sla_pass']}")

    # 3. SLA target analysis (expected: vision-blocked — measured 1-crop
    #    vision ≈ 581 ms already exceeds the 500 ms budget)
    sla = model.find_sla_pruning_target(N_MAX)
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
