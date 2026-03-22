#!/usr/bin/env python3
"""
Phase 2 — Formal Cost Model
============================
Predictive equations for T_prefill, T_decode, and total latency.

Based on:
1. Empirical quadratic fit for T_prefill(N_tokens)
2. Roofline model for T_decode (memory-bandwidth bound)
3. ParVTS migration cost for token pruning
4. KV-cache fragmentation and batching effects
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

@dataclass
class CostModelConfig:
    """Configuration for the cost model."""
    # Quadratic prefill coefficients (calibrated in Phase 2)
    gamma: float = 3.64e-6      # ms/token²  (placeholder, calibrated)
    beta: float = -0.21         # ms/token
    alpha: float = 8709.0       # ms (constant)
    
    # Roofline constants
    m3_compute_peak_tflops: float = 3.6      # TFLOPS (FP16)
    m3_bandwidth_gbs: float = 100.0          # GB/s
    ridge_point_flops_per_byte: float = 36.0  # FLOP/B (compute_peak / bandwidth)
    
    # Model architecture
    num_layers: int = 24
    hidden_size: int = 1152
    num_heads: int = 16
    vocab_size: int = 32000
    
    # Quantization savings (how bits reduce memory footprint)
    quantization_savings: Dict[int, float] = None  # bit_width -> fraction of original
    
    # Migration depth (ParVTS)
    migration_depth_default: int = 3  # layers that process full tokens
    
    def __post_init__(self):
        if self.quantization_savings is None:
            self.quantization_savings = {
                32: 1.0,   # FP32 baseline
                16: 0.5,   # FP16 half size
                8: 0.25,   # INT8
                4: 0.125,  # INT4
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
        
        Args:
            config: CostModelConfig with calibrated coefficients
            calibration_path: Path to calibration_results.json from Phase 2 calibration
        """
        self.config = config or CostModelConfig()
        
        # Load calibration if available
        if calibration_path and calibration_path.exists():
            with open(calibration_path) as f:
                calib = json.load(f)
                self.config.gamma = calib.get("gamma", self.config.gamma)
                self.config.beta = calib.get("beta", self.config.beta)
                self.config.alpha = calib.get("alpha", self.config.alpha)
                self.validation_r2 = calib.get("r2", None)
        else:
            self.validation_r2 = None
    
    def predict_t_prefill(self, n_tokens: int) -> float:
        """
        Predict T_prefill for given token count.
        
        T_prefill(N) = γ·N² + β·N + α
        
        Args:
            n_tokens: Number of visual tokens
            
        Returns:
            Predicted prefill latency in milliseconds
        """
        return (
            self.config.gamma * n_tokens**2 +
            self.config.beta * n_tokens +
            self.config.alpha
        )
    
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
        # KV cache bytes for one token generation
        # KV = 2 × seq_len × num_layers × hidden_size × (bits/8)
        kv_bytes = 2 * seq_len * self.config.num_layers * self.config.hidden_size * 2  # FP16
        
        # Weight bytes (quantized)
        quant_factor = self.config.quantization_savings.get(quantization_bits, 1.0)
        # Rough estimate: total weights ≈ 7B params × quant
        weight_bytes = 7e9 * quantization_bits / 8
        
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
        n_tokens_after_prune = int(n_visual_tokens * pruning_ratio)
        n_tokens_after_prune = max(n_tokens_after_prune, 32)  # minimum viable
        
        # Predict components
        t_prefill = self.predict_t_prefill(n_tokens_after_prune)
        t_migration = self.predict_migration_cost(
            n_visual_tokens, n_tokens_after_prune, migration_depth
        )
        
        # Decode: seq_len grows with each token, use average (rough)
        avg_seq_len = (n_tokens_after_prune + n_decode_tokens) // 2
        t_decode_per_tok = self.predict_t_decode_roofline(
            avg_seq_len, n_decode_tokens, quantization_bits
        )
        t_decode_total = t_decode_per_tok * n_decode_tokens
        
        # Batching effect: linear scaling (no fairness overhead at single-GPU)
        batch_overhead = 1.0 + 0.1 * (batch_size - 1)  # 10% per additional request
        
        t_total = (t_prefill + t_migration + t_decode_total) * batch_overhead
        
        return {
            "t_prefill_ms": round(t_prefill, 1),
            "t_migration_ms": round(t_migration, 1),
            "t_decode_total_ms": round(t_decode_total, 1),
            "t_decode_per_token_ms": round(t_decode_per_tok, 2),
            "t_total_ms": round(t_total, 1),
            "n_tokens_effective": n_tokens_after_prune,
            "pruning_ratio": round(pruning_ratio, 3),
            "batch_overhead": round(batch_overhead, 2),
            "sla_pass": t_total <= 500.0,
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
        # Binary search for maximum token count that meets SLA
        low, high = 32, n_visual_tokens
        target_tokens = 32
        
        while low <= high:
            mid = (low + high) // 2
            t_pred = self.predict_latency(
                n_visual_tokens=n_visual_tokens,
                n_decode_tokens=60,
                quantization_bits=quantization_bits,
                pruning_ratio=mid / n_visual_tokens,
            )["t_total_ms"]
            
            if t_pred <= sla_budget_ms:
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
        
        t_prefill_1548 = self.predict_t_prefill(1548)  # This is ~8489ms
        
        # Can we prune below 500ms prefill? Solve γ·N² + β·N + α = 500
        a, b, c = self.config.gamma, self.config.beta, (self.config.alpha - sla_budget_ms)
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            # Can't reach SLA even with no tokens (impossible)
            min_viable_tokens = None
            sla_achievable = False
        else:
            sqrt_disc = np.sqrt(discriminant)
            n_sla_1 = (-b + sqrt_disc) / (2*a)
            n_sla_2 = (-b - sqrt_disc) / (2*a)
            min_viable_tokens = min([n for n in [n_sla_1, n_sla_2] if n > 0])
            sla_achievable = min_viable_tokens > 0
        
        return {
            "sla_achievable_with_pruning": sla_achievable,
            "min_tokens_for_sla": int(min_viable_tokens) if sla_achievable else None,
            "baseline_tokens": 1548,
            "required_pruning_ratio": round(min_viable_tokens / 1548, 3) if sla_achievable else None,
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
    # Quick test
    model = CostModel()
    
    print("=" * 70)
    print("COST MODEL TEST")
    print("=" * 70)
    
    # Test 1: Baseline (no pruning)
    result = model.predict_latency(n_visual_tokens=1548)
    print("\n1. Baseline (1548 tokens, no pruning):")
    print(f"   T_prefill: {result['t_prefill_ms']:.0f}ms")
    print(f"   T_decode: {result['t_decode_total_ms']:.0f}ms")
    print(f"   T_total: {result['t_total_ms']:.0f}ms")
    print(f"   SLA pass: {result['sla_pass']}")
    
    # Test 2: With pruning
    result_pruned = model.predict_latency(n_visual_tokens=1548, pruning_ratio=0.25)
    print("\n2. With 75% pruning (→ ~387 tokens):")
    print(f"   T_prefill: {result_pruned['t_prefill_ms']:.0f}ms")
    print(f"   T_decode: {result_pruned['t_decode_total_ms']:.0f}ms")
    print(f"   T_total: {result_pruned['t_total_ms']:.0f}ms")
    print(f"   SLA pass: {result_pruned['sla_pass']}")
    
    # Test 3: Find SLA target
    sla_target = model.find_sla_pruning_target(1548)
    print("\n3. SLA target (T_total ≤ 500ms):")
    print(f"   Target tokens: {sla_target['target_tokens']}")
    print(f"   Pruning ratio: {sla_target['pruning_ratio']}")
    print(f"   Compression: {sla_target['compression_ratio']}×")
    
    # Test 4: Resolution wall
    wall = model.find_resolution_wall()
    print("\n4. Resolution wall analysis:")
    print(f"   SLA achievable: {wall['sla_achievable_with_pruning']}")
    print(f"   Min tokens for SLA: {wall['min_tokens_for_sla']}")
    
    # Test 5: Batching wall
    batch_wall = model.find_batching_wall()
    print("\n5. Batching wall (8GB memory):")
    print(f"   Max batch size: {batch_wall['estimated_max_batch_size']}")
    print(f"   KV cache/batch: {batch_wall['kv_cache_per_batch_mb']:.1f}MB")
