"""
W4A8 Quantizer — Dual-Precision Weight / Activation Quantization (Phase 4)

MEASURED BASELINE (baseline/results_v2.json): decode on the shipped
mlx-community/SmolVLM-Instruct-4bit checkpoint (weights ALREADY 4-bit,
i.e. W4A16) measures TBT(ctx) = 18.33 + 0.00320·ctx ms at batch 1 —
≈23.3 ms at 1548 ctx (raw measured mean 24.7 ms).  The old 217.7 ms
single-trial figure is SUPERSEDED.  All gains in this module are reported
relative to the measured W4A16 reference; the FP16 column is a MODELED
HYPOTHETICAL (what an FP16-weight variant would cost under the same
roofline model), clearly labeled as such.

ROOFLINE HONESTY: streaming the 1390 MiB W4 weights takes ~14.6 ms of the
~23.3 ms step; the remaining ~8.7 ms is fixed framework/attention overhead
plus the ctx-proportional KV read.  Weights are already at 4 bits — the
only remaining byte-reduction levers (KV quant, activation quant) touch a
small slice of the step, so quantisation gains are structurally tiny
(computed below, typically ≈1.0x vs W4A16).  Decode's non-GEMM overhead
share is computed and printed, not asserted.

NO FP8 ON M3: the previous "7.2 TFLOPS FP8 on AMX" claim was fabricated —
M3 has no FP8 GEMM path and MLX exposes none.  The activation-speedup knob
is retained only as a clearly-labeled HYPOTHETICAL INT8-accelerated path
(not available in MLX on M3) and is applied ONLY to the compute-bound term
of a proper roofline max(bytes/BW, flops/FLOPS) — never to bandwidth time.

Decode step model (shared with simulation/sm_orchestrator.py):

    TBT(b) = overhead + gemm_roofline(scheme) × (1 + dequant)
             + (b − 1) × (seq_len × kv_bytes/token / BW)

where `overhead` is the residual of the MEASURED W4A16 baseline after its
modeled weight-stream time (≈8.7 ms of the 23.3 ms step), and batching
shares the per-step weight read (the batch term is a modeled assumption).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Allow sibling / parent imports when run as a script
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import amio_constants
from simulation.sm_orchestrator import (
    kv_read_ms_per_seq,
    KV_BYTES_PER_TOKEN_FP16,
    KV_BYTES_PER_TOKEN_W8,
)

# ---------------------------------------------------------------------------
# M3 hardware constants
# ---------------------------------------------------------------------------
M3_BW_GBps: float = amio_constants.M3_MEMORY_BW_GBPS   # 100 GB/s
M3_FLOPS_FP16: float = 3.6e12        # FP16 TFLOPS (3.6 TFLOPS)
RIDGE_POINT_FP16: float = M3_FLOPS_FP16 / (M3_BW_GBps * 1e9)   # 36 FLOP/byte
# NOTE: no FP8 ridge point — M3 has no FP8 GEMM path.

# Measured checkpoint: 1390 MiB, 4-bit, vision + LM combined (amio_constants)
WEIGHT_BYTES_W4: float = amio_constants.MODEL_WEIGHTS_MB * 1024 ** 2
# LM is SmolLM2-1.7B-class (see amio_constants model-identity correction);
# used only for the FLOP term of the roofline (2 FLOPs/param/token).
LM_PARAMS: int = 1_700_000_000
BITS_FP16: int = 16
BITS_W4: int = 4
BITS_A8: int = 8

# MEASURED decode baseline (baseline/results_v2.json), collected on the
# ALREADY 4-bit checkpoint — i.e. this is the W4A16 reference, NOT an FP16
# baseline.  Evaluated from the batch-1 fit at the 1548-ctx reference
# context: 18.33 + 0.00320·1548 ≈ 23.3 ms (raw measured mean 24.7 ms).
# The old single-trial 217.7 ms / 1926.5 ms spike figures are SUPERSEDED.
BASELINE_CTX_TOKENS: int = 1548
BASELINE_TBT_MS: float = (
    amio_constants.DECODE_OVERHEAD_MS_MEASURED
    + amio_constants.DECODE_KV_MS_PER_CTX_TOKEN * BASELINE_CTX_TOKENS
)
TBT_HUMAN_THRESHOLD_MS: float = amio_constants.TBT_SLA_MS  # 80 ms


# ---------------------------------------------------------------------------
# Quantisation configuration
# ---------------------------------------------------------------------------

@dataclass
class QuantConfig:
    """
    Configuration for one quantisation scheme.

    Attributes
    ----------
    name           : human-readable label
    weight_bits    : precision of stored weights (4 = INT4, 16 = FP16)
    activation_bits: precision of activations during GEMM (8, 16)
    group_size     : number of weights per quantisation group (for W4 group quant)
    use_gar        : enable Group-Aware Reordering (modeling assumption)
    dequant_overhead_pct : extra latency overhead from dequantisation (%)
    """
    name: str
    weight_bits: int = 16
    activation_bits: int = 16
    group_size: int = 128
    use_gar: bool = False
    dequant_overhead_pct: float = 0.0

    @property
    def weight_scale(self) -> float:
        """Compression factor vs FP16 weight storage."""
        return BITS_FP16 / self.weight_bits

    @property
    def activation_throughput_scale(self) -> float:
        """
        HYPOTHETICAL INT8-accelerated compute path (NOT available in MLX on
        M3 — the earlier 'FP8 on AMX' claim was fabricated).  Applied ONLY
        to the compute-bound term of the roofline, never to bandwidth time.
        """
        if self.activation_bits <= 8:
            return 2.0
        return 1.0

    @property
    def kv_bytes_per_tok(self) -> int:
        """Modeled KV bytes/token: 8-bit KV when activations are 8-bit."""
        if self.activation_bits <= 8:
            return KV_BYTES_PER_TOKEN_W8
        return KV_BYTES_PER_TOKEN_FP16


SCHEMES: dict[str, QuantConfig] = {
    "fp16":    QuantConfig("FP16 (modeled)",        weight_bits=16, activation_bits=16),
    "w8a16":   QuantConfig("W8A16 (modeled)",       weight_bits=8,  activation_bits=16),
    "w4a16":   QuantConfig("W4A16 (= baseline)",    weight_bits=4,  activation_bits=16, dequant_overhead_pct=8.0),
    "w4a8":    QuantConfig("W4A8 (hyp. INT8)",      weight_bits=4,  activation_bits=8,  use_gar=True, dequant_overhead_pct=6.0),
    "w4a8_gar":QuantConfig("W4A8+GAR (hyp. INT8)",  weight_bits=4,  activation_bits=8,  use_gar=True, dequant_overhead_pct=4.0),
}

# The measured ≈23.3 ms baseline corresponds to THIS scheme (shipped 4-bit
# checkpoint, FP16 activations).  All gains are reported relative to it.
BASELINE_SCHEME_KEY: str = "w4a16"


# ---------------------------------------------------------------------------
# Quantisation analysis
# ---------------------------------------------------------------------------

@dataclass
class QuantAnalysisResult:
    """Full analysis of one quantisation scheme applied to the model."""
    config: QuantConfig

    # Memory
    weight_bytes_fp16: int      # modeled hypothetical FP16 footprint
    weight_bytes_quantized: int
    memory_reduction_x: float   # vs the modeled FP16 footprint

    # Bandwidth
    bw_required_GBps: float     # weight-stream bandwidth at predicted TBT
    bw_utilization_pct: float   # bw_required / available (the honest number)
    bw_bound: bool              # True only if weight streaming dominates the
                                # step (> 50% of TBT) — decode here is NOT

    # Throughput predictions
    tbt_theoretical_ms: float   # overhead + roofline GEMM (no dequant)
    tbt_predicted_ms: float     # with dequant overhead
    tbt_gain_vs_w4a16: float    # speedup vs the MEASURED W4A16 baseline
    tbt_gain_vs_fp16: float     # speedup vs the MODELED FP16 hypothetical

    # SLA assessment
    sla_tbt_pass: bool          # tbt_predicted_ms ≤ 80 ms?
    tbt_at_batch4_ms: float     # shared batching model (weights read once)
    max_batch_below_threshold: int  # 0 = threshold unmeetable at any batch

    # GAR note
    gar_note: str = ""


class W4A8Analyzer:
    """
    Analytical predictor for quantisation impact on SmolVLM decode.

    Model (per decode step):
        gemm_ms(scheme) = max(weight_bytes / BW, flops / FLOPS)   [roofline]
        TBT(scheme)     = overhead + gemm_ms × (1 + dequant_pct)

    `overhead` is anchored to the MEASURED W4A16 baseline: it is whatever
    the ≈23.3 ms step (18.33 + 0.00320·1548, measured batch-1 fit) spends
    outside the modeled weight stream (≈8.7 ms — framework overhead,
    attention, KV read).  That residual is untouched by weight-byte
    reduction, and the weights are already 4-bit, so further quantisation
    gains are structurally tiny — decode has no weight bytes left to shed.
    """

    def __init__(
        self,
        lm_params: int = LM_PARAMS,
        bandwidth_GBps: float = M3_BW_GBps,
        baseline_tbt_ms: float = BASELINE_TBT_MS,
    ):
        self.lm_params = lm_params
        self.bandwidth_GBps = bandwidth_GBps
        self.baseline_tbt_ms = baseline_tbt_ms
        # Modeled FP16 footprint = 4× the measured 4-bit blob
        self._weight_bytes_fp16 = int(WEIGHT_BYTES_W4 * 4)

        # Anchor the non-GEMM overhead so that the W4A16 scheme reproduces
        # the measured baseline exactly.
        ref = SCHEMES[BASELINE_SCHEME_KEY]
        ref_gemm_ms = self._gemm_ms(ref) * (1.0 + ref.dequant_overhead_pct / 100.0)
        self.overhead_ms = self.baseline_tbt_ms - ref_gemm_ms

    # -- roofline terms ---------------------------------------------------

    def _weight_bytes(self, config: QuantConfig) -> float:
        """Scale the MEASURED 4-bit blob to the scheme's weight width."""
        return WEIGHT_BYTES_W4 * config.weight_bits / 4.0

    def _stream_ms(self, config: QuantConfig) -> float:
        """Weight-stream time per step (bandwidth term)."""
        return self._weight_bytes(config) / (self.bandwidth_GBps * 1e9) * 1000.0

    def _compute_ms(self, config: QuantConfig) -> float:
        """
        GEMM FLOP time per token: ~2 FLOPs per LM parameter.  The
        activation_throughput_scale (hypothetical INT8 path, not available
        in MLX on M3) applies here and ONLY here.
        """
        flops = 2.0 * self.lm_params
        return flops / (M3_FLOPS_FP16 * config.activation_throughput_scale) * 1000.0

    def _gemm_ms(self, config: QuantConfig) -> float:
        """Roofline: a kernel is limited by the slower of bytes and flops."""
        return max(self._stream_ms(config), self._compute_ms(config))

    # -- analysis ----------------------------------------------------------

    def analyze(self, config: QuantConfig) -> QuantAnalysisResult:
        """Run full analysis for one QuantConfig (gains vs W4A16 baseline)."""
        weight_bytes_q = int(self._weight_bytes(config))
        memory_reduction = self._weight_bytes_fp16 / weight_bytes_q

        gemm_ms = self._gemm_ms(config)
        tbt_theoretical_ms = self.overhead_ms + gemm_ms
        overhead_factor = 1.0 + config.dequant_overhead_pct / 100.0
        tbt_predicted_ms = self.overhead_ms + gemm_ms * overhead_factor

        # Gains: vs the MEASURED baseline (w4a16) and vs the MODELED FP16
        gain_vs_w4a16 = self.baseline_tbt_ms / tbt_predicted_ms
        fp16_cfg = SCHEMES["fp16"]
        tbt_fp16_modeled = self.overhead_ms + self._gemm_ms(fp16_cfg)
        gain_vs_fp16 = tbt_fp16_modeled / tbt_predicted_ms

        # Roofline honesty: actual bandwidth utilization at this TBT
        stream_ms = self._stream_ms(config)
        bw_required = weight_bytes_q / (tbt_predicted_ms / 1000.0) / 1e9
        bw_utilization_pct = bw_required / self.bandwidth_GBps * 100.0
        # Fixed semantics (previously inverted): bandwidth-bound only if the
        # weight stream dominates the step.
        bw_bound = stream_ms > 0.5 * tbt_predicted_ms

        sla_pass = tbt_predicted_ms <= TBT_HUMAN_THRESHOLD_MS

        # Batching (shared model with sm_orchestrator; MODELED — batch
        # scaling unmeasured): weights are read once per step; each extra
        # sequence adds only its KV reads (≈3.0 ms/seq at 1548 ctx FP16).
        kv_ms = kv_read_ms_per_seq(BASELINE_CTX_TOKENS, config.kv_bytes_per_tok)
        tbt_batch4 = tbt_predicted_ms + 3 * kv_ms

        # Largest batch with TBT(b) ≤ threshold; 0 = unmeetable even at b=1
        if tbt_predicted_ms > TBT_HUMAN_THRESHOLD_MS:
            max_batch = 0
        else:
            max_batch = 1 + int(
                (TBT_HUMAN_THRESHOLD_MS - tbt_predicted_ms) / kv_ms
            )

        gar_note = ""
        if config.use_gar:
            gar_note = (
                "GAR ('Group-Aware Reordering') is a MODELING ASSUMPTION of "
                "this study, not a published method; its accuracy figures "
                "are illustrative, with no literature citation."
            )

        return QuantAnalysisResult(
            config=config,
            weight_bytes_fp16=self._weight_bytes_fp16,
            weight_bytes_quantized=weight_bytes_q,
            memory_reduction_x=round(memory_reduction, 2),
            bw_required_GBps=round(bw_required, 2),
            bw_utilization_pct=round(bw_utilization_pct, 1),
            bw_bound=bw_bound,
            tbt_theoretical_ms=round(tbt_theoretical_ms, 2),
            tbt_predicted_ms=round(tbt_predicted_ms, 2),
            tbt_gain_vs_w4a16=round(gain_vs_w4a16, 3),
            tbt_gain_vs_fp16=round(gain_vs_fp16, 2),
            sla_tbt_pass=sla_pass,
            tbt_at_batch4_ms=round(tbt_batch4, 2),
            max_batch_below_threshold=max_batch,
            gar_note=gar_note,
        )

    def compare_all(self) -> dict[str, QuantAnalysisResult]:
        """Analyze all built-in quantisation schemes."""
        return {key: self.analyze(cfg) for key, cfg in SCHEMES.items()}


# ---------------------------------------------------------------------------
# GAR weight reorder simulator (analytical)
# ---------------------------------------------------------------------------

@dataclass
class GARConfig:
    """Parameters for Group-Aware Reordering."""
    group_size: int = 128        # weights per quantisation group
    n_groups_per_layer: int = 0  # auto-computed from model
    hessian_percentile: float = 0.95  # top-95% salient weights kept at higher precision


class GARAnalyzer:
    """
    Models an offline "Group-Aware Reordering" (GAR) pass intended to limit
    accuracy loss from W4 + A8 double quantisation.

    HONESTY NOTE: "GAR" as described here matches no published method, and
    the accuracy numbers below are NOT from QuaRot/SpinQuant (those are
    rotation-based techniques and were previously miscited).  Everything in
    this class is an ILLUSTRATIVE MODELING ASSUMPTION with no literature
    citation, pending an actual quantisation-accuracy experiment.

    The modeled mechanism: sort weight groups per layer by a proxy Hessian
    (e.g., squared activation magnitudes), then reorder INT4 packing so the
    most important groups get the numerically best INT4 values.  Offline,
    one-time; zero inference-time overhead is the modeling claim.
    """

    def __init__(
        self,
        model_params: int = LM_PARAMS,
        group_size: int = 128,
    ):
        self.model_params = model_params
        self.group_size = group_size
        self.n_groups = model_params // group_size

    def accuracy_preservation_estimate(self) -> dict:
        """
        Return accuracy-recovery figures for different reordering strategies.

        THESE VALUES ARE ILLUSTRATIVE MODELING ASSUMPTIONS — they are not
        drawn from QuaRot, SpinQuant, or any published W4A8 result, and no
        citation supports them.
        """
        return {
            "w4a8_no_gar": {
                "ppl_degradation_pct": 8.5,
                "notes": "Illustrative assumption (no citation): naive W4A8 loss"
            },
            "w4a8_with_gar": {
                "ppl_degradation_pct": 1.2,
                "notes": "Illustrative assumption (no citation): GAR recovery"
            },
            "w4a16_with_gar": {
                "ppl_degradation_pct": 0.5,
                "notes": "Illustrative assumption (no citation): near-FP16"
            },
        }

    def overhead_analysis(self) -> dict:
        """
        Quantify the offline GAR pass cost (one-time, not per-inference).
        """
        # Hessian proxy: forward pass through calibration dataset
        calibration_samples = 512
        forward_pass_ms = 2500.0   # approx from Phase 2 calibration
        hessian_compute_ms = calibration_samples * forward_pass_ms
        # Reorder: O(N_groups × log N_groups) argsort
        reorder_ms = self.n_groups * math.log2(max(self.n_groups, 1)) * 0.001
        return {
            "hessian_forward_passes": calibration_samples,
            "hessian_compute_s": hessian_compute_ms / 1000.0,
            "reorder_sort_ms": round(reorder_ms, 2),
            "total_gar_pass_s": round((hessian_compute_ms + reorder_ms) / 1000.0, 1),
            "inference_overhead_ms": 0.0,
            "notes": "Modeled one-time offline pass; zero per-token latency (assumption)",
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("AMIO Phase 4 — W4A8 Quantiser Analysis (corrected W4A16 baseline)")
    print("=" * 78)

    analyzer = W4A8Analyzer()

    print(f"\nCheckpoint: SmolVLM-Instruct-4bit — weights ALREADY 4-bit "
          f"({WEIGHT_BYTES_W4/1e6:.0f} MB measured blob).")
    print(f"Baseline (MEASURED batch-1 fit, baseline/results_v2.json): W4A16 at "
          f"{BASELINE_TBT_MS:.1f} ms/token ({BASELINE_CTX_TOKENS} ctx; raw "
          f"measured mean 24.7 ms).  FP16 row is a MODELED hypothetical.")
    print(f"M3 bandwidth: {M3_BW_GBps:.0f} GB/s | FP16 ridge point: "
          f"{RIDGE_POINT_FP16:.0f} FLOP/byte | no FP8 path exists on M3")
    print(f"Non-GEMM overhead residual: {analyzer.overhead_ms:.1f} ms/step "
          f"({analyzer.overhead_ms/BASELINE_TBT_MS*100:.0f}% of the step; "
          f"weights are already W4 — no weight bytes left to shed, so "
          f"further quantisation gains are structurally tiny)")
    print()

    results = analyzer.compare_all()
    print(f"  {'Scheme':<20} {'Wt MB':>6} {'Mem⇩':>5}  "
          f"{'BW GB/s':>7} {'BW util':>7}  {'TBT ms':>7}  "
          f"{'vs W4A16':>8} {'vs FP16*':>8}  {'SLA':>4}")
    print("  " + "-" * 84)
    for key, r in results.items():
        flag = "PASS" if r.sla_tbt_pass else "FAIL"
        print(
            f"  {r.config.name:<20} "
            f"{r.weight_bytes_quantized/1e6:>5.0f}  "
            f"{r.memory_reduction_x:>4.1f}x  "
            f"{r.bw_required_GBps:>7.1f} "
            f"{r.bw_utilization_pct:>6.1f}%  "
            f"{r.tbt_predicted_ms:>7.1f}  "
            f"{r.tbt_gain_vs_w4a16:>7.3f}x "
            f"{r.tbt_gain_vs_fp16:>7.2f}x  "
            f"{flag}"
        )
    print("  (* vs FP16 = vs the MODELED FP16 hypothetical, not a measurement)")

    print()
    fp16_r = results["fp16"]
    w4a8_result = results["w4a8_gar"]
    print("  Roofline honesty (computed):")
    print(f"    MODELED FP16 decode would stream "
          f"{fp16_r.bw_required_GBps:.1f} GB/s of {M3_BW_GBps:.0f} GB/s "
          f"({fp16_r.bw_utilization_pct:.1f}% utilization);")
    print(f"    the measured W4 baseline uses {results['w4a16'].bw_utilization_pct:.1f}%. "
          f"With weights already at 4 bits, the step is dominated by")
    print(f"    fixed overhead + the minimal W4 weight stream — the remaining "
          f"quantisation levers touch only "
          f"{100 - analyzer.overhead_ms/BASELINE_TBT_MS*100:.0f}% of the step, "
          f"and W4A8's computed gain vs W4A16 is "
          f"{w4a8_result.tbt_gain_vs_w4a16:.3f}x.")
    print()
    print(f"  W4A8+GAR (hypothetical INT8 path, not available in MLX on M3):")
    print(f"    TBT = {w4a8_result.tbt_predicted_ms:.1f} ms  "
          f"(gain vs measured W4A16 baseline: {w4a8_result.tbt_gain_vs_w4a16:.3f}x)")
    print(f"    TBT at batch 4 (shared batching model): "
          f"{w4a8_result.tbt_at_batch4_ms:.1f} ms")
    print(f"    Max batch at TBT <= {TBT_HUMAN_THRESHOLD_MS:.0f} ms: "
          f"{w4a8_result.max_batch_below_threshold} (0 = unmeetable)")
    print()

    # GAR analysis
    gar = GARAnalyzer()
    acc = gar.accuracy_preservation_estimate()
    overhead = gar.overhead_analysis()
    print("  GAR (illustrative modeling assumptions — no literature citation):")
    for scheme, info in acc.items():
        print(f"    {scheme:<20}: assumed perplexity degradation {info['ppl_degradation_pct']}%")
    print(f"  Offline GAR pass: ~{overhead['total_gar_pass_s']:.0f} s  "
          f"(inference overhead: {overhead['inference_overhead_ms']} ms)")
    print()
    print("W4A8 quantiser analysis complete")
