"""Single source of truth for model architecture and hardware constants.

Every value here is either (a) read directly from the shipped checkpoint's
config.json / safetensors, (b) a documented measurement with provenance, or
(c) an explicit modeling assumption flagged as such. Simulation modules must
import from here instead of redefining their own copies — the previous
per-module constants disagreed with each other (hidden 1152 vs 2048, pool
5694 vs 5742 MB, KV/token 110,592 vs 196,608, four decode baselines).

Checkpoint: mlx-community/SmolVLM-Instruct-4bit
  (Idefics3ForConditionalGeneration; verified from the HF cache config.json)

NOTE the model-identity correction: the LM is SmolLM2-1.7B-class
(hidden 2048, 24 layers), NOT a "500M-param LM" as earlier docs claimed.
"""

# ---------------------------------------------------------------------------
# Language model (text_config from checkpoint config.json)
# ---------------------------------------------------------------------------
LM_NUM_LAYERS = 24
LM_HIDDEN_SIZE = 2048
LM_NUM_HEADS = 32
LM_NUM_KV_HEADS = 32          # MHA — this checkpoint has no GQA
LM_HEAD_DIM = 64              # hidden / heads

# KV cache bytes per token: 2 (K+V) x layers x kv_heads x head_dim x dtype
KV_BYTES_PER_TOKEN_FP16 = 2 * LM_NUM_LAYERS * LM_NUM_KV_HEADS * LM_HEAD_DIM * 2
assert KV_BYTES_PER_TOKEN_FP16 == 196_608
KV_BYTES_PER_TOKEN_W8 = KV_BYTES_PER_TOKEN_FP16 // 2   # 98,304 (modeled)
KV_BYTES_PER_TOKEN_W4 = KV_BYTES_PER_TOKEN_FP16 // 4   # 49,152 (modeled)

# ---------------------------------------------------------------------------
# Vision encoder (vision_config from checkpoint config.json)
# ---------------------------------------------------------------------------
VISION_NUM_LAYERS = 27        # SigLIP-SO400M
VISION_HIDDEN_SIZE = 1152
VISION_IMAGE_SIZE = 384       # px per crop
VISION_PATCH_SIZE = 14
# (384/14)^2 = 729 patches, pixel-shuffle scale_factor 3 -> 729/9
TOKENS_PER_CROP = 81          # from config.json image_seq_len

# ---------------------------------------------------------------------------
# Memory accounting (measured)
# ---------------------------------------------------------------------------
# stat on the checkpoint's model.safetensors blob: 1,457,817,543 bytes.
MODEL_WEIGHTS_MB = 1390.0     # MiB, 4-bit quantized, vision + LM combined
OS_RESERVE_MB = 2048.0        # modeling assumption: macOS + runtime overhead
TOTAL_MEMORY_MB = 8192.0      # Apple M3 base, 8 GB unified memory
KV_POOL_BUDGET_MB = TOTAL_MEMORY_MB - MODEL_WEIGHTS_MB - OS_RESERVE_MB  # 4754

# ---------------------------------------------------------------------------
# Hardware (Apple M3 base)
# ---------------------------------------------------------------------------
M3_GPU_CORES = 10             # actual GPU core count of the M3 base die
M3_MEMORY_BW_GBPS = 100.0     # unified-memory bandwidth
# The scheduler models compute allocation in abstract "shares" (historically
# mislabeled "SMs" — an NVIDIA term; M3 has 10 GPU cores and Metal exposes no
# per-task core partitioning). Shares express *modeled* fractional compute
# priority, not hardware partitions.
TOTAL_COMPUTE_SHARES = 38

# ---------------------------------------------------------------------------
# MEASURED stage models (baseline/results_v2.json, 2026-07-28)
# Method: direct stage isolation with mx.eval on outputs, 1 warm-up pass per
# shape, 5 trials per config, real image embeddings (not synthetic), on the
# actual M3-8GB target. Crop count driven via the Idefics3 processor
# (do_image_splitting / size.longest_edge). These SUPERSEDE the synthetic
# Phase-2 calibration below.
# ---------------------------------------------------------------------------
# Vision tower + connector: near-perfectly linear in crop count.
#   T_vision(c) = VISION_MS_PER_CROP * c + VISION_FIXED_MS
VISION_MS_PER_CROP = 553.5      # MEASURED (per-crop ratios 540-566 across 1-17)
VISION_FIXED_MS = 27.4
MAX_CROPS = 17                  # processor maximum (4x4 grid + global view).
                                # The "24 crops" setting in earlier docs does
                                # not exist in this pipeline.
# Serving-policy map: processor size.longest_edge -> crop count (measured).
CROP_SETTINGS = {384: 1, 768: 5, 1152: 10, 1536: 17}
TOKENS_PER_CONFIG = {1: 100, 5: 466, 10: 922, 17: 1560}  # total input tokens

# LM prefill quadratic, fit to the 4 measured points (in-sample, no held-out
# yet — more N values needed for real validation):
#   T_prefill(N) = GAMMA*N^2 + BETA*N + ALPHA   [ms]
PREFILL_GAMMA = 1.170e-3        # MEASURED fit; 56x the synthetic-embedding fit
PREFILL_BETA = 1.2073
PREFILL_ALPHA = 244.60          # positive intercept (physically sensible)
PREFILL_DOMAIN = (100, 1560)

# Decode TBT at batch=1 (per-token timestamps over 160 tokens/config):
#   TBT(ctx) = DECODE_OVERHEAD_MS + DECODE_KV_MS_PER_CTX_TOKEN * ctx   [ms]
# Batch scaling is UNMEASURED; the theoretical per-extra-sequence KV-read
# cost at 1548 ctx is ~3.0 ms (196,608 B/tok * 1548 / 100 GB/s).
DECODE_OVERHEAD_MS_MEASURED = 18.33
DECODE_KV_MS_PER_CTX_TOKEN = 0.00320

# Measured TTFT anchors (stage sums; the end-to-end user path adds CPU
# preprocessing + framework overhead — measured ~20.0 s at 17 crops vs
# 14.4 s stage sum. stream_generate ignores per-call processor kwargs, so
# only the 17-crop end-to-end cross-check is valid.)
TTFT_MS_MEASURED_1_CROP = 897.0     # 500 ms SLA is NOT met even at 1 crop
TTFT_MS_MEASURED_17_CROP = 14395.1

# ---------------------------------------------------------------------------
# SUPERSEDED: synthetic-embedding Phase-2 calibration (model_calibration/).
# Fit on 7 points, N in [128, 1548], IN-SAMPLE R^2 = 0.9978 — but it
# underestimates real prefill ~2x at N=1560 because the synthetic-embedding
# path is not timing-equivalent to real multimodal prefill.
# ---------------------------------------------------------------------------
PREFILL_GAMMA_SYNTHETIC = 2.096e-5
PREFILL_BETA_SYNTHETIC = 1.591
PREFILL_ALPHA_SYNTHETIC = -20.08

# ---------------------------------------------------------------------------
# Stage baselines — provenance-annotated. Values marked UNVALIDATED are
# residuals or assumptions pending the corrected measurement campaign
# (baseline/run_experiments.py v2).
# ---------------------------------------------------------------------------
# SUPERSEDED RESIDUAL: 8489 (old end-to-end prefill) minus 2498 (synthetic
# cost-model LM prefill) — never directly measured (lazy-eval misattribution;
# the old script's argument-less mx.eval() synchronized nothing). The
# direction was right (vision dominates) but the magnitude and the "24 crops"
# attribution were wrong; use VISION_MS_PER_CROP / MAX_CROPS instead.
VISION_BASE_MS_UNVALIDATED = 5991.0
VISION_BASE_CROPS = 24        # crop count the residual was attributed to

# Old single-trial end-to-end TTFT (baseline/results.json, no warm-up
# control); superseded by TTFT_MS_MEASURED_17_CROP.
BASELINE_TTFT_MS_MEASURED = 8617.81

# SUPERSEDED decode TBT model (Phase 5): TBT(B) = OVERHEAD + BW_COST*B [ms].
# No calibration data ever existed for these; measured batch=1 TBT is
# 17-25 ms (see DECODE_OVERHEAD_MS_MEASURED), not 87.7 ms.
DECODE_OVERHEAD_MS = 83.75
DECODE_BW_COST_MS = 3.95

# ---------------------------------------------------------------------------
# SLA targets
# ---------------------------------------------------------------------------
TTFT_SLA_MS = 500.0
TBT_SLA_MS = 80.0             # NOTE: unmeetable under the current TBT model
                              # (TBT(1)=87.7 ms); reported honestly, not hidden.
