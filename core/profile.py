"""Per-(model, backend, hardware) calibrated cost profiles.

This is the Tier-3 generalization of amio_constants.py: instead of one
global module of SmolVLM/MLX-specific constants, a `ModelHardwareProfile`
is a portable, loadable record of what was ACTUALLY calibrated for one
(model, backend, hardware) combination. Every field traces back to a real
measurement already in this repo -- nothing here invents new numbers.

Two profiles are loaded from real data gathered so far:
  - SmolVLM-Instruct-4bit on MLX / Apple M3   (amio_constants.py)
  - Qwen2.5-0.5B-Instruct on llama.cpp/Metal / Apple M3
    (model_calibration/llamacpp_prefill_fit.json,
     baseline/results_quantization_tradeoff.json when present)

Techniques (core/technique.py) are written against this interface, not
against either model's specific constants, so the same technique code
can run against any calibrated profile.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class PrefillCostModel:
    """T_prefill(N) = gamma*N^2 + beta*N + alpha, in ms."""
    gamma: float
    beta: float
    alpha: float
    domain_n: Tuple[float, float]
    loocv_mape_pct: Optional[float]
    source: str
    # Worst single held-out LOOCV point, not the average -- see
    # core/TIER4_CLOSED_LOOP_FINDINGS.md. None for profiles calibrated
    # before this field existed (no per-point LOOCV data was retained);
    # PrefillBudgetTechnique falls back to loocv_mape_pct in that case.
    loocv_max_ape_pct: Optional[float] = None

    def predict_ms(self, n_tokens: float) -> float:
        return self.gamma * n_tokens ** 2 + self.beta * n_tokens + self.alpha

    def max_tokens_for_budget(self, budget_ms: float) -> Optional[float]:
        """Solve gamma*N^2 + beta*N + (alpha - budget_ms) = 0 for the
        positive root. Returns None if no positive-N solution exists."""
        if budget_ms <= self.alpha:
            return None
        disc = self.beta ** 2 - 4 * self.gamma * (self.alpha - budget_ms)
        if disc < 0 or self.gamma <= 0:
            return None
        n = (-self.beta + disc ** 0.5) / (2 * self.gamma)
        return n if n > 0 else None


@dataclass(frozen=True)
class DecodeCostModel:
    """TBT(ctx) = overhead_ms + per_ctx_token_ms*ctx, batch=1."""
    overhead_ms: float
    per_ctx_token_ms: float
    source: str

    def predict_ms(self, ctx_tokens: float) -> float:
        return self.overhead_ms + self.per_ctx_token_ms * ctx_tokens


@dataclass(frozen=True)
class QuantLevelStats:
    """Real measured stats for one quantization level of one model."""
    file_size_mb: float
    perplexity: float
    prefill_ms_at_reference_n: float
    reference_n: int
    decode_tbt_ms: float
    gguf_filename: Optional[str] = None


@dataclass(frozen=True)
class ModelHardwareProfile:
    model_id: str
    backend: str      # e.g. "mlx", "llama.cpp"
    hardware: str      # e.g. "Apple M3 (10-core GPU, 8GB unified)"
    prefill: PrefillCostModel
    decode: DecodeCostModel
    # Only populated when multiple quant levels were actually measured for
    # this (model, backend) pair -- absent, not fabricated, otherwise.
    quant_levels: Optional[Dict[str, QuantLevelStats]] = None
    supports_vision: bool = False
    # KV cache bytes/token = 2(K+V) * layers * kv_heads * head_dim * dtype_bytes,
    # derived from the checkpoint's real config.json -- never fabricated.
    kv_bytes_per_token: Optional[float] = None
    bandwidth_gbps: Optional[float] = None
    # The GGUF/weights filename an Executor should load by default (single-
    # quant-level profiles, e.g. auto-calibrated ones). Profiles with
    # multiple measured quant levels instead resolve a filename per level
    # via quant_levels[level].gguf_filename.
    gguf_filename: Optional[str] = None


def load_smolvlm_mlx_profile() -> ModelHardwareProfile:
    """SmolVLM-Instruct-4bit / MLX / Apple M3 -- from amio_constants.py.

    Ships pre-quantized (4-bit) with no alternate quant levels ever
    measured in this repo, so quant_levels is None here, not a guess.
    """
    import amio_constants as C

    return ModelHardwareProfile(
        model_id="mlx-community/SmolVLM-Instruct-4bit",
        backend="mlx",
        hardware="Apple M3 base (10-core GPU, 8GB unified, 100GB/s)",
        prefill=PrefillCostModel(
            gamma=C.PREFILL_GAMMA, beta=C.PREFILL_BETA, alpha=C.PREFILL_ALPHA,
            domain_n=C.PREFILL_DOMAIN, loocv_mape_pct=20.7,
            # Worst single held-out point of the 4-point LOOCV (held-out
            # N=100 -> 59.9% APE), computed one-off from
            # baseline/results_v2.json with the same leave-one-out method
            # as core/calibrate.py's fit_quadratic_with_loocv (not stored
            # in a JSON for this hand-built profile, unlike auto-calibrated
            # ones). Notably much closer to the 41-108% real closed-loop
            # error this profile actually showed than the 20.7% mean was --
            # see core/TIER4_CLOSED_LOOP_FINDINGS.md.
            loocv_max_ape_pct=59.9,
            source="baseline/results_v2.json (4 points)",
        ),
        decode=DecodeCostModel(
            overhead_ms=C.DECODE_OVERHEAD_MS_MEASURED,
            per_ctx_token_ms=C.DECODE_KV_MS_PER_CTX_TOKEN,
            source="baseline/results_v2.json",
        ),
        quant_levels=None,
        supports_vision=True,
        kv_bytes_per_token=C.KV_BYTES_PER_TOKEN_FP16,
        bandwidth_gbps=C.M3_MEMORY_BW_GBPS,
    )


# Qwen2.5-0.5B-Instruct architecture, verified from the real checkpoint
# config.json (huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json),
# not guessed: hidden_size=896, num_attention_heads=14 (head_dim=64),
# num_hidden_layers=24, num_key_value_heads=2 (GQA -- 7x fewer KV heads than
# attention heads). llama.cpp's default KV cache dtype is F16 (2 bytes);
# measure_llamacpp.py / measure_quantization_tradeoff.py did not override
# type_k/type_v, so this is what those measurements actually used.
QWEN25_05B_NUM_LAYERS = 24
QWEN25_05B_NUM_KV_HEADS = 2
QWEN25_05B_HEAD_DIM = 64
QWEN25_05B_KV_DTYPE_BYTES = 2
QWEN25_05B_KV_BYTES_PER_TOKEN = (
    2 * QWEN25_05B_NUM_LAYERS * QWEN25_05B_NUM_KV_HEADS
    * QWEN25_05B_HEAD_DIM * QWEN25_05B_KV_DTYPE_BYTES
)  # = 12,288 B/token -- 16x smaller than SmolVLM's 196,608 (smaller model + GQA)

# GGUF filenames per quant level, as downloaded in baseline/TIER2_LLAMACPP_FINDINGS.md.
# Single source of truth: core/executors/llamacpp_executor.py resolves a
# profile's files through quant_levels[level].gguf_filename / gguf_filename,
# not a module-local constant, so this is the only place these are listed.
QWEN25_05B_QUANT_FILENAMES = {
    "fp16": "qwen2.5-0.5b-instruct-fp16.gguf",
    "q8_0": "qwen2.5-0.5b-instruct-q8_0.gguf",
    "q4_k_m": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
}


def load_qwen_llamacpp_profile() -> ModelHardwareProfile:
    """Qwen2.5-0.5B-Instruct / llama.cpp (Metal) / Apple M3 -- from
    model_calibration/llamacpp_prefill_fit.json and, when present,
    baseline/results_quantization_tradeoff.json."""
    fit_path = ROOT / "model_calibration" / "llamacpp_prefill_fit.json"
    fit = json.loads(fit_path.read_text())

    quant_levels = None
    quant_path = ROOT / "baseline" / "results_quantization_tradeoff.json"
    if quant_path.exists():
        quant_data = json.loads(quant_path.read_text())
        reference_n = 1024
        quant_levels = {
            name: QuantLevelStats(
                file_size_mb=e["file_size_mb"],
                perplexity=e["perplexity"],
                prefill_ms_at_reference_n=e["prefill_by_n_tokens"][str(reference_n)]["mean_ms"],
                reference_n=reference_n,
                decode_tbt_ms=e["decode_tbt"]["mean_ms"],
                gguf_filename=QWEN25_05B_QUANT_FILENAMES.get(name),
            )
            for name, e in quant_data["levels"].items()
        }

    return ModelHardwareProfile(
        model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        backend="llama.cpp",
        hardware="Apple M3 base (10-core GPU, 8GB unified, 100GB/s), Metal offload",
        prefill=PrefillCostModel(
            gamma=fit["gamma"], beta=fit["beta"], alpha=fit["alpha"],
            domain_n=tuple(fit["domain_n"]),
            loocv_mape_pct=fit["loocv_mape_pct"],
            loocv_max_ape_pct=fit.get("loocv_max_ape_pct"),
            source="baseline/results_llamacpp.json (10 points)",
        ),
        decode=DecodeCostModel(
            overhead_ms=fit["decode_tbt_overhead_ms"],
            per_ctx_token_ms=fit["decode_tbt_per_ctx_token_ms"],
            source="baseline/results_llamacpp.json",
        ),
        quant_levels=quant_levels,
        supports_vision=False,
        kv_bytes_per_token=QWEN25_05B_KV_BYTES_PER_TOKEN,
        bandwidth_gbps=100.0,  # same M3 unified-memory bandwidth as the MLX profile
        gguf_filename=QWEN25_05B_QUANT_FILENAMES["q4_k_m"],  # default when no quant_level given
    )


def load_profile_from_calibration_json(path) -> ModelHardwareProfile:
    """Generic loader for the self-describing JSON schema core/calibrate.py
    emits -- no per-model code required. This is the automated counterpart
    to load_qwen_llamacpp_profile()/load_smolvlm_mlx_profile(), which are
    hand-built from this project's own measurement history; this is the
    path for any NEW model calibrated with core/calibrate.py."""
    data = json.loads(Path(path).read_text())
    p = data["prefill"]
    d = data["decode"]
    return ModelHardwareProfile(
        model_id=data["model_id"],
        backend=data["backend"],
        hardware=data["hardware"],
        prefill=PrefillCostModel(
            gamma=p["gamma"], beta=p["beta"], alpha=p["alpha"],
            domain_n=tuple(p["domain_n"]), loocv_mape_pct=p["loocv_mape_pct"],
            loocv_max_ape_pct=p.get("loocv_max_ape_pct"),
            source=str(path),
        ),
        decode=DecodeCostModel(
            overhead_ms=d["overhead_ms"], per_ctx_token_ms=d["per_ctx_token_ms"],
            source=str(path),
        ),
        quant_levels=None,
        supports_vision=data.get("supports_vision", False),
        kv_bytes_per_token=data.get("kv_bytes_per_token"),
        bandwidth_gbps=data.get("bandwidth_gbps"),
        gguf_filename=data.get("gguf_filename"),
    )


def load_all_profiles() -> Dict[str, ModelHardwareProfile]:
    profiles = {"smolvlm_mlx": load_smolvlm_mlx_profile()}
    try:
        profiles["qwen_llamacpp"] = load_qwen_llamacpp_profile()
    except FileNotFoundError:
        pass

    # Any auto-calibrated profiles produced by core/calibrate.py or
    # core/calibrate_mlx.py -- both emit the same self-describing schema,
    # distinguished only by filename suffix convention.
    calib_dir = ROOT / "model_calibration"
    if calib_dir.exists():
        patterns = ("*_llamacpp_profile.json", "*_mlx_profile.json")
        calib_files = sorted({f for p in patterns for f in calib_dir.glob(p)})
        for f in calib_files:
            try:
                profiles[f.stem] = load_profile_from_calibration_json(f)
            except Exception as e:
                print(f"warning: failed to load {f}: {e}")
    return profiles
