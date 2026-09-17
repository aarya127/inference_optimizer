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
    )


def load_all_profiles() -> Dict[str, ModelHardwareProfile]:
    profiles = {"smolvlm_mlx": load_smolvlm_mlx_profile()}
    try:
        profiles["qwen_llamacpp"] = load_qwen_llamacpp_profile()
    except FileNotFoundError:
        pass
    return profiles
