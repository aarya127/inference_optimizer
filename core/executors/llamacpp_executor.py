"""Real llama.cpp executor: actually loads the recommended GGUF quant level
and runs a real prefill, instead of only predicting one.

Reuses the same real-call methodology as baseline/measure_llamacpp.py /
baseline/measure_quantization_tradeoff.py (llm.eval(tokens), llm.reset()
between trials) -- duplicated rather than imported, since baseline/ is
one-off measurement tooling, not a library core/ should depend on.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

from core.executor import Executor, ExecutionResult
from core.profile import ModelHardwareProfile

MODEL_DIR = Path.home() / ".cache" / "inference_optimizer_models"
QUANT_FILES: Dict[str, str] = {
    "fp16": "qwen2.5-0.5b-instruct-fp16.gguf",
    "q8_0": "qwen2.5-0.5b-instruct-q8_0.gguf",
    "q4_k_m": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
}
DEFAULT_QUANT_LEVEL = "q4_k_m"

_FILLER = ("Please look closely at the background, the foreground, the "
           "colors, and the overall composition before answering. ")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class LlamaCppExecutor(Executor):
    def __init__(self):
        self._loaded = {}  # quant_level -> Llama instance, cached across calls

    def supports(self, profile: ModelHardwareProfile) -> bool:
        return profile.backend == "llama.cpp"

    def _get_llm(self, quant_level: str, n_ctx: int):
        key = (quant_level, n_ctx)
        if key not in self._loaded:
            from llama_cpp import Llama

            path = MODEL_DIR / QUANT_FILES[quant_level]
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found -- download it first (see "
                    f"baseline/TIER2_LLAMACPP_FINDINGS.md)"
                )
            self._loaded[key] = Llama(model_path=str(path), n_gpu_layers=-1,
                                       n_ctx=n_ctx, verbose=False)
        return self._loaded[key]

    def _build_tokens(self, llm, n_tokens: int) -> list[int]:
        text = _FILLER * (n_tokens // 8 + 5)
        toks = llm.tokenize(text.encode("utf-8"), add_bos=True)
        if len(toks) < n_tokens:
            raise ValueError(f"filler text too short for n_tokens={n_tokens}")
        return toks[:n_tokens]

    def run_prefill(
        self,
        profile: ModelHardwareProfile,
        n_tokens: int,
        quant_level: Optional[str] = None,
        trials: int = 3,
    ) -> ExecutionResult:
        if not self.supports(profile):
            raise ValueError(f"LlamaCppExecutor does not support backend {profile.backend!r}")

        quant_level = quant_level or DEFAULT_QUANT_LEVEL
        if quant_level not in QUANT_FILES:
            raise ValueError(f"unknown quant_level {quant_level!r}, "
                              f"expected one of {list(QUANT_FILES)}")

        n_ctx = max(n_tokens + 64, 512)
        llm = self._get_llm(quant_level, n_ctx)
        tokens = self._build_tokens(llm, n_tokens)

        # Discarded warm-up (shape-dependent kernel dispatch).
        llm.reset()
        llm.eval(tokens)

        times = []
        for _ in range(trials):
            llm.reset()
            t0 = _now_ms()
            llm.eval(tokens)
            times.append(_now_ms() - t0)
        measured = sum(times) / len(times)

        predicted = profile.prefill.predict_ms(n_tokens)
        error_pct = abs(measured - predicted) / measured * 100 if measured else float("nan")

        lo, hi = profile.prefill.domain_n
        note = "" if lo <= n_tokens <= hi else \
            f"n_tokens={n_tokens} is outside the calibrated domain [{lo:.0f},{hi:.0f}]"

        return ExecutionResult(
            n_tokens=n_tokens,
            predicted_prefill_ms=predicted,
            measured_prefill_ms=measured,
            error_pct=error_pct,
            quant_level_used=quant_level,
            note=note,
        )
