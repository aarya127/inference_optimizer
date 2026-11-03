"""Real llama.cpp executor: actually loads the recommended GGUF file and
runs a real prefill, instead of only predicting one.

Resolves which file to load entirely from the profile itself (not a
hardcoded per-model dict): a quant_level name maps through
profile.quant_levels[level].gguf_filename when the profile has multiple
measured levels (e.g. Qwen2.5-0.5B), otherwise falls back to
profile.gguf_filename (the single file auto-calibrated profiles carry).
This is what lets the same Executor run against ANY profile -- including
one core/calibrate.py just produced for a model this file has never heard
of -- without new code here.

Reuses the same real-call methodology as baseline/measure_llamacpp.py /
baseline/measure_quantization_tradeoff.py (llm.eval(tokens), llm.reset()
between trials) -- duplicated rather than imported, since baseline/ is
one-off measurement tooling, not a library core/ should depend on.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from core.executor import Executor, ExecutionResult
from core.profile import ModelHardwareProfile

MODEL_DIR = Path.home() / ".cache" / "inference_optimizer_models"

_FILLER = ("Please look closely at the background, the foreground, the "
           "colors, and the overall composition before answering. ")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def resolve_gguf_filename(profile: ModelHardwareProfile, quant_level: Optional[str]) -> str:
    """The single place that turns (profile, quant_level) into a filename."""
    if quant_level is not None:
        if not profile.quant_levels or quant_level not in profile.quant_levels:
            raise ValueError(
                f"quant_level={quant_level!r} requested but profile "
                f"{profile.model_id!r} has no measured data for it "
                f"(available: {list(profile.quant_levels or {})})"
            )
        filename = profile.quant_levels[quant_level].gguf_filename
        if filename is None:
            raise ValueError(
                f"profile {profile.model_id!r} has quant_level={quant_level!r} "
                f"stats but no recorded gguf_filename for it"
            )
        return filename

    if profile.gguf_filename is not None:
        return profile.gguf_filename

    raise ValueError(
        f"profile {profile.model_id!r} has no gguf_filename and no "
        f"quant_level was given -- nothing to load"
    )


class LlamaCppExecutor(Executor):
    def __init__(self):
        self._loaded = {}  # (filename, n_ctx) -> Llama instance, cached across calls

    def supports(self, profile: ModelHardwareProfile) -> bool:
        return profile.backend == "llama.cpp"

    def _get_llm(self, filename: str, n_ctx: int):
        key = (filename, n_ctx)
        if key not in self._loaded:
            from llama_cpp import Llama

            path = MODEL_DIR / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found -- download it first (see "
                    f"baseline/TIER2_LLAMACPP_FINDINGS.md or core/calibrate.py)"
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

        filename = resolve_gguf_filename(profile, quant_level)
        n_ctx = max(n_tokens + 64, 512)
        llm = self._get_llm(filename, n_ctx)
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
