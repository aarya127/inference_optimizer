"""Real MLX-LM executor: actually runs a real text-only prefill through the
model, instead of only predicting one. The generic counterpart to
MLXExecutor (which is restricted to vision profiles via mlx_vlm) -- this
one works against any plain-text MLX-LM profile by loading
profile.model_id directly, with no per-model code.

Reuses the same real-call methodology as core/calibrate_mlx.py
(model(input_ids, cache=...), mx.eval on actual outputs) -- duplicated
rather than imported, matching this repo's convention of keeping
measurement/calibration tooling and the executors that verify it as
separate, independently-readable code paths.
"""

from __future__ import annotations

import time
from typing import Optional

from core.executor import Executor, ExecutionResult
from core.profile import ModelHardwareProfile

_FILLER = ("The quick brown fox jumps over the lazy dog near the river bank "
           "while the sun sets slowly behind the distant mountains. ")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class MLXLMExecutor(Executor):
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded_model_id = None

    def supports(self, profile: ModelHardwareProfile) -> bool:
        return profile.backend == "mlx" and not profile.supports_vision

    def _ensure_loaded(self, model_id: str):
        if self._model is not None and self._loaded_model_id == model_id:
            return
        import mlx.core as mx
        from mlx_lm import load

        self._model, self._tokenizer = load(model_id)
        mx.eval(self._model.parameters())
        self._loaded_model_id = model_id

    def _build_tokens(self, n_tokens: int) -> list[int]:
        text = _FILLER * (n_tokens // 8 + 5)
        toks = self._tokenizer.encode(text)
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
            raise ValueError(f"MLXLMExecutor does not support this profile "
                              f"(backend={profile.backend!r}, supports_vision={profile.supports_vision})")

        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        self._ensure_loaded(profile.model_id)
        tokens = self._build_tokens(n_tokens)

        def _one_prefill():
            cache = make_prompt_cache(self._model)
            t0 = _now_ms()
            out = self._model(mx.array(tokens)[None], cache=cache)
            logits = getattr(out, "logits", out)
            mx.eval(logits)
            return _now_ms() - t0

        _one_prefill()  # discarded warm-up
        times = [_one_prefill() for _ in range(trials)]
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
            quant_level_used=None,
            note=note,
        )
