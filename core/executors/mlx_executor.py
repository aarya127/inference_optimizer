"""Real MLX/SmolVLM executor: actually runs a real image through the real
vision encoder + LM prefill, instead of only predicting a latency.

Reuses the same real-call methodology as baseline/measure_v2.py /
baseline/measure_prefill_text_calibration.py (get_input_embeddings ->
make_prompt_cache -> language_model prefill, mx.eval on actual outputs) --
duplicated rather than imported, since baseline/ is one-off measurement
tooling, not a library core/ should depend on. Crops are fixed at 1
(do_image_splitting=False, ~100 real image tokens); text length is varied
to reach the requested total n_tokens, exactly as in the Tier-1 calibration
densification.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from core.executor import Executor, ExecutionResult
from core.profile import ModelHardwareProfile

MODEL_ID = "mlx-community/SmolVLM-Instruct-4bit"
_FILLER = ("Please look closely at the background, the foreground, the "
           "colors, and the overall composition before answering. ")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class MLXExecutor(Executor):
    def __init__(self):
        self._model = None
        self._processor = None
        self._image = None

    def supports(self, profile: ModelHardwareProfile) -> bool:
        return profile.backend == "mlx"

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from mlx_vlm import load

        self._model, self._processor = load(MODEL_ID)

        rng = np.random.default_rng(42)
        base = rng.random((1024, 1024, 3)) * 0.5
        grad = np.linspace(0, 0.5, 1024)[None, :, None]
        img_arr = ((base + grad) * 255).astype("uint8")
        self._image = Image.fromarray(img_arr)

    def _build_inputs(self, target_n_tokens: int):
        import mlx.core as mx

        # Binary-ish search over filler repeats to land close to
        # target_n_tokens; the tokenizer's per-repeat token count is
        # roughly constant, so a couple of probes converge quickly.
        repeats = 0
        for _ in range(20):
            text = "Describe this image. " + _FILLER * repeats
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": text}]}]
            prompt = self._processor.apply_chat_template(msgs, add_generation_prompt=True)
            out = self._processor(text=prompt, images=[self._image],
                                   return_tensors="np", do_image_splitting=False)
            n = int(np.asarray(out["input_ids"][0]).shape[0])
            if n >= target_n_tokens or repeats > 2000:
                break
            # ~20 tokens per filler repeat (measured in
            # baseline/prefill_text_calibration.json, e.g. repeats 6->10
            # goes 221->301 tokens = 20/repeat); step proportionally, but
            # never by less than 1 so this always terminates.
            repeats += max(1, (target_n_tokens - n) // 20)

        arrays = {k: mx.array(np.asarray(v)) for k, v in out.items()}
        return arrays, n

    def run_prefill(
        self,
        profile: ModelHardwareProfile,
        n_tokens: int,
        quant_level: Optional[str] = None,
        trials: int = 3,
    ) -> ExecutionResult:
        if not self.supports(profile):
            raise ValueError(f"MLXExecutor does not support backend {profile.backend!r}")

        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        self._ensure_loaded()
        arrays, actual_n = self._build_inputs(n_tokens)

        def _one_prefill():
            input_ids = arrays["input_ids"]
            pixel_values = arrays["pixel_values"]
            extra = {k: v for k, v in arrays.items()
                     if k not in ("input_ids", "pixel_values", "attention_mask")}

            embeds = self._model.get_input_embeddings(input_ids, pixel_values, **extra)
            inputs_embeds = getattr(embeds, "inputs_embeds", embeds)
            mx.eval(inputs_embeds)

            cache = make_prompt_cache(self._model.language_model)
            t0 = _now_ms()
            logits = self._model.language_model(
                inputs=input_ids, cache=cache, inputs_embeds=inputs_embeds)
            logits = getattr(logits, "logits", logits)
            next_tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_tok)
            return _now_ms() - t0

        _one_prefill()  # discarded warm-up
        times = [_one_prefill() for _ in range(trials)]
        measured = sum(times) / len(times)

        predicted = profile.prefill.predict_ms(actual_n)
        error_pct = abs(measured - predicted) / measured * 100 if measured else float("nan")

        lo, hi = profile.prefill.domain_n
        note = "" if lo <= actual_n <= hi else \
            f"actual_n={actual_n} is outside the calibrated domain [{lo:.0f},{hi:.0f}]"
        if actual_n != n_tokens:
            note += (f" (requested n_tokens={n_tokens}; text-length search "
                      f"landed on {actual_n} -- tokenizer granularity, not "
                      f"an exact match by construction)")

        return ExecutionResult(
            n_tokens=actual_n,
            predicted_prefill_ms=predicted,
            measured_prefill_ms=measured,
            error_pct=error_pct,
            quant_level_used=None,  # SmolVLM ships pre-quantized; no choice here
            note=note.strip(),
        )
