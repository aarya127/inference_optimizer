"""Automated calibration pipeline for MLX vision-language models (the VLM
counterpart to core/calibrate_mlx.py's text-only pipeline).

Deliberately targets mlx-community/SmolVLM-256M-Instruct-4bit -- the same
Idefics3 family already hand-calibrated as smolvlm_mlx (the ~2B-class
SmolVLM-Instruct-4bit), but a genuinely different, much smaller instance
(SmolLM2-135M-class LM backbone: hidden_size=576, 30 layers, 3 KV
heads/GQA, vs the original's SmolLM2-1.7B-class backbone) -- so this is
the vision-side counterpart to Tier 5's SmolLM2-360M proof: a second,
automatically-calibrated model in the same family, not a repeat of the
same checkpoint.

Reuses fit_quadratic_with_loocv / fit_decode_linear / fetch_architecture
from core.calibrate unchanged -- fetch_architecture already handles the
nested "text_config" that vision-language configs use (see its docstring).
Only the real-call measurement is VLM-specific, and even that reuses the
exact methodology already proven in core/executors/mlx_executor.py and
baseline/measure_prefill_text_calibration.py: crops fixed at 1
(do_image_splitting=False) on a synthetic seeded image, text length varied
to reach each target N, get_input_embeddings -> make_prompt_cache ->
language_model prefill, mx.eval on actual outputs.

Usage:
    scope/venv_phase0/bin/python -m core.calibrate_mlx_vlm \
        --model-id mlx-community/SmolVLM-256M-Instruct-4bit \
        --out model_calibration/smolvlm_256m_mlx_profile.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

from core.calibrate import (
    DEFAULT_TOKEN_COUNTS,
    fetch_architecture,
    fit_decode_linear,
    fit_quadratic_with_loocv,
)

_FILLER = ("Please look closely at the background, the foreground, the "
           "colors, and the overall composition before answering. ")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _make_seeded_image():
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)
    base = rng.random((1024, 1024, 3)) * 0.5
    grad = np.linspace(0, 0.5, 1024)[None, :, None]
    img_arr = ((base + grad) * 255).astype("uint8")
    return Image.fromarray(img_arr)


def _build_inputs_for_n(processor, image, target_n_tokens: int):
    """Fixed crops=1 (~100 real image tokens); search filler repeats to
    land close to target_n_tokens, same approach as MLXExecutor._build_inputs."""
    import numpy as np

    repeats = 0
    out = None
    for _ in range(20):
        text = "Describe this image. " + _FILLER * repeats
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        out = processor(text=prompt, images=[image], return_tensors="np",
                        do_image_splitting=False)
        n = int(np.asarray(out["input_ids"][0]).shape[0])
        if n >= target_n_tokens or repeats > 2000:
            break
        repeats += max(1, (target_n_tokens - n) // 20)
    return out, n


def calibrate_prefill_vlm(model, processor, image, token_counts: list[int], trials: int) -> list[dict]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    def _one_prefill(arrays):
        input_ids = mx.array(arrays["input_ids"])
        pixel_values = mx.array(arrays["pixel_values"])
        extra = {k: mx.array(v) for k, v in arrays.items()
                 if k not in ("input_ids", "pixel_values", "attention_mask")}

        embeds = model.get_input_embeddings(input_ids, pixel_values, **extra)
        inputs_embeds = getattr(embeds, "inputs_embeds", embeds)
        mx.eval(inputs_embeds)

        cache = make_prompt_cache(model.language_model)
        t0 = _now_ms()
        logits = model.language_model(inputs=input_ids, cache=cache, inputs_embeds=inputs_embeds)
        logits = getattr(logits, "logits", logits)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_tok)
        return _now_ms() - t0

    points = []
    warm_arrays, _ = _build_inputs_for_n(processor, image, token_counts[0])
    _one_prefill(warm_arrays)  # global warm-up, discarded

    for target_n in token_counts:
        arrays, actual_n = _build_inputs_for_n(processor, image, target_n)
        _one_prefill(arrays)  # per-shape warm-up, discarded

        times = [_one_prefill(arrays) for _ in range(trials)]
        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"  target_N={target_n:5d} actual_N={actual_n:5d}  "
              f"prefill={mean_t:8.1f}ms  std={std_t:6.2f}")
        points.append({"n_tokens": actual_n, "mean_ms": mean_t, "std_ms": std_t})
    return points


def calibrate_decode_vlm(model, processor, image, ctx_targets: list[int], n_decode: int = 32) -> list[dict]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    results = []
    for target_n in ctx_targets:
        arrays, actual_n = _build_inputs_for_n(processor, image, target_n)
        input_ids = mx.array(arrays["input_ids"])
        pixel_values = mx.array(arrays["pixel_values"])
        extra = {k: mx.array(v) for k, v in arrays.items()
                 if k not in ("input_ids", "pixel_values", "attention_mask")}

        embeds = model.get_input_embeddings(input_ids, pixel_values, **extra)
        inputs_embeds = getattr(embeds, "inputs_embeds", embeds)
        mx.eval(inputs_embeds)

        cache = make_prompt_cache(model.language_model)
        logits = model.language_model(inputs=input_ids, cache=cache, inputs_embeds=inputs_embeds)
        logits = getattr(logits, "logits", logits)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_tok)

        tbt = []
        tok = next_tok
        for _ in range(n_decode):
            t0 = _now_ms()
            out = model.language_model(inputs=tok[None], cache=cache)
            logits = getattr(out, "logits", out)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            tbt.append(_now_ms() - t0)
        results.append({"ctx_tokens": actual_n, "mean_ms": statistics.mean(tbt)})
    return results


def run_calibration(
    model_id: str,
    token_counts: Optional[list[int]] = None,
    trials: int = 8,
) -> dict:
    from mlx_vlm import load

    token_counts = token_counts or DEFAULT_TOKEN_COUNTS

    print(f"=== Auto-calibrating {model_id} (MLX, vision-language) ===")
    arch = fetch_architecture(model_id)
    print(f"[architecture] {arch}")

    print(f"[load] {model_id}")
    t0 = _now_ms()
    model, processor = load(model_id)
    print(f"  loaded in {_now_ms() - t0:.0f} ms")

    image = _make_seeded_image()

    print("[calibrate] prefill sweep (crops=1, text length varied)")
    prefill_points = calibrate_prefill_vlm(model, processor, image, token_counts, trials)
    prefill_fit = fit_quadratic_with_loocv(prefill_points)
    print(f"  gamma={prefill_fit['gamma']:.6e} beta={prefill_fit['beta']:.6f} "
          f"alpha={prefill_fit['alpha']:.2f}  R^2={prefill_fit['r2']:.4f}  "
          f"LOOCV={prefill_fit['loocv_mape_pct']:.2f}%  "
          f"max_LOOCV_AE={prefill_fit['loocv_max_ape_pct']:.2f}%")

    print("[calibrate] decode sweep")
    decode_points = calibrate_decode_vlm(model, processor, image, token_counts[:5])
    decode_fit = fit_decode_linear(decode_points)
    print(f"  TBT(ctx) ~= {decode_fit['overhead_ms']:.3f} + "
          f"{decode_fit['per_ctx_token_ms']:.6f}*ctx  ms"
          f"{'  [clamped from a negative raw slope]' if decode_fit['clamped_negative_slope'] else ''}")

    return {
        "model_id": model_id,
        "backend": "mlx",
        "hardware": "Apple M3 base (10-core GPU, 8GB unified, 100GB/s)",
        "supports_vision": True,
        "architecture": arch,
        "kv_bytes_per_token": arch["kv_bytes_per_token"],
        "bandwidth_gbps": 100.0,
        "prefill": prefill_fit,
        "decode": decode_fit,
        "calibration_method": "crops=1 (do_image_splitting=False, ~100 real "
                               "image tokens via a synthetic seeded image), "
                               "text length varied to reach each target N -- "
                               "same method as baseline/measure_prefill_text_calibration.py",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    profile = run_calibration(model_id=args.model_id, trials=args.trials)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2))
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
