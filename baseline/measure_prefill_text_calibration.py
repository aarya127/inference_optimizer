"""Dense LM-prefill calibration by varying TEXT length, not crop count (v3).

Motivation
----------
The existing calibration of T_prefill(N) = gamma*N^2 + beta*N + alpha
(amio_constants.PREFILL_GAMMA/BETA/ALPHA) is fit to only 4 points, because
the Idefics3 processor only exposes 4 discrete crop settings and each crop
count maps to exactly one total_input_tokens value (baseline/results_v2.json:
100 / 466 / 922 / 1560). With 3 free parameters and 4 points, leave-one-out
cross-validation is close to exact interpolation -- LOOCV MAPE is 20.7%, and
the model is not trustworthy for extrapolation.

An earlier attempt to get more points (model_calibration/calibrate_cost_model.py)
injected synthetic random embeddings directly into `model.language_model(...)`
with NO KV cache object, bypassing `get_input_embeddings` and
`make_prompt_cache` entirely. amio_constants.py already documents that this
path is NOT timing-equivalent to real prefill (it underestimates ~2x at
N=1548: 2498ms predicted vs 4983ms measured) -- so it is not reused here.

This script instead reuses the EXACT real methodology of
baseline/measure_v2.py (real image -> model.get_input_embeddings -> real
make_prompt_cache -> model.language_model(...) prefill, mx.eval on actual
outputs) and only varies the TEXT portion of the prompt length while holding
crops=1 fixed (image_tokens constant at 100). LM prefill attends causally
over the full mixed image+text sequence and has no architectural reason to
treat token provenance differently, so this densifies N coverage using the
identical, already-trusted code path -- not a new, unvalidated one.

Usage (from repo root, using the phase-0 venv):
    scope/venv_phase0/bin/python baseline/measure_prefill_text_calibration.py \
        [--trials 8] [--out baseline/prefill_text_calibration.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

MODEL_ID = "mlx-community/SmolVLM-Instruct-4bit"
FILLER = ("Please look closely at the background, the foreground, the colors, "
          "and the overall composition before answering. ")
# Repeat counts chosen to spread total_input_tokens roughly evenly from just
# above the processor's 1-crop floor (100 image tokens + short text) out to
# ~2400 tokens -- beyond the previous max of 1560, to also probe
# extrapolation, not just denser interpolation.
FILLER_REPEATS = [0, 1, 3, 6, 10, 16, 24, 34, 46, 60, 76, 94, 114]


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _summ(xs):
    return {
        "mean_ms": round(statistics.mean(xs), 2),
        "std_ms": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
        "min_ms": round(min(xs), 2),
        "trials_ms": [round(x, 2) for x in xs],
    }


def build_inputs(processor, image, prompt):
    out = processor(text=prompt, images=[image], return_tensors="np",
                    do_image_splitting=False)
    arrays = {}
    for key, val in out.items():
        arr = np.asarray(val)
        arrays[key] = mx.array(arr)
    ids = np.asarray(out["input_ids"][0])
    return arrays, int(ids.shape[0])


def measure_pipeline(model, arrays, decode_tokens=1):
    from mlx_lm.models.cache import make_prompt_cache

    input_ids = arrays["input_ids"]
    pixel_values = arrays["pixel_values"]
    extra = {
        k: v
        for k, v in arrays.items()
        if k not in ("input_ids", "pixel_values", "attention_mask")
    }

    t0 = _now_ms()
    embeds = model.get_input_embeddings(input_ids, pixel_values, **extra)
    inputs_embeds = getattr(embeds, "inputs_embeds", embeds)
    mx.eval(inputs_embeds)
    t_vision = _now_ms() - t0

    cache = make_prompt_cache(model.language_model)
    t0 = _now_ms()
    logits = model.language_model(
        inputs=input_ids, cache=cache, inputs_embeds=inputs_embeds
    )
    logits = getattr(logits, "logits", logits)
    next_tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(next_tok)
    t_prefill = _now_ms() - t0

    tok = next_tok
    for _ in range(decode_tokens):
        t0 = _now_ms()
        logits = model.language_model(inputs=tok[None], cache=cache)
        logits = getattr(logits, "logits", logits)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        _ = _now_ms() - t0

    return {"t_vision_ms": t_vision, "t_prefill_ms": t_prefill}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--out", default="baseline/prefill_text_calibration.json")
    args = ap.parse_args()

    from mlx_vlm import load

    print(f"[load] {MODEL_ID}")
    t0 = _now_ms()
    model, processor = load(MODEL_ID)
    load_ms = _now_ms() - t0
    print(f"[load] done in {load_ms:.0f} ms")

    rng = np.random.default_rng(42)
    base = rng.random((1024, 1024, 3)) * 0.5
    grad = np.linspace(0, 0.5, 1024)[None, :, None]
    img_arr = ((base + grad) * 255).astype("uint8")
    image = Image.fromarray(img_arr)

    results = {
        "meta": {
            "model": MODEL_ID,
            "mlx_version": mx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "trials": args.trials,
            "load_ms": round(load_ms, 1),
            "method": "real image -> get_input_embeddings -> make_prompt_cache "
                      "-> language_model prefill; crops fixed at 1 "
                      "(do_image_splitting=False); text length varied to "
                      "densify N coverage using the same real code path as "
                      "measure_v2.py. mx.eval on actual output arrays; 1 "
                      "discarded warm-up per shape (Metal kernel compile is "
                      "shape-dependent).",
        },
        "points": [],
    }

    for repeats in FILLER_REPEATS:
        text = "Describe this image. " + FILLER * repeats
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        arrays, n_tokens = build_inputs(processor, image, prompt)

        print(f"\n[repeats={repeats:3d}] total_tokens={n_tokens}")

        # Per-shape warm-up (discarded): Metal kernel compilation is
        # shape-dependent.
        measure_pipeline(model, arrays)
        mx.clear_cache()

        pre = []
        for i in range(args.trials):
            r = measure_pipeline(model, arrays)
            pre.append(r["t_prefill_ms"])
            print(f"  trial {i+1}: prefill={r['t_prefill_ms']:8.1f} ms")

        results["points"].append({
            "n_tokens": n_tokens,
            "filler_repeats": repeats,
            "lm_prefill": _summ(pre),
        })
        mx.clear_cache()

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {out}")

    print(f"\n{'tokens':>7s} {'prefill mean ms':>16s} {'std ms':>8s}")
    for p in results["points"]:
        print(f"{p['n_tokens']:7d} {p['lm_prefill']['mean_ms']:16.1f} "
              f"{p['lm_prefill']['std_ms']:8.1f}")


if __name__ == "__main__":
    sys.exit(main())
