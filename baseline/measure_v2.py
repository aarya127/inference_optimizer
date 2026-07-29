"""Corrected baseline measurement campaign (v2).

Fixes every methodology defect of the original run_experiments.py:

  1. Real synchronization: `mx.eval(<output arrays>)` on actual results —
     the old script called `mx.eval()` with no arguments, a no-op, so lazy
     evaluation folded vision compute into whatever stage evaluated first.
  2. Direct stage isolation: the vision tower + connector is timed via
     `model.get_input_embeddings(...)`, LM prefill via a cached
     `model.language_model(...)` call, decode via a stepwise cached loop
     with per-token timestamps. No stage is derived as a residual.
  3. Crop control that actually varies: the Idefics3 processor resizes every
     image to longest_edge=1536 by default (upscaling small images!), which
     is why the old resolution sweep produced 1548 visual tokens at every
     "resolution". We drive crop count directly via `do_image_splitting` /
     `size={"longest_edge": N}`:  1 / 5 / 10 / 17 crops (81 tokens each).
     NOTE: 17 crops is the processor's maximum — the study's "24 crops"
     setting does not exist in this pipeline.
  4. Warm-up: one discarded full pipeline pass per crop configuration
     (Metal kernel compilation is shape-dependent).
  5. Replication: N trials per configuration (default 5), mean/std/min
     reported; raw trials saved. No fabricated percentiles.
  6. Cross-check: one `stream_generate` end-to-end TTFT per configuration
     to validate the manual stage sum against the real user-visible path.

Usage (from repo root, using the phase-0 venv):
    scope/venv_phase0/bin/python baseline/measure_v2.py \
        [--trials 5] [--decode-tokens 32] [--out baseline/results_v2.json]
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
IMAGE_TOKEN_ID = 49153

# Crop configurations: label -> processor kwargs
CROP_CONFIGS = [
    ("crops_1", dict(do_image_splitting=False)),
    ("crops_5", dict(size={"longest_edge": 768})),
    ("crops_10", dict(size={"longest_edge": 1152})),
    ("crops_17", dict()),  # processor default: longest_edge=1536
]


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _summ(xs):
    return {
        "mean_ms": round(statistics.mean(xs), 2),
        "std_ms": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
        "min_ms": round(min(xs), 2),
        "trials_ms": [round(x, 2) for x in xs],
    }


def build_inputs(processor, image, prompt, proc_kwargs):
    """Run the HF processor and convert to mx arrays."""
    out = processor(text=prompt, images=[image], return_tensors="np", **proc_kwargs)
    arrays = {}
    for key, val in out.items():
        arr = np.asarray(val)
        arrays[key] = mx.array(arr)
    ids = np.asarray(out["input_ids"][0])
    meta = {
        "total_input_tokens": int(ids.shape[0]),
        "image_tokens": int((ids == IMAGE_TOKEN_ID).sum()),
        "n_crops": int(np.asarray(out["pixel_values"]).shape[1]),
    }
    return arrays, meta


def measure_pipeline(model, arrays, decode_tokens):
    """One fully synchronized pass; returns per-stage timings in ms."""
    from mlx_lm.models.cache import make_prompt_cache

    input_ids = arrays["input_ids"]
    pixel_values = arrays["pixel_values"]
    extra = {
        k: v
        for k, v in arrays.items()
        if k not in ("input_ids", "pixel_values", "attention_mask")
    }

    # -- Stage 1: vision tower + connector + embedding merge ---------------
    t0 = _now_ms()
    embeds = model.get_input_embeddings(input_ids, pixel_values, **extra)
    inputs_embeds = getattr(embeds, "inputs_embeds", embeds)
    mx.eval(inputs_embeds)
    t_vision = _now_ms() - t0

    # -- Stage 2: LM prefill ------------------------------------------------
    cache = make_prompt_cache(model.language_model)
    t0 = _now_ms()
    logits = model.language_model(
        inputs=input_ids, cache=cache, inputs_embeds=inputs_embeds
    )
    logits = getattr(logits, "logits", logits)
    next_tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(next_tok)
    t_prefill = _now_ms() - t0

    # -- Stage 3: decode with per-token timestamps --------------------------
    tbt_ms = []
    tok = next_tok
    for _ in range(decode_tokens):
        t0 = _now_ms()
        logits = model.language_model(inputs=tok[None], cache=cache)
        logits = getattr(logits, "logits", logits)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        tbt_ms.append(_now_ms() - t0)

    return {
        "t_vision_ms": t_vision,
        "t_prefill_ms": t_prefill,
        "t_ttft_ms": t_vision + t_prefill,
        "tbt_ms": tbt_ms,
    }


def stream_ttft_crosscheck(model, processor, image_path, prompt, proc_kwargs):
    """End-to-end TTFT via the real generation path, for validation."""
    from mlx_vlm import stream_generate

    t0 = _now_ms()
    first = None
    n = 0
    for chunk in stream_generate(
        model,
        processor,
        prompt,
        image=[str(image_path)],
        max_tokens=4,
        **proc_kwargs,
    ):
        n += 1
        if first is None:
            first = _now_ms() - t0
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--out", default="baseline/results_v2.json")
    ap.add_argument("--skip-stream-check", action="store_true")
    args = ap.parse_args()

    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template

    print(f"[load] {MODEL_ID}")
    t0 = _now_ms()
    model, processor = load(MODEL_ID)
    load_ms = _now_ms() - t0
    print(f"[load] done in {load_ms:.0f} ms; "
          f"active_mem={mx.get_active_memory()/2**20:.0f} MiB")

    # Deterministic synthetic test image (seeded noise + gradient).
    rng = np.random.default_rng(42)
    base = rng.random((1024, 1024, 3)) * 0.5
    grad = np.linspace(0, 0.5, 1024)[None, :, None]
    img_arr = ((base + grad) * 255).astype("uint8")
    image = Image.fromarray(img_arr)
    img_path = Path("baseline/_test_image_1024.png")
    image.save(img_path)

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)

    results = {
        "meta": {
            "model": MODEL_ID,
            "mlx_version": mx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "trials": args.trials,
            "decode_tokens": args.decode_tokens,
            "load_ms": round(load_ms, 1),
            "method": "direct stage isolation with mx.eval on outputs; "
                      "1 warm-up pass per crop config; synthetic seeded image",
        },
        "configs": {},
    }

    for label, kw in CROP_CONFIGS:
        arrays, meta = build_inputs(processor, image, prompt, kw)
        print(f"\n[{label}] crops={meta['n_crops']} "
              f"image_tokens={meta['image_tokens']} "
              f"total_tokens={meta['total_input_tokens']}")

        # Warm-up (discarded): shape-specific Metal kernel compilation.
        measure_pipeline(model, arrays, decode_tokens=4)
        mx.clear_cache()

        vis, pre, ttft, tbt_all = [], [], [], []
        for i in range(args.trials):
            r = measure_pipeline(model, arrays, args.decode_tokens)
            vis.append(r["t_vision_ms"])
            pre.append(r["t_prefill_ms"])
            ttft.append(r["t_ttft_ms"])
            tbt_all.extend(r["tbt_ms"])
            print(f"  trial {i+1}: vision={r['t_vision_ms']:8.1f}  "
                  f"prefill={r['t_prefill_ms']:8.1f}  "
                  f"ttft={r['t_ttft_ms']:8.1f}  "
                  f"tbt_mean={statistics.mean(r['tbt_ms']):6.1f} ms")

        entry = {
            **meta,
            "vision": _summ(vis),
            "lm_prefill": _summ(pre),
            "ttft_stage_sum": _summ(ttft),
            "decode_tbt": {
                "mean_ms": round(statistics.mean(tbt_all), 2),
                "std_ms": round(statistics.stdev(tbt_all), 2),
                "p95_ms": round(float(np.percentile(tbt_all, 95)), 2),
                "n_tokens": len(tbt_all),
            },
            "peak_mem_mib": round(mx.get_peak_memory() / 2**20, 1),
        }

        if not args.skip_stream_check:
            e2e = stream_ttft_crosscheck(model, processor, img_path, prompt, kw)
            entry["stream_generate_ttft_ms"] = round(e2e, 1) if e2e else None
            print(f"  stream_generate end-to-end TTFT: {e2e:.1f} ms")

        results["configs"][label] = entry
        mx.clear_cache()

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {out}")

    # Compact summary table
    print(f"\n{'config':10s} {'crops':>5s} {'tokens':>7s} "
          f"{'vision ms':>10s} {'prefill ms':>11s} {'TTFT ms':>9s} {'TBT ms':>7s}")
    for label, e in results["configs"].items():
        print(f"{label:10s} {e['n_crops']:5d} {e['total_input_tokens']:7d} "
              f"{e['vision']['mean_ms']:10.1f} {e['lm_prefill']['mean_ms']:11.1f} "
              f"{e['ttft_stage_sum']['mean_ms']:9.1f} "
              f"{e['decode_tbt']['mean_ms']:7.1f}")


if __name__ == "__main__":
    sys.exit(main())
