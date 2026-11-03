"""Automated calibration pipeline for MLX-LM text models (the MLX
counterpart to core/calibrate.py's llama.cpp pipeline).

Deliberately targets mlx-community/Qwen2.5-0.5B-Instruct-4bit -- the SAME
base architecture (hidden_size=896, 24 layers, 2 KV heads/GQA) already
calibrated on llama.cpp as qwen_llamacpp -- so this produces a genuine
same-model, cross-backend comparison, not just another unrelated model.

Reuses fit_quadratic_with_loocv/fit_decode_linear/fetch_architecture from
core.calibrate unchanged: those functions only operate on plain (n, mean,
std) records and a HF config.json URL, so they are already backend-
agnostic. Only the real-call measurement (mlx_lm's model(...) / cache API,
no GGUF download step -- mlx_lm.load() handles HF caching itself and did
not hang in testing, unlike huggingface_hub's hf_hub_download on a GGUF
file earlier in this project) is MLX-specific.

Usage:
    scope/venv_phase0/bin/python -m core.calibrate_mlx \
        --model-id mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --out model_calibration/qwen25_05b_mlx_profile.json
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

_FILLER = ("The quick brown fox jumps over the lazy dog near the river bank "
           "while the sun sets slowly behind the distant mountains. ")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _build_tokens(tokenizer, n_tokens: int) -> list[int]:
    text = _FILLER * (n_tokens // 8 + 5)
    toks = tokenizer.encode(text)
    if len(toks) < n_tokens:
        raise ValueError(f"filler text too short for n_tokens={n_tokens}")
    return toks[:n_tokens]


def calibrate_prefill_mlx(model, tokenizer, token_counts: list[int], trials: int) -> list[dict]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    points = []

    def _one_prefill(tokens):
        cache = make_prompt_cache(model)
        t0 = _now_ms()
        out = model(mx.array(tokens)[None], cache=cache)
        logits = getattr(out, "logits", out)
        mx.eval(logits)
        return _now_ms() - t0

    warm = _build_tokens(tokenizer, token_counts[0])
    _one_prefill(warm)  # global warm-up, discarded

    for n_tok in token_counts:
        tokens = _build_tokens(tokenizer, n_tok)
        _one_prefill(tokens)  # per-shape warm-up, discarded

        times = [_one_prefill(tokens) for _ in range(trials)]
        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"  N={n_tok:5d}  prefill={mean_t:8.1f}ms  std={std_t:6.2f}")
        points.append({"n_tokens": n_tok, "mean_ms": mean_t, "std_ms": std_t})
    return points


def calibrate_decode_mlx(model, tokenizer, ctx_tokens_list: list[int], n_decode: int = 32) -> list[dict]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    results = []
    for ctx in ctx_tokens_list:
        tokens = _build_tokens(tokenizer, ctx)
        cache = make_prompt_cache(model)
        out = model(mx.array(tokens)[None], cache=cache)
        logits = getattr(out, "logits", out)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_tok)

        tbt = []
        tok = next_tok
        for _ in range(n_decode):
            t0 = _now_ms()
            out = model(tok[None], cache=cache)
            logits = getattr(out, "logits", out)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            tbt.append(_now_ms() - t0)
        results.append({"ctx_tokens": ctx, "mean_ms": statistics.mean(tbt)})
    return results


def run_calibration(
    model_id: str,
    token_counts: Optional[list[int]] = None,
    trials: int = 8,
) -> dict:
    import mlx.core as mx
    from mlx_lm import load

    token_counts = token_counts or DEFAULT_TOKEN_COUNTS

    print(f"=== Auto-calibrating {model_id} (MLX) ===")
    arch = fetch_architecture(model_id)  # the MLX repo's own config.json
    print(f"[architecture] {arch}")

    print(f"[load] {model_id}")
    t0 = _now_ms()
    model, tokenizer = load(model_id)
    mx.eval(model.parameters())  # materialize weights before timing starts
    print(f"  loaded in {_now_ms() - t0:.0f} ms")

    print("[calibrate] prefill sweep")
    prefill_points = calibrate_prefill_mlx(model, tokenizer, token_counts, trials)
    prefill_fit = fit_quadratic_with_loocv(prefill_points)
    print(f"  gamma={prefill_fit['gamma']:.6e} beta={prefill_fit['beta']:.6f} "
          f"alpha={prefill_fit['alpha']:.2f}  R^2={prefill_fit['r2']:.4f}  "
          f"LOOCV={prefill_fit['loocv_mape_pct']:.2f}%")

    print("[calibrate] decode sweep")
    decode_points = calibrate_decode_mlx(model, tokenizer, token_counts[:5])
    decode_fit = fit_decode_linear(decode_points)
    print(f"  TBT(ctx) ~= {decode_fit['overhead_ms']:.3f} + "
          f"{decode_fit['per_ctx_token_ms']:.6f}*ctx  ms")

    return {
        "model_id": model_id,
        "backend": "mlx",
        "hardware": "Apple M3 base (10-core GPU, 8GB unified, 100GB/s)",
        "supports_vision": False,
        "architecture": arch,
        "kv_bytes_per_token": arch["kv_bytes_per_token"],
        "bandwidth_gbps": 100.0,
        "prefill": prefill_fit,
        "decode": decode_fit,
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
