"""Second-backend, second-model calibration campaign: llama.cpp + Metal.

Purpose (Tier 2 of the roadmap): everything measured so far in this repo
uses one model (SmolVLM-Instruct-4bit) on one backend (MLX). This script
applies the SAME honest measurement discipline established in
baseline/measure_v2.py -- real inference calls, real synchronization
(llama.cpp calls are synchronous, so no mx.eval-style lazy-eval trap here),
warm-up before timing, N independent trials with mean/std/min reported, no
derived residuals, no fabricated percentiles -- to a DIFFERENT model on a
DIFFERENT inference engine (llama.cpp via llama-cpp-python, Metal GPU
offload), to see whether the quadratic T_prefill(N) cost-model form and the
"decode is roughly flat with context" finding from the MLX/SmolVLM study
generalize, or were specific to that one (model, backend) pair.

This is NOT a benchmark comparing MLX vs llama.cpp speed (different model,
different quantization, different architecture -- not a fair fight and not
the point). It is a test of whether the MEASUREMENT METHODOLOGY and the
COST-MODEL FUNCTIONAL FORM transfer.

Model: Qwen2.5-0.5B-Instruct, Q4_K_M GGUF (text-only, unrelated architecture
to SmolVLM/SmolLM2 -- a genuinely different model, not a re-export of the
same one).

Usage (from repo root, using the phase-0 venv):
    scope/venv_phase0/bin/python baseline/measure_llamacpp.py \
        [--trials 8] [--decode-tokens 32] [--out baseline/results_llamacpp.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
# Downloaded via curl (see repo notes), not huggingface_hub's hf_hub_download:
# that downloader has no read-timeout and hung indefinitely on a dead
# connection (TCP CLOSE_WAIT, zero progress) during this project's own
# download. curl with --speed-time/--speed-limit aborts and retries instead.
LOCAL_MODEL_PATH = Path.home() / ".cache" / "inference_optimizer_models" / MODEL_FILE

# Token counts to test. A 0.5B model at Q4 is cheap enough per-token that we
# can span a wide range without the multi-second-per-trial cost that limited
# the SmolVLM campaign's range.
TOKEN_COUNTS = [32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
FILLER = ("The quick brown fox jumps over the lazy dog near the river bank "
          "while the sun sets slowly behind the distant mountains. ")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _summ(xs):
    return {
        "mean_ms": round(statistics.mean(xs), 2),
        "std_ms": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
        "min_ms": round(min(xs), 2),
        "trials_ms": [round(x, 2) for x in xs],
    }


def build_token_sequence(llm, n_tokens: int) -> list[int]:
    """Tokenize repeated filler text and truncate/pad to exactly n_tokens."""
    text = FILLER * (n_tokens // 8 + 5)  # generous over-supply, then truncate
    toks = llm.tokenize(text.encode("utf-8"), add_bos=True)
    if len(toks) < n_tokens:
        raise ValueError(f"filler text too short for n_tokens={n_tokens}")
    return toks[:n_tokens]


def measure_prefill(llm, tokens: list[int]) -> float:
    """Real prefill: llm.eval() processes all tokens through the model and
    updates the KV cache -- llama.cpp calls are synchronous, so the wall
    time IS the compute time (no lazy-eval trap to guard against here)."""
    llm.reset()
    t0 = _now_ms()
    llm.eval(tokens)
    return _now_ms() - t0


def measure_decode(llm, n_decode_tokens: int) -> list[float]:
    """Per-token decode latency via manual sample+eval loop, matching the
    stepwise cached-decode methodology in baseline/measure_v2.py."""
    tbt = []
    for _ in range(n_decode_tokens):
        t0 = _now_ms()
        tok = llm.sample()
        llm.eval([tok])
        tbt.append(_now_ms() - t0)
    return tbt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--out", default="baseline/results_llamacpp.json")
    ap.add_argument("--n-ctx", type=int, default=4096)
    args = ap.parse_args()

    from llama_cpp import Llama
    import llama_cpp

    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{LOCAL_MODEL_PATH} not found. Download it first, e.g.:\n"
            f"  curl -L --retry 5 --retry-delay 3 --connect-timeout 15 "
            f"--speed-time 20 --speed-limit 1000 -o {LOCAL_MODEL_PATH} \\\n"
            f'    "https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"'
        )
    model_path = str(LOCAL_MODEL_PATH)

    print(f"[load] {model_path}")
    t0 = _now_ms()
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=args.n_ctx,
                verbose=False)
    load_ms = _now_ms() - t0
    print(f"[load] done in {load_ms:.0f} ms (n_gpu_layers=-1, Metal offload)")

    results = {
        "meta": {
            "model_repo": MODEL_REPO,
            "model_file": MODEL_FILE,
            "backend": "llama.cpp (llama-cpp-python)",
            "gpu_backend": "Metal",
            "llama_cpp_python_version": getattr(llama_cpp, "__version__", "unknown"),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "trials": args.trials,
            "decode_tokens": args.decode_tokens,
            "n_ctx": args.n_ctx,
            "load_ms": round(load_ms, 1),
            "method": "llm.eval(tokens) for prefill (synchronous, real KV-cache "
                      "update); llm.reset() between trials to clear cache "
                      "without reloading weights; 1 discarded warm-up trial "
                      "per token count; N timed trials, mean/std/min reported.",
        },
        "points": [],
    }

    # Global warm-up at the smallest shape (first Metal kernel dispatch,
    # library init, etc. -- discarded).
    warm_tokens = build_token_sequence(llm, TOKEN_COUNTS[0])
    measure_prefill(llm, warm_tokens)

    for n_tok in TOKEN_COUNTS:
        tokens = build_token_sequence(llm, n_tok)
        print(f"\n[N={n_tok:5d}]", end="", flush=True)

        # Per-shape warm-up (discarded).
        measure_prefill(llm, tokens)

        pre = []
        for _ in range(args.trials):
            t = measure_prefill(llm, tokens)
            pre.append(t)
            print(".", end="", flush=True)

        # Decode (TBT) at this context length, right after the last prefill.
        tbt = measure_decode(llm, args.decode_tokens)

        entry = {
            "n_tokens": n_tok,
            "lm_prefill": _summ(pre),
            "decode_tbt": {
                "mean_ms": round(statistics.mean(tbt), 2),
                "std_ms": round(statistics.stdev(tbt), 2),
                "n_tokens_generated": len(tbt),
            },
        }
        results["points"].append(entry)
        print(f"  prefill={entry['lm_prefill']['mean_ms']:.1f}"
              f"±{entry['lm_prefill']['std_ms']:.1f}ms  "
              f"tbt={entry['decode_tbt']['mean_ms']:.1f}ms")

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
