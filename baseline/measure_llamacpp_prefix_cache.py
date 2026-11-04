"""Real measured prefix/prompt-cache reuse: does skipping recomputation of
a shared prefix (e.g. a system prompt reused across requests) actually
save the time it should, on this backend?

Method, using only the high-level llama-cpp-python API (no low-level
ctypes needed here, unlike Tier 6's native-batching work -- this is just
sequential llm.eval() calls on one context, the same pattern used
throughout this repo):

  COLD: llm.reset(); eval(prefix + suffix) in one call, timed in full.
  WARM: llm.reset(); eval(prefix) UNTIMED (simulates the prefix's KV cache
        already being populated from a prior request); then eval(suffix)
        WITHOUT resetting, timed -- the suffix is appended to the existing
        cache, so this measures only the incremental cost of the new
        tokens, not the whole sequence.

Real savings = (t_cold - t_warm_suffix_only) / t_cold. This is NOT the
same as a prediction from the quadratic prefill model (which would say
savings ~= predict_ms(prefix_len), an approximation that assumes perfect
additivity the quadratic doesn't actually have) -- it's a direct
measurement of the real mechanism.

IMPORTANT -- a real bug found and fixed while building this (see
core/TIER7_PREFIX_CACHE_FINDINGS.md for the full writeup): `llama_decode()`
can return before the underlying Metal GPU work has actually completed, so
un-synchronized `t0 = now(); llm.eval(tokens); dt = now() - t0` timing can
silently attribute one call's GPU work to whichever call happens to force
synchronization next. This was invisible when every timed call was the
SAME shape in a tight loop (as in every other llama.cpp script in this
repo), but interleaving two DIFFERENT shapes (cold vs warm here) exposed
it clearly: without a sync point, COLD measured ~28% of its real cost and
WARM measured ~3x its real cost. `_sync()` below forces completion (reads
one logit, which the C API cannot return correctly without the decode
having finished) after every timed AND untimed eval -- this is the same
class of bug as the mx.eval()-with-no-arguments issue that originally
motivated this entire project's honesty rewrite on the MLX side
(README.md), independently rediscovered here on the llama.cpp side.

Usage:
    scope/venv_phase0/bin/python baseline/measure_llamacpp_prefix_cache.py \
        [--prefix-tokens 200] [--suffix-tokens 50 100 200] [--trials 8]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "inference_optimizer_models"
MODEL_FILE = "SmolLM2-360M-Instruct-Q4_K_M.gguf"
PREFIX_FILLER = ("You are a careful, detail-oriented assistant. Always answer "
                  "concisely, cite assumptions explicitly, and never guess "
                  "when you are uncertain. Follow the user's formatting "
                  "instructions precisely. ")
SUFFIX_FILLER = ("Please analyze the following passage and summarize its "
                  "main argument in two sentences. ")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _build_tokens(llm, filler: str, n_tokens: int) -> list[int]:
    text = filler * (n_tokens // 8 + 5)
    toks = llm.tokenize(text.encode("utf-8"), add_bos=True)
    return toks[:n_tokens]


def _sync(llm) -> None:
    """Force any pending GPU work from the most recent eval() to actually
    complete. Reading a logit cannot return correct data before the
    decode it depends on has finished, so this is a real synchronization
    point -- without it, llama_decode() can return before the underlying
    Metal work completes, and un-synchronized timing silently attributes
    one call's GPU work to whichever call happens to force sync next. See
    the module docstring and core/TIER7_PREFIX_CACHE_FINDINGS.md."""
    llm._ctx.get_logits_ith(0)


def _eval_sync(llm, tokens: list[int]) -> None:
    llm.eval(tokens)
    _sync(llm)


def measure_one(llm, prefix_tokens: list[int], suffix_tokens: list[int], trials: int) -> dict:
    cold_times = []
    warm_suffix_times = []

    # Per-shape warm-up (discarded), for BOTH shapes this function times:
    # Metal kernel compilation is shape-dependent (same caution as every
    # other measurement script in this repo -- baseline/measure_v2.py,
    # core/calibrate.py, etc.). An earlier version of this script warmed
    # up only the combined (prefix+suffix) shape once at the very start,
    # for the FIRST suffix length only -- every other (suffix length,
    # cold-vs-warm) combination then hit an un-warmed shape during timed
    # trials, producing physically impossible results (warm time ~84ms
    # constant regardless of suffix length 50/100/200; cold time
    # non-monotonic across increasing total token counts). Warming up
    # every shape this function will time, every time it's called, fixes
    # that.
    llm.reset()
    _eval_sync(llm, prefix_tokens + suffix_tokens)  # warms the COLD shape
    llm.reset()
    _eval_sync(llm, prefix_tokens)
    _eval_sync(llm, suffix_tokens)  # warms the WARM (suffix-only-timed) shape

    for _ in range(trials):
        # COLD: everything recomputed from scratch.
        llm.reset()
        t0 = _now_ms()
        _eval_sync(llm, prefix_tokens + suffix_tokens)
        cold_times.append(_now_ms() - t0)

        # WARM: prefix already cached (untimed), only suffix is new.
        llm.reset()
        _eval_sync(llm, prefix_tokens)  # untimed -- simulates a cache hit already in place
        t0 = _now_ms()
        _eval_sync(llm, suffix_tokens)  # continues from the cached prefix position
        warm_suffix_times.append(_now_ms() - t0)

    cold_mean = statistics.mean(cold_times)
    warm_mean = statistics.mean(warm_suffix_times)
    savings_pct = (cold_mean - warm_mean) / cold_mean * 100

    return {
        "prefix_tokens": len(prefix_tokens),
        "suffix_tokens": len(suffix_tokens),
        "cold_ms": {"mean": cold_mean, "std": statistics.stdev(cold_times), "trials_ms": cold_times},
        "warm_suffix_only_ms": {"mean": warm_mean, "std": statistics.stdev(warm_suffix_times),
                                 "trials_ms": warm_suffix_times},
        "savings_pct": savings_pct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-tokens", type=int, default=200)
    ap.add_argument("--suffix-tokens", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--out", default="baseline/results_llamacpp_prefix_cache.json")
    args = ap.parse_args()

    from llama_cpp import Llama

    model_path = str(MODEL_DIR / MODEL_FILE)
    print(f"[load] {model_path}")
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=args.n_ctx, verbose=False)

    prefix_tokens = _build_tokens(llm, PREFIX_FILLER, args.prefix_tokens)

    results = {
        "meta": {
            "model_file": MODEL_FILE,
            "prefix_tokens": len(prefix_tokens),
            "trials": args.trials,
            "method": "COLD: llm.reset() + eval(prefix+suffix) combined, timed "
                      "in full. WARM: llm.reset() + eval(prefix) UNTIMED, then "
                      "eval(suffix) WITHOUT reset, timed -- measures only the "
                      "incremental suffix cost with the prefix already cached.",
        },
        "points": [],
    }

    for suffix_n in args.suffix_tokens:
        suffix_tokens = _build_tokens(llm, SUFFIX_FILLER, suffix_n)
        print(f"\n[suffix={suffix_n}] measuring...")
        r = measure_one(llm, prefix_tokens, suffix_tokens, args.trials)
        print(f"  cold={r['cold_ms']['mean']:.1f}ms  "
              f"warm(suffix-only)={r['warm_suffix_only_ms']['mean']:.1f}ms  "
              f"savings={r['savings_pct']:.1f}%")
        results["points"].append(r)

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
