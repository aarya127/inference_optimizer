"""Real llama.cpp NATIVE multi-sequence batched decode measurement -- the
scenario core/techniques/concurrency_throughput.py is actually meant to
predict, and the thing baseline/measure_llamacpp_concurrency.py's
independent-contexts-plus-threads approach did NOT test (see
core/TIER6_REAL_CONCURRENCY_FINDINGS.md's "what would resolve this
properly" section).

This uses ONE context with ONE shared KV cache sliced across B sequence
IDs via llama.cpp's low-level llama_batch API (llama_batch_init /
llama_decode with per-token seq_id), not B independent model copies.
Concretely: B sequences are prefilled into one context (one add_sequence()
call per sequence), then N_STEPS decode rounds each submit exactly one
token per sequence in a SINGLE combined llama_decode() call -- this is the
real mechanism continuous-batching servers use to get GPU throughput
gains from batching, unlike the earlier independent-threads experiment.

Gotcha discovered and worked around: llama-cpp-python's high-level Llama
class has no public constructor parameter for n_seq_max (the number of
independent sequences one context supports); it is hardcoded to 1 unless
embedding=True. llama_context_default_params() is called as
`llama_cpp.llama_cpp.llama_context_default_params` inside llama.py (that
module does `import llama_cpp.llama_cpp as llama_cpp`), so the ONLY way to
raise n_seq_max without subclassing/reimplementing __init__ is to
monkey-patch that exact submodule attribute for the duration of
construction, as done in _make_multi_seq_llama() below. This was verified
correct (not just crash-free) before trusting any timing: two sequences
given different prompts ("The capital of France is" / "The largest planet
is") correctly and independently completed as "...Paris..." and
"...Jupiter...", including over multiple subsequent decode steps, with no
cross-sequence contamination, before this script was written.

Usage:
    scope/venv_phase0/bin/python baseline/measure_llamacpp_native_batching.py \
        [--concurrency 1 2 4 8] [--ctx-tokens 256] [--n-steps 30] [--n-trials 5]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

MODEL_DIR = Path.home() / ".cache" / "inference_optimizer_models"
MODEL_FILE = "SmolLM2-360M-Instruct-Q4_K_M.gguf"
FILLER = ("The quick brown fox jumps over the lazy dog near the river bank "
          "while the sun sets slowly behind the distant mountains. ")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _make_multi_seq_llama(model_path: str, n_seq_max: int, n_ctx: int, n_batch: int):
    """See module docstring: the only way to raise n_seq_max above the
    hardcoded default of 1 without reimplementing Llama.__init__."""
    import llama_cpp.llama_cpp as llama_cpp_bindings
    from llama_cpp import Llama

    orig = llama_cpp_bindings.llama_context_default_params

    def patched():
        params = orig()
        params.n_seq_max = n_seq_max
        return params

    llama_cpp_bindings.llama_context_default_params = patched
    try:
        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=n_ctx,
                    n_batch=n_batch, verbose=False)
    finally:
        llama_cpp_bindings.llama_context_default_params = orig

    assert llm.context_params.n_seq_max == n_seq_max, (
        f"n_seq_max patch did not take effect: got "
        f"{llm.context_params.n_seq_max}, expected {n_seq_max}"
    )
    return llm


def _build_tokens(llm, n_tokens: int) -> list[int]:
    text = FILLER * (n_tokens // 8 + 5)
    toks = llm.tokenize(text.encode("utf-8"), add_bos=True)
    return toks[:n_tokens]


def measure_native_batching(concurrency: int, ctx_tokens: int, n_steps: int, n_trials: int) -> dict:
    from llama_cpp._internals import LlamaBatch

    model_path = str(MODEL_DIR / MODEL_FILE)
    n_ctx_total = (ctx_tokens + n_steps + 16) * concurrency  # headroom per sequence
    # n_batch bounds tokens in a SINGLE llama_decode call. The prefill call
    # below submits one sequence's ctx_tokens at a time (not all B
    # sequences combined), so n_batch only needs to cover one sequence's
    # prefill plus the B-token decode-step batch -- not ctx_tokens*B. An
    # earlier version prefilled all B sequences in one combined batch and
    # hit GGML_ASSERT(n_tokens_all <= cparams.n_batch) at B=8 (2048 tokens
    # > the then-512 n_batch), crashing the process via a native abort
    # with nothing flushed/saved -- see git history / TIER6 findings.
    n_batch = max(ctx_tokens, concurrency, 512)

    print(f"\n[B={concurrency}] one context, n_seq_max={concurrency}, "
          f"n_ctx={n_ctx_total}, n_batch={n_batch}...", flush=True)
    llm = _make_multi_seq_llama(model_path, concurrency, n_ctx_total, n_batch)
    n_vocab = llm._model.n_vocab()

    def get_logits(row: int):
        return np.ctypeslib.as_array(llm._ctx.get_logits_ith(row), shape=(n_vocab,))

    tokens = _build_tokens(llm, ctx_tokens)

    trial_tbt_ms = []
    for trial in range(n_trials):
        # Fresh prefill each trial: clear each sequence's KV cache range.
        for seq_id in range(concurrency):
            llm._ctx.kv_cache_seq_rm(seq_id, 0, -1)

        # One llama_decode call per sequence (not all B combined into one
        # call): each call only needs n_batch >= ctx_tokens, not
        # ctx_tokens*B, and this is also closer to how sequences actually
        # arrive in a real server (not necessarily all at once).
        prefill_batch = LlamaBatch(n_tokens=ctx_tokens, embd=0,
                                    n_seq_max=concurrency, verbose=False)
        cur_tok = [0] * concurrency
        for seq_id in range(concurrency):
            prefill_batch.reset()
            prefill_batch.add_sequence(tokens, seq_id, False)
            llm._ctx.decode(prefill_batch)
            cur_tok[seq_id] = int(np.argmax(get_logits(ctx_tokens - 1)))
        pos = [ctx_tokens] * concurrency

        step_batch = LlamaBatch(n_tokens=concurrency, embd=0, n_seq_max=concurrency, verbose=False)

        t0 = _now_ms()
        for _ in range(n_steps):
            step_batch.reset()
            for seq_id in range(concurrency):
                i = step_batch.batch.n_tokens
                step_batch.batch.token[i] = cur_tok[seq_id]
                step_batch.batch.pos[i] = pos[seq_id]
                step_batch.batch.seq_id[i][0] = seq_id
                step_batch.batch.n_seq_id[i] = 1
                step_batch.batch.logits[i] = True
                step_batch.batch.n_tokens += 1
            llm._ctx.decode(step_batch)
            for seq_id in range(concurrency):
                nxt = int(np.argmax(get_logits(seq_id)))
                cur_tok[seq_id] = nxt
                pos[seq_id] += 1
        wall_ms = _now_ms() - t0

        effective_tbt_ms = wall_ms / n_steps
        trial_tbt_ms.append(effective_tbt_ms)
        print(f"  trial {trial + 1}: wall={wall_ms:.1f}ms over {n_steps} steps "
              f"(each step = 1 llama_decode call covering all {concurrency} "
              f"sequences) -> effective TBT/sequence={effective_tbt_ms:.2f}ms",
              flush=True)

    del llm

    return {
        "concurrency": concurrency,
        "ctx_tokens": ctx_tokens,
        "n_steps": n_steps,
        "effective_tbt_ms": {
            "mean": statistics.mean(trial_tbt_ms),
            "std": statistics.stdev(trial_tbt_ms) if len(trial_tbt_ms) > 1 else 0.0,
            "trials_ms": trial_tbt_ms,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ctx-tokens", type=int, default=256)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--out", default="baseline/results_llamacpp_native_batching.json")
    args = ap.parse_args()

    print(f"=== Real NATIVE multi-sequence batched decode: {MODEL_FILE} ===")
    print(f"concurrency levels: {args.concurrency}, ctx_tokens={args.ctx_tokens}, "
          f"n_steps={args.n_steps}, n_trials={args.n_trials}")

    results = {
        "meta": {
            "model_file": MODEL_FILE,
            "method": "ONE context, ONE shared KV cache sliced across B seq_ids "
                      "via the low-level llama_batch API (llama_batch_init / "
                      "llama_decode with per-token seq_id) -- real continuous-"
                      "batching-style execution, one llama_decode call per step "
                      "covering all B sequences. Verified correct (not just "
                      "crash-free) before use: see module docstring. "
                      "Effective TBT/sequence = wall time for n_steps combined "
                      "decode calls / n_steps.",
        },
        "runs": [],
    }

    out = Path(args.out)
    for b in args.concurrency:
        r = measure_native_batching(b, args.ctx_tokens, args.n_steps, args.n_trials)
        results["runs"].append(r)
        # Write after every concurrency level, not just at the end: a
        # native-code crash (ctypes/ggml abort) does not raise a Python
        # exception and skips any code after it, including a final write
        # -- this is exactly what happened at B=8 on an earlier version of
        # this script, losing B=1/2/4's already-successful results.
        out.write_text(json.dumps(results, indent=2))
        print(f"  [saved progress through B={b} -> {out}]", flush=True)

    print(f"\n[done] wrote {out}")

    print(f"\n{'B':>3s} {'effective TBT/seq (ms)':>22s} {'std':>8s}")
    for r in results["runs"]:
        print(f"{r['concurrency']:3d} {r['effective_tbt_ms']['mean']:22.2f} "
              f"{r['effective_tbt_ms']['std']:8.2f}")


if __name__ == "__main__":
    sys.exit(main())
