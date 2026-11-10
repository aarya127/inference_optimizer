"""Real measured concurrent-decode throughput, to check
core/techniques/concurrency_throughput.py's MODELED prediction against
actual hardware behavior -- that technique has never been validated
against a real multi-sequence run; it only extends the measured batch=1
decode cost with a theoretical per-extra-sequence KV-bandwidth term.

Method: B independent llama.cpp contexts (same model, same n_gpu_layers=-1
Metal offload), each prefilled to the same context length, then B Python
threads each repeatedly decode their own next token with no coordination
between them beyond a shared start signal. llama-cpp-python's C calls
release the GIL during llama_decode, so if there is real GPU-level
parallelism/contention to observe, concurrent wall-clock time should
reveal it; if llama.cpp or Metal serializes these calls instead, that is
also a real, honest finding (not what the MODELED formula assumes).

Effective per-sequence TBT at concurrency B is defined as:
    (wall-clock time for all B threads to complete N_STEPS decode steps
     each) / N_STEPS
-- i.e. how long one of the B concurrent streams actually waits between
its own tokens, which is the quantity core/techniques/concurrency_throughput.py
predicts and core/optimize.py-style recommendations would be acted on.

Uses the smallest cached model (SmolLM2-360M-Instruct-Q4_K_M, ~270MB) and
a modest context length to keep B independent model copies within this
8GB M3's memory budget -- each concurrent instance loads its own full copy
of the weights plus its own KV cache; this is NOT llama.cpp's native
multi-sequence batching (one shared KV cache, one GPU call per step across
sequences), which would need the low-level llama_batch API. This measures
"what happens if you run B independent llama.cpp processes/contexts on
one GPU", which is still a real and useful number, just a different
scenario from server-side continuous batching -- labeled as such below.

Usage:
    scope/venv_phase0/bin/python baseline/measure_llamacpp_concurrency.py \
        [--concurrency 1 2 4] [--ctx-tokens 256] [--n-steps 30]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "inference_optimizer_models"
MODEL_FILE = "SmolLM2-360M-Instruct-Q4_K_M.gguf"
FILLER = ("The quick brown fox jumps over the lazy dog near the river bank "
          "while the sun sets slowly behind the distant mountains. ")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _build_tokens(llm, n_tokens: int) -> list[int]:
    text = FILLER * (n_tokens // 8 + 5)
    toks = llm.tokenize(text.encode("utf-8"), add_bos=True)
    return toks[:n_tokens]


def _sync(llm) -> None:
    """llama_get_logits_ith() calls llama_synchronize(): blocks until this
    context's in-flight GPU work is done. See core/SYNC_AUDIT_FINDINGS.md."""
    llm._ctx.get_logits_ith(-1)


WARMUP_STEPS = 5


def measure_concurrency(concurrency: int, ctx_tokens: int, n_steps: int, n_trials: int) -> dict:
    from llama_cpp import Llama

    model_path = MODEL_DIR / MODEL_FILE
    # The KV cache is NOT reset between trials (each trial keeps decoding
    # where the last stopped), so size for prefill + warm-up + all trials
    # explicitly rather than relying on llama.cpp padding n_ctx upward.
    n_ctx = ctx_tokens + WARMUP_STEPS + n_steps * n_trials + 16

    print(f"\n[B={concurrency}] loading {concurrency} independent context(s)...")
    llms = [Llama(model_path=str(model_path), n_gpu_layers=-1, n_ctx=n_ctx, verbose=False)
            for _ in range(concurrency)]

    tokens = _build_tokens(llms[0], ctx_tokens)
    for llm in llms:
        llm.reset()
        llm.eval(tokens)  # untimed prefill to the common context length
        _sync(llm)  # so no prefill work is still in flight when trial 1's timer starts

    def worker(llm, steps: int):
        tok = llm.sample()
        for _ in range(steps):
            llm.eval([tok])
            tok = llm.sample()

    # Untimed decode warm-up at this B: the prefill above exercises the
    # prompt-processing path, not the single-token decode path the timed
    # loop uses. Without this, trial 1 measured 50-54ms vs a steady 7.2ms
    # at B=1 (and was inflated ~1.5x at B=2/B=4 too) -- see
    # core/SYNC_AUDIT_FINDINGS.md.
    warm = [threading.Thread(target=worker, args=(llm, WARMUP_STEPS)) for llm in llms]
    for t in warm:
        t.start()
    for t in warm:
        t.join()

    trial_tbt_ms = []
    for trial in range(n_trials):
        threads = [threading.Thread(target=worker, args=(llm, n_steps)) for llm in llms]
        t0 = _now_ms()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall_ms = _now_ms() - t0
        effective_tbt_ms = wall_ms / n_steps
        trial_tbt_ms.append(effective_tbt_ms)
        print(f"  trial {trial + 1}: wall={wall_ms:.1f}ms over {n_steps} steps "
              f"-> effective TBT/sequence={effective_tbt_ms:.2f}ms")

    for llm in llms:
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
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--ctx-tokens", type=int, default=256)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--out", default="baseline/results_llamacpp_concurrency.json")
    args = ap.parse_args()

    print(f"=== Real concurrent-decode measurement: {MODEL_FILE} ===")
    print(f"concurrency levels: {args.concurrency}, ctx_tokens={args.ctx_tokens}, "
          f"n_steps={args.n_steps}, n_trials={args.n_trials}")

    results = {
        "meta": {
            "model_file": MODEL_FILE,
            "method": "B independent llama.cpp contexts (own weights copy, own "
                      "KV cache each), B Python threads decoding concurrently "
                      "with no server-side batching -- NOT llama.cpp's native "
                      "multi-sequence batched llama_decode. Effective TBT/"
                      "sequence = wall time for B threads x n_steps steps / n_steps. "
                      "Prefill synced and an untimed decode warm-up run before "
                      "the first timed trial (see core/SYNC_AUDIT_FINDINGS.md).",
        },
        "runs": [],
    }

    for b in args.concurrency:
        r = measure_concurrency(b, args.ctx_tokens, args.n_steps, args.n_trials)
        results["runs"].append(r)

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {out}")

    print(f"\n{'B':>3s} {'effective TBT/seq (ms)':>22s} {'std':>8s}")
    for r in results["runs"]:
        print(f"{r['concurrency']:3d} {r['effective_tbt_ms']['mean']:22.2f} "
              f"{r['effective_tbt_ms']['std']:8.2f}")


if __name__ == "__main__":
    sys.exit(main())
