"""Real, measured quantization-level tradeoff: fp16 vs Q8_0 vs Q4_K_M.

This repo's own REVIEW_FINDINGS.md flagged that every quantization "result"
in the original study was fabricated or double-counted (a manufactured 2x
FP8-on-M3 figure with no such GPU path, W4A8 applied on top of a baseline
that was already 4-bit). This script measures a REAL quantization tradeoff
end-to-end: same model (Qwen2.5-0.5B-Instruct), same backend (llama.cpp,
Metal offload), three real GGUF quantization levels, each honestly measured
for (a) file size on disk, (b) prefill speed, (c) decode speed, and (d) a
real perplexity proxy for quality -- not fabricated, not a proxy like
n_crops*keep_ratio.

Perplexity method: manual, understood computation from raw per-position
logits (`Llama(..., logits_all=True)`, then `llm.scores[i]` predicts token
i+1). NOTE: llama-cpp-python's high-level `create_completion(echo=True,
logprobs=1)` was tried first and rejected -- it returned 569 "token"
logprobs for a 23-token sentence (confirmed via llm.tokenize()), an
unexplained discrepancy not worth trusting. The manual approach reproduces
a sane perplexity (~12 for a generic English sentence under Q4_K_M) and is
fully auditable.

This is a quality PROXY (perplexity on one fixed passage), not a task
benchmark (no VQA/accuracy-style eval exists here either) -- stated
explicitly, per this repo's convention of not overclaiming what a number
measures.

Usage (from repo root, using the phase-0 venv):
    scope/venv_phase0/bin/python baseline/measure_quantization_tradeoff.py \
        [--trials 8] [--out baseline/results_quantization_tradeoff.json]
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
QUANT_LEVELS = {
    "fp16":    "qwen2.5-0.5b-instruct-fp16.gguf",
    "q8_0":    "qwen2.5-0.5b-instruct-q8_0.gguf",
    "q4_k_m":  "qwen2.5-0.5b-instruct-q4_k_m.gguf",
}

TOKEN_COUNTS = [256, 1024]

# Fixed public-domain reference passage (opening of Jane Austen's "Pride and
# Prejudice", public domain) for the perplexity proxy -- same text scored
# under every quant level so the comparison is apples-to-apples.
REFERENCE_TEXT = (
    "It is a truth universally acknowledged, that a single man in "
    "possession of a good fortune, must be in want of a wife. However "
    "little known the feelings or views of such a man may be on his "
    "first entering a neighbourhood, this truth is so well fixed in the "
    "minds of the surrounding families, that he is considered as the "
    "rightful property of some one or other of their daughters. My dear "
    "Mr. Bennet, said his lady to him one day, have you heard that "
    "Netherfield Park is let at last? Mr. Bennet replied that he had "
    "not. But it is, returned she; for Mrs. Long has just been here, and "
    "she told me all about it."
)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _summ(xs):
    return {
        "mean_ms": round(statistics.mean(xs), 2),
        "std_ms": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
        "min_ms": round(min(xs), 2),
    }


def measure_perplexity(llm) -> float:
    """Manual, auditable perplexity: log_softmax of raw logits at each
    position predicting the next real token, averaged, negated, exponentiated."""
    toks = llm.tokenize(REFERENCE_TEXT.encode("utf-8"), add_bos=True)
    llm.reset()
    llm.eval(toks)
    n = llm.n_tokens
    logprobs = []
    for i in range(n - 1):
        logits = llm.scores[i]
        m = float(np.max(logits))
        logsumexp = m + float(np.log(np.sum(np.exp(logits - m))))
        logprobs.append(float(logits[toks[i + 1]]) - logsumexp)
    return float(np.exp(-np.mean(logprobs)))


def build_filler_tokens(llm, n_tokens: int) -> list[int]:
    filler = ("Please look closely at the background, the foreground, the "
              "colors, and the overall composition before answering. ")
    text = filler * (n_tokens // 8 + 5)
    toks = llm.tokenize(text.encode("utf-8"), add_bos=True)
    if len(toks) < n_tokens:
        raise ValueError(f"filler text too short for n_tokens={n_tokens}")
    return toks[:n_tokens]


def measure_one_level(name: str, path: Path, n_ctx: int, trials: int) -> dict:
    from llama_cpp import Llama

    file_size_mb = path.stat().st_size / 2**20
    print(f"\n=== {name}  ({file_size_mb:.1f} MiB on disk) ===")

    t0 = _now_ms()
    llm = Llama(model_path=str(path), n_gpu_layers=-1, n_ctx=n_ctx,
                verbose=False, logits_all=True)
    load_ms = _now_ms() - t0
    print(f"  load: {load_ms:.0f} ms")

    perplexity = measure_perplexity(llm)
    print(f"  perplexity (fixed reference passage): {perplexity:.3f}")

    prefill_by_n = {}
    for n_tok in TOKEN_COUNTS:
        tokens = build_filler_tokens(llm, n_tok)
        # per-shape warm-up (discarded)
        llm.reset()
        llm.eval(tokens)

        times = []
        for _ in range(trials):
            llm.reset()
            t0 = _now_ms()
            llm.eval(tokens)
            times.append(_now_ms() - t0)
        prefill_by_n[str(n_tok)] = _summ(times)
        print(f"  prefill N={n_tok}: {_summ(times)['mean_ms']:.1f} ms")

    # Decode TBT at ctx = TOKEN_COUNTS[-1]
    tbt = []
    for _ in range(32):
        t0 = _now_ms()
        tok = llm.sample()
        llm.eval([tok])
        tbt.append(_now_ms() - t0)
    tbt_summary = _summ(tbt)
    print(f"  decode TBT: {tbt_summary['mean_ms']:.1f} ms")

    del llm

    return {
        "file_size_mb": round(file_size_mb, 1),
        "load_ms": round(load_ms, 1),
        "perplexity": round(perplexity, 4),
        "prefill_by_n_tokens": prefill_by_n,
        "decode_tbt": tbt_summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--out", default="baseline/results_quantization_tradeoff.json")
    args = ap.parse_args()

    results = {
        "meta": {
            "model": "Qwen2.5-0.5B-Instruct",
            "backend": "llama.cpp (llama-cpp-python), Metal GPU offload",
            "quant_levels": list(QUANT_LEVELS.keys()),
            "trials": args.trials,
            "reference_passage": "Pride and Prejudice, opening paragraph "
                                  "(public domain)",
            "perplexity_method": "manual log_softmax over raw per-position "
                                  "logits (logits_all=True); NOT the "
                                  "high-level create_completion(echo=True, "
                                  "logprobs=1) API, which returned an "
                                  "unexplained 569 'tokens' for a 23-token "
                                  "sentence during development and was "
                                  "rejected as untrustworthy.",
        },
        "levels": {},
    }

    for name, filename in QUANT_LEVELS.items():
        path = MODEL_DIR / filename
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        results["levels"][name] = measure_one_level(
            name, path, args.n_ctx, args.trials)

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {out}")

    print(f"\n{'level':8s} {'size (MiB)':>11s} {'perplexity':>11s} "
          f"{'prefill@1024 (ms)':>18s} {'TBT (ms)':>9s}")
    for name, e in results["levels"].items():
        print(f"{name:8s} {e['file_size_mb']:11.1f} {e['perplexity']:11.3f} "
              f"{e['prefill_by_n_tokens']['1024']['mean_ms']:18.1f} "
              f"{e['decode_tbt']['mean_ms']:9.1f}")


if __name__ == "__main__":
    sys.exit(main())
