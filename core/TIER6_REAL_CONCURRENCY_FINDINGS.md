# Tier 6: real concurrent-decode measurement vs the MODELED prediction

`core/techniques/concurrency_throughput.py`'s `ConcurrencyThroughputTechnique`
was built entirely from a MODELED extension of batch=1 decode measurements
(every decode measurement in this repo was batch=1 only) — it was never
checked against a real multi-sequence run. This closes that gap with
`baseline/measure_llamacpp_concurrency.py`: B independent llama.cpp
contexts (SmolLM2-360M-Instruct-Q4_K_M, Metal offload), each prefilled to
ctx=256, then B Python threads each decoding their own tokens
concurrently. Effective TBT/sequence = wall-clock time for all B threads
to complete 30 steps, divided by 30 — i.e. how long one of the B streams
actually waits between its own tokens, the exact quantity the technique
predicts.

**This is not llama.cpp's native multi-sequence batching** (one shared KV
cache, one `llama_decode` call per step across all sequences via the
low-level `llama_batch` API) — that would need direct ctypes-level work
not attempted here. This measures a different, still real and relevant
scenario: B independent model copies/contexts sharing one GPU via naive
Python threading, which is closer to "run B separate llama.cpp processes"
than to a continuous-batching server.

## Result: the model was wildly too optimistic

| B | MODELED TBT/seq (ms) | MEASURED TBT/seq (ms) | measured/modeled |
|---|---|---|---|
| 1 | 11.5 | ~7.25 (steady-state; one outlier trial of 53.7ms excluded, see below) | 0.63x |
| 2 | 11.6 | 13.74 | 1.18x |
| 4 | 11.8 | 27.17 | **2.30x** |

The model predicts **near-flat** scaling (11.5ms → 11.8ms, a 2.7% increase
from B=1 to B=4) because it only accounts for KV-cache-bandwidth
contention, which is genuinely tiny for a 360M model at ctx=256
(0.1ms/extra sequence). Reality shows **close to linear** scaling instead:
measured TBT roughly doubles from B=1 to B=2, and roughly triples-to-quadruples
from B=1 to B=4 — consistent with little-to-no real concurrency benefit in
this setup, not the near-total parallelism the model assumed.

**Likely cause, not confirmed**: either (a) `llama-cpp-python`'s ctypes
call into `llama_decode` does not release the Python GIL for long enough
to let threads interleave effectively, or (b) Metal has a single command
queue per process and the two contexts' GPU work serializes there
regardless of Python-level threading, or (c) both. This script does not
instrument GPU utilization or GIL hold time, so it cannot distinguish
these — reported as an open question, not resolved.

## A genuine measurement artifact, disclosed

B=1's first trial measured 53.7ms vs. the next four trials' tight
7.07-7.39ms cluster — a >7x outlier. The untimed prefill-to-ctx warm-up
done before timing starts uses the prefill code path
(`llm.eval(tokens)`), not the single-token decode path
(`llm.sample()` + `llm.eval([tok])`) the timed loop actually uses, so the
first-ever decode call in a fresh thread may absorb some one-time
Metal/thread-startup cost the warm-up didn't cover. The reported B=1 mean
above excludes this outlier (using the steady-state trials instead);
`baseline/results_llamacpp_concurrency.json` keeps the raw trial, not
smoothed over.

## Conclusion: don't trust `ConcurrencyThroughputTechnique`'s numbers for capacity planning

Checked against both an independent-threads proxy (2.3x too optimistic at
B=4) and, now, the real native-batching scenario it was actually meant to
predict (1.72x too optimistic at B=4, 3.26x at B=8, and growing) —
`concurrency_throughput.py`'s own docstring and `core/README.md` are
updated to point here. The technique itself is NOT rewritten to fit these
measurements — that would mean fitting a formula to 4 (B, ctx) points at
one model size, which risks being just as wrong for a different model or
context length (see "What's still open" above). The honest state of this
technique is: it is a documented extrapolation that has now been checked
against reality twice, by two different real methods, and found to
understate real-world contention substantially both times — treat its
output as a lower bound on TBT degradation, not an estimate, until a
proper compute-aware cost model replaces it.

## Follow-up: real native batching, via the low-level `llama_batch` API

The independent-contexts experiment above explicitly was not the scenario
`ConcurrencyThroughputTechnique` is meant to predict (continuous-batching
servers share one KV cache and submit one combined `llama_decode` call per
step across sequences). `baseline/measure_llamacpp_native_batching.py`
measures that scenario for real, using llama.cpp's low-level `llama_batch`
API directly: one context, one shared KV cache sliced across B sequence
IDs, one `llama_decode` call per step covering all B sequences.

**This required a real workaround, documented for anyone extending this
further**: `llama-cpp-python`'s high-level `Llama` class hardcodes
`n_seq_max=1` (the number of independent sequences one context supports)
unless `embedding=True`, with no public constructor parameter to raise it.
The only way found to override it without reimplementing `Llama.__init__`
is to monkey-patch `llama_cpp.llama_cpp.llama_context_default_params`
(the exact submodule `llama.py` imports it as — patching the top-level
`llama_cpp.llama_context_default_params` does nothing, since
`from X import *`-style re-exports don't alias back) for the duration of
construction. See `_make_multi_seq_llama()` in the script.

**Correctness was verified before trusting any timing**, not assumed: two
sequences given different prompts ("The capital of France is" / "The
largest planet is") correctly and independently completed as "...Paris..."
and "...Jupiter, which is about 1[1 times the size of Earth]...",
including over multiple subsequent decode steps, with zero cross-sequence
contamination. A first version of the script also crashed outright at
B=8 with a native `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` abort
(from prefilling all B sequences in one combined batch exceeding
`n_batch`) — a real bug, fixed by prefilling one sequence per
`llama_decode` call instead of all B combined, with incremental JSON
writes added afterward so a future crash at one B doesn't lose already-
measured results from lower B values, since a native abort skips any
Python code after it, including a final save.

### Three-way result

| B | MODELED (ms) | measured: independent threads (ms) | measured: native batched (ms) | native/modeled |
|---|---|---|---|---|
| 1 | 11.50 | 16.54 (outlier-skewed; ~7.25 steady-state) | 7.95 | 0.69x |
| 2 | 11.60 | 13.74 | 12.18 | 1.05x |
| 4 | 11.81 | 27.17 | 20.31 | 1.72x |
| 8 | 12.23 | *(not tested)* | 39.92 | **3.26x** |

**Native batching is real and helps** — it is consistently and
substantially faster than the independent-threads proxy at every B tested
(e.g. 20.31ms vs 27.17ms at B=4), confirming that real GPU-level batching
does provide a genuine throughput benefit over naively running independent
contexts. But it is **still far from the MODELED "near-flat" prediction**:
3.26x worse at B=8, and growing with B, not shrinking.

A quick linear fit to the native-batched data (TBT(B) ≈ 3.4 + 4.6×B ms)
fits all four points well (predicts 12.5ms at B=2, 21.7ms at B=4, vs.
measured 12.2ms and 20.3ms) — a per-extra-sequence marginal cost around
**4.6ms**, roughly **44x** the model's assumed 0.105ms/sequence
KV-bandwidth-only cost. This points to the actual mechanism the model
misses: decode for a 360M model at small batch sizes is not purely
memory-bandwidth-bound the way the model assumes — each additional
sequence in the batch adds real per-sequence compute (a full forward pass
worth of FLOPs), and at these batch sizes that compute cost, not KV-cache
bandwidth, is the dominant term. This is the same roofline
compute-vs-bandwidth distinction `simulation/w4a8_quantizer.py`'s honesty
rewrite already identified for this repo's *other* decode model
(`core/TIER2_LLAMACPP_FINDINGS.md`'s quantization findings) — same
underlying physics, independently rediscovered here via direct
measurement rather than re-used as an assumption.

## What's still open

1. ~~Measure real server-side continuous batching~~ — done above, via the
   low-level `llama_batch` API. Confirms the gap is real, not an artifact
   of the independent-threads proxy (smaller gap, same direction, still
   substantial and growing with B).
2. A proper compute-vs-bandwidth cost model for decode (e.g.
   `TBT(ctx, B) ≈ overhead + per_ctx_token*ctx + per_sequence_compute*B`,
   fit to real multi-B measurements like these) would fix
   `ConcurrencyThroughputTechnique`'s core assumption. Not attempted here
   — fitting a corrected formula to only 4 (B, ctx) points at one model
   size and one context length risks the same overfitting-to-one-setup
   problem already flagged for the margin fix elsewhere in this repo; it
   would need a proper multi-ctx, multi-model sweep first.
3. Repeat at larger B, larger ctx, and a second model to see whether the
   ~4.6ms/sequence marginal cost found here is a property of this model
   size or holds more generally.
4. Instrument GPU utilization (e.g. `powermetrics`) to directly attribute
   the per-sequence marginal cost to compute vs. any remaining
   coordination/dispatch overhead, rather than inferring it from the
   roofline argument alone.
