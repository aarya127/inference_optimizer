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

Given this real/modeled gap (as much as 2.3x too optimistic at B=4, in
this one setup), `core/techniques/concurrency_throughput.py`'s own
docstring and `core/README.md` are updated to point here. The technique
itself is NOT rewritten to fit this one real measurement — that would
mean re-calibrating a formula to a single (model, backend, ctx, threading
strategy) data point, which risks being just as wrong for a different
setup (a real multi-sequence batched server, a different model size, a
different ctx). The honest state of this technique is: it is a documented
extrapolation that has now been checked once against reality and found to
understate real-world contention substantially — treat its output as a
lower bound on TBT degradation, not an estimate, until it is recalibrated
against real multi-sequence serving data (ideally via llama.cpp's actual
`llama_batch` API, which was out of scope here).

## What would resolve this properly

1. Measure real server-side continuous batching (llama.cpp's low-level
   `llama_batch` API, or just running `llama-server` and sending B
   concurrent HTTP requests) instead of B independent contexts — this
   is the scenario `ConcurrencyThroughputTechnique` is actually meant to
   predict, and this experiment did not test it.
2. Instrument GPU utilization (e.g. `powermetrics`) during the concurrent
   run to distinguish GIL-bound from GPU-queue-bound serialization.
3. Repeat at larger B and larger ctx to see whether the measured/modeled
   gap grows, shrinks, or stays proportionally similar.
