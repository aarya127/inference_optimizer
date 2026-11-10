# Tier 7: prefix-cache savings, and a real GPU-synchronization bug

## The technique: real, measured prefix-cache reuse savings

`PrefixCacheTechnique` answers a question nothing in this repo had asked
before: if a request shares a cached prefix (e.g. a system prompt) with a
prior request, how much prefill time does reusing that cache actually
save? Like `ThroughputOptimalBatchTechnique`, it only ever reports real
measured savings at suffix lengths actually measured — no interpolation,
no extrapolation.

`baseline/measure_llamacpp_prefix_cache.py` measures this directly:
COLD = `reset()` + `eval(prefix + suffix)` combined, timed in full. WARM =
`reset()` + `eval(prefix)` (untimed, simulates a cache hit already in
place) + `eval(suffix)` without resetting (timed — appended to the
existing cache). Real savings = `(cold - warm) / cold`.

**Final, correct result** (SmolLM2-360M, llama.cpp, 200-token prefix):

| suffix tokens | cold (ms) | warm (ms) | savings |
|---|---|---|---|
| 50 | 100.1 | 32.1 | **67.9%** |
| 100 | 132.1 | 56.4 | **57.3%** |
| 200 | 175.2 | 94.5 | **46.0%** |

Savings decrease as the suffix grows relative to the fixed prefix — makes
sense: the prefix's absolute savings (~68–81ms, roughly constant) become a
smaller fraction of total cost as the suffix itself gets more expensive.
All three points are internally consistent: cold time increases
monotonically with total tokens, warm time increases monotonically with
suffix length, and warm is always substantially below cold.

## The real finding: this result is not what the first attempt measured

The first version of this measurement produced numbers that were wrong in
an interesting and important way, caught in two stages.

**Stage 1 — missing per-shape warm-up.** The very first attempt warmed up
only one shape (prefix+suffix[0]) at the very start, then measured three
different suffix lengths without warming each one. Result: warm time came
out ~84ms *regardless of suffix length* (50, 100, or 200 tokens) — a
physical impossibility, and cold times were non-monotonic across
increasing total token counts. Fixed by warming up every (shape,
cold-vs-warm) combination before its own timed trials, matching every
other measurement script in this repo's established convention
(`baseline/measure_v2.py`, `core/calibrate.py`, etc.).

**Stage 2 — a real GPU-dispatch synchronization bug, found after the
warm-up fix didn't fully resolve it.** Even with per-shape warm-up, warm
time stayed suspiciously constant (~84ms) across all three suffix
lengths, and cold time for the shortest suffix measured an implausible
27.7ms — while an isolated, non-interleaved diagnostic of the exact same
token sequences measured a correct, scaling cold time of ~95-98ms. The
anomaly was reproduced in isolation and traced definitively: **it only
appears when COLD and WARM timings are interleaved in the same trial
loop**, not when either is measured alone, repeatedly, in its own loop.

The cause: `llama_decode()` (the C function underneath
`llama-cpp-python`'s `Llama.eval()`) can return to Python before the
underlying Metal GPU work has actually finished. Unsynchronized
`t0 = now(); llm.eval(tokens); dt = now() - t0` timing then silently
misattributes one call's GPU work to whichever call happens to force
synchronization next (reading a logit — needed to pick a token — is a
real sync point; nothing else in a bare `eval()` call is). When every
timed call in a loop is the *same* shape, this stays invisible (there's
nothing for the pipeline bleed-over to visibly corrupt relative to a
correct baseline). Interleaving two *different* shapes exposed it
immediately and unambiguously. Confirmed and fixed by calling
`llm._ctx.get_logits_ith(0)` after every timed and untimed `eval()` call
(`_sync()` in the script) — with it, cold/warm timings become exactly what
an isolated, non-interleaved measurement independently confirms.

**This is the same class of bug, independently rediscovered, that
originally motivated this entire project's honesty rewrite on the MLX
side** (README.md: "the old script's argument-less `mx.eval()` [...]
synchronized nothing"). Uncontrolled lazy/async execution silently
corrupting stage-attribution is apparently not a one-framework problem —
it is a generic hazard of benchmarking any backend with asynchronous
dispatch (MLX's lazy evaluation graph, here Metal command buffers behind
`llama_decode()`), and the fix is the same in spirit both times: force a
real synchronization point and never trust a timer around a call that
might not have actually finished.

## Resolved: audit of this project's other llama.cpp timing scripts

This section originally flagged `baseline/measure_llamacpp.py`,
`core/calibrate.py`, `baseline/measure_quantization_tradeoff.py`, and
`baseline/measure_llamacpp_concurrency.py` as an open, unverified risk.
The audit is done; full detail in
[SYNC_AUDIT_FINDINGS.md](SYNC_AUDIT_FINDINGS.md). In brief:

- **Prefill timing was not materially affected.** Same-shape loops
  self-correct on the mean (each call waits for the previous one's tail),
  but only by accident of call order. All call sites now sync explicitly.
- **Decode timing was affected.** The first decode step absorbed the whole
  preceding prefill, inflating mean TBT in proportion to context length
  (up to ~85%). That produced a fake per-context-token decode slope in both
  llama.cpp profiles (Qwen: 0.00446 → 0.00167 ms/token; SmolLM2: 0.0139 →
  flat), and through that, Tier 6's MODELED concurrency baseline.
- **The quantization script had a different bug:** it timed prefill on its
  `logits_all=True` perplexity instance, so every timed `eval()` included a
  ~620 MB logits copy. Fixed, and the corrected numbers reverse the
  published "quantization speeds up prefill slightly" finding.
- **Recalibration exposed a real cost-model blind spot** (n_batch chunk
  steps the calibration grid never samples) that the old, noisier Qwen
  margin had been covering. See SYNC_AUDIT_FINDINGS.md §6.

## What this doesn't do

- Only one profile (`smollm2_360m_llamacpp_profile`) has real prefix-cache
  data; `PrefixCacheTechnique` reports "not applicable" for every other
  profile, honestly.
- Only three suffix lengths at one fixed prefix length were measured; the
  technique does not interpolate or extrapolate beyond them.
- Does not measure prefix caching across requests in a real multi-request
  server (e.g. two different HTTP requests sharing a system prompt) — this
  is a single-process, single-context simulation of that scenario.
