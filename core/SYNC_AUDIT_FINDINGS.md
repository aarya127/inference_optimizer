# Sync-bug audit: which llama.cpp numbers were actually affected

Tier 7 (`core/TIER7_PREFIX_CACHE_FINDINGS.md`) found that `llama_decode()`
returns before its Metal GPU work completes. It flagged, without
resolving, whether this repo's other llama.cpp timing scripts were
affected. This is that audit. Short answer: **prefill numbers were not
materially affected; decode numbers were, by up to ~85%, which produced a
fake per-context-token decode slope in both llama.cpp profiles.** One
script had a different, unrelated timing bug. Recalibrating also exposed a
real cost-model blind spot that the old, noisier numbers had been hiding.

All before/after comparisons below were run **in the same process,
back-to-back**, not by diffing against months-old committed numbers, so
hardware or session drift can't masquerade as a bug effect. Harness:
`sync_audit.py` (session scratch, not committed). It ran each script's
original timing pattern, imported from the script itself where possible,
against a synced variant.

## 1. The mechanism, measured directly

llama-cpp-python 0.3.35's `Llama.eval()` never reads logits when
`logits_all=False` (the code that did is commented out, since sampling
moved into `llama_sampler`). Nothing inside `eval()` waits for the GPU.
`llama_get_logits_ith()` calls `llama_synchronize()` first, so it is a
real sync point. `Llama.sample()` goes through it as well.

SmolLM2-360M, time for `eval()` to return vs time until synced:

| N | `eval()` returns | actually done | outstanding at return |
|---|---|---|---|
| 32 | 0.7 ms | 16.9 ms | 96% |
| 256 | 0.9 ms | 94.2 ms | 99% |
| 512 | 1.6 ms | 203.9 ms | 99% |
| 1024 | 205.8 ms | 433.9 ms | 53% |
| 2048 | 703.9 ms | 991.9 ms | 29% |

Up to `n_batch=512`, essentially the whole prefill is still in flight when
`eval()` returns. Above it, `eval()` issues one `llama_decode` per
512-token chunk, each waits for the previous one, and only the last chunk
is outstanding.

## 2. Per-script verdict

| script | pattern | verdict |
|---|---|---|
| `baseline/measure_llamacpp.py` (Qwen, Tier 2) | prefill | **not materially affected** (self-corrects, see below) |
| | decode | **affected**: mean TBT inflated up to ~65–85%, growing with ctx |
| `core/calibrate.py` (SmolLM2, Tier 5) | prefill | **not materially affected** |
| | decode | **affected**: same mechanism; decode slope ~3x too steep |
| `baseline/measure_quantization_tradeoff.py` | prefill/decode | **not affected by the sync bug** (`logits_all=True` makes `eval()` copy logits, which syncs), **but a different bug**: see §4 |
| `baseline/measure_llamacpp_concurrency.py` | decode | **mostly safe** (workers end on `sample()`, which syncs, before `join()`), **but** trial 1 inflated at every B by a missing decode warm-up |
| `core/executors/llamacpp_executor.py` (Tier 4, not on the original list) | prefill | **not materially affected** (same self-correction as prefill above) |
| `baseline/measure_llamacpp_native_batching.py`, `..._prefix_cache.py` | | already safe (read logits / explicit `_sync()`) |
| `core/calibrate_mlx*.py` | | safe: `mx.eval(next_tok)` before timing and inside every timed step |

### Why prefill self-corrected (by accident)

In a same-shape loop (`reset; t0; eval; dt` × N), each `eval()` has to
wait for the previous call's in-flight work before its own can run. The
per-shape warm-up's tail lands in trial 1, and the last trial's tail is
never timed. These cancel, so the **mean** comes out right. Measured
directly (SmolLM2, 16 trials per arm, same process): unsynced and synced
**min** agree within 0.5–4 ms (≤2%) at every N from 32 to 2048. Unsynced
reads ~1–2% *low*, because CPU graph-build for call k+1 overlaps GPU work
for call k. Mean gaps at large N were noise spikes landing in either arm.

This depends entirely on call order. In a probe where trial 1 started
from a clean sync instead of an unsynced warm-up, trial 1 measured ~2 ms
instead of ~97 ms. All prefill call sites are now synced explicitly, so
correctness no longer rests on that accident.

### Why decode did not self-correct

The decode loop is `t0; sample(); eval([tok]); dt`. `sample()` syncs on
the **previous** `eval()`, so each step's time is really the previous
step's GPU work. That averages out over a steady loop, **except for step
1**, which absorbs whatever was in flight before the loop started: the
entire preceding prefill (timed in `measure_llamacpp.py`, untimed in
`calibrate.py`, unsynced in both). That one step is ~prefill-sized, and
prefill grows with ctx, so mean TBT grew with ctx even though real decode
barely does.

Same process, 3 × 32 steps per arm, SmolLM2:

| ctx | unsynced mean | unsynced step 1 | synced mean | inflation |
|---|---|---|---|---|
| 32 | 7.47 | 18.9 | 7.38 | +1% |
| 256 | 10.85 | 98.8 | 8.59 | +26% |
| 384 | 13.42 | 156.4 | 9.26 | +45% |
| 1024 | 17.82 | 253.7 | 9.67 | +84% |
| 2048 | 20.59 | 303.9 | 11.91 | +73% |

Qwen2.5-0.5B shows the same thing (e.g. ctx 1024: 16.05 vs 10.00 ms). The
committed data carried the fingerprint all along: decode std of 17–45 ms
on means of 10–18 ms, i.e. one huge outlier per run.

## 3. Corrected numbers

All five call sites now sync explicitly. Every affected measurement was
re-run.

**Decode cost models (the main correction):**

| profile | before | after |
|---|---|---|
| Qwen2.5-0.5B / llama.cpp (`llamacpp_prefill_fit.json`) | 10.26 + 0.00446·ctx ms | **9.03 + 0.00167·ctx** ms |
| SmolLM2-360M / llama.cpp (`smollm2_360m_llamacpp_profile.json`) | 7.93 + 0.0139·ctx ms | **8.59 flat** (raw fit −0.0053, clamped) |

SmolLM2's raw slope over the calibrated range [32, 384] is now within
noise of zero (−0.005 here, +0.004 in the audit run). Five points over
that narrow range can't resolve a slope this small. Measured directly,
synced TBT is ~11.9 ms at ctx 2048, so a flat 8.6 ms model under-predicts
well outside its calibrated domain. The old 0.0139 slope over-predicted
at the same point (36 ms). Neither should be trusted beyond ctx 384; the
`domain_ctx` field already says so.

**Prefill fits** (re-measured; changes are mostly run-to-run variation,
not the sync fix, per §2):

| profile | LOOCV mean / max, before | after |
|---|---|---|
| Qwen2.5-0.5B / llama.cpp | 5.72% / 31.7% | 1.68% / 3.38% |
| SmolLM2-360M / llama.cpp | 2.70% / 10.4% | 2.98% / 6.64% |

## 4. A different bug: quantization prefill timed a 620 MB logits copy

`measure_quantization_tradeoff.py` timed prefill on the same
`logits_all=True` instance it used for perplexity. With `logits_all=True`,
every `eval()` copies n_tokens × n_vocab floats into Python (1024 × 151,936
× 4 B ≈ 620 MB for Qwen2.5). The copy inflated the committed prefill
numbers by ~45–75% over the real compute time, and under memory pressure on this 8 GB machine, single
runs swung to several seconds. Now perplexity runs on a `logits_all=True`
instance, which is freed before timing starts on a separate
`logits_all=False` instance.

| level | prefill@1024 before | after | decode TBT before | after |
|---|---|---|---|---|
| fp16 | 703.9 | **401.8** | 17.8 | 14.1 |
| q8_0 | 683.0 | **418.2** | 12.3 | 13.8 (std 7.1, noisy) |
| q4_k_m | 658.4 | **454.3** | 10.1 | 9.2 |

**This reverses a published conclusion.** The old numbers said
quantization sped up prefill "modestly (~6.5% fp16→q4_k_m)". With the copy
removed, **q4_k_m prefill is ~13% *slower* than fp16** (and 11% slower at
N=256: 102.8 vs 92.7 ms). That fits compute-bound prompt processing paying
a dequantization cost that bandwidth savings don't recover. The decode
conclusion (lower precision → faster, bandwidth-bound) still holds.
`QuantizationLevelTechnique` picks by file size, then perplexity, so its
recommendations don't change; only the prefill figure in its rationale
does.

## 5. Concurrency (Tier 6): trial-1 artifact fixed, conclusion strengthened

The trial-1 outlier reproduced (50.7 ms vs a 7.2 ms steady state at B=1)
and disappeared with a synced prefill plus an untimed 5-step decode warm-up
(added to the script). At ~1.5 s of extra wall time, it was mostly a
missing decode-path warm-up, not the prefill tail. The Tier 6 doc excluded
it only at B=1, but trial 1 was inflated at B=2 (18.5 vs ~11.7 ms) and B=4
(34.3 vs ~23.5 ms) too, and those means had included it.

Re-run with the fix: **7.69 / 13.20 / 24.84 ms** at B=1/2/4 (was
16.54 / 13.74 / 27.17).

The bigger change is on the MODELED side. `ConcurrencyThroughputTechnique`
starts from `profile.decode.predict_ms(ctx)`, which was the inflated SmolLM2
decode fit. At ctx=256 its batch=1 baseline drops from 11.5 to 8.6 ms:

| B | modeled before → after | native batched (unchanged, already safe) | native/modeled before → after |
|---|---|---|---|
| 1 | 11.50 → 8.59 | 7.95 | 0.69x → **0.92x** |
| 2 | 11.60 → 8.70 | 12.18 | 1.05x → **1.40x** |
| 4 | 11.81 → 8.91 | 20.31 | 1.72x → **2.28x** |
| 8 | 12.23 → 9.33 | 39.92 | 3.26x → **4.28x** |

The odd Tier 6 result that the model *over*-predicted at B=1 is explained:
the inflated decode fit, not the batching model, was the cause. With that
fixed, the actual Tier 6 conclusion (the model badly under-prices extra
sequences) gets **stronger**.

## 6. Exposed, not caused: the prefill cost model is blind to n_batch chunk steps

Re-running the Tier 4 closed loop with the recalibrated profiles (500 ms
SLA, 3 runs each):

| profile | margin (max LOOCV AE) | N | predicted | measured | SLA met |
|---|---|---|---|---|---|
| qwen_llamacpp | 31.7% → **3.4%** | 1100 | 483.2 | 509.0 / 493.2 / 499.4 | **no** / yes / yes |
| smollm2_360m_llamacpp_profile | 10.4% → **6.6%** | 1088 | 466.7 | 491.0 / 499.8 / 486.0 | yes (barely) |

All six runs came in above prediction (+2% to +6.6%). Neither the
executor's smaller n_ctx (1164 vs 4096) nor its different filler text
explains it (tested: both give 495–515 ms at N=1100). The cause is
`eval()`'s 512-token chunking. **Each additional `llama_decode` chunk adds
a ~26–33 ms step**, and the calibration grid samples only points that sit
exactly *on* chunk boundaries (512, 1024, 1536, 2048), never just past
them:

| N | chunks | measured | predicted | error |
|---|---|---|---|---|
| 512 | 1 | 215.8 | 213.1 | +1.3% |
| 520 | 2 | 241.8 | 216.5 | **+11.7%** |
| 1024 | 2 | 446.2 | 446.3 | −0.0% |
| 1032 | 3 | 479.2 | 450.1 | **+6.5%** |
| 1100 | 3 | 498.3 | 483.2 | +3.1% |
| 1536 | 3 | 707.5 | 706.6 | +0.1% |
| 1544 | 4 | 735.2 | 710.9 | +3.4% |

LOOCV can't detect a sawtooth that the sample grid never lands on. The old
31.7% Qwen margin came from a noisier calibration run and happened to
cover the step. The corrected 3.4% margin doesn't. **This is the next
thing to fix**, either with a per-chunk term in the cost model
(`ceil(N / n_batch)` × step) or with calibration points placed just past
each boundary so LOOCV can see the error. It is recorded here, not
patched, because a fix changes the cost model's functional form across
every llama.cpp profile.

## What this does not resolve

- One machine, one llama-cpp-python version (0.3.35). Whether `eval()`
  syncs depends on the binding version; an older version that still copied
  last-token logits inside `eval()` would have been sync-safe on decode.
- Decode TBT on this machine has occasional single-step spikes (e.g. Qwen
  ctx 1536: 15.2 ± 13.8 ms in the re-run) that tilt a 10-point linear fit.
  The Qwen decode slope (0.00167) is still partly noise.
- The SmolLM2 decode sweep covers ctx [32, 384] only. Extending it is a
  methodology change, not part of this audit.
