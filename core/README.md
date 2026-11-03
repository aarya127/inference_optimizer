# core/ — the Technique abstraction (Tier 3)

Everything in `simulation/`, `amio_constants.py`, and `model_calibration/`
is specific to one model on one backend (SmolVLM-Instruct-4bit on MLX).
`core/` is the first step toward the stated goal of a universal tool with
different techniques for different models: one interface
(`core/technique.py`'s `Technique`), implemented once per technique, that
runs against a `ModelHardwareProfile` (`core/profile.py`) instead of
hardcoded constants.

## Why this design

A `ModelHardwareProfile` is a portable record of what was *actually*
calibrated for one (model, backend, hardware) combination — every field
traces back to a real measurement already in this repo, never a new
invented number:

- `smolvlm_mlx` — SmolVLM-Instruct-4bit / MLX / Apple M3, from
  `amio_constants.py` (`baseline/results_v2.json`, 4 points, LOOCV 20.7%).
  Ships pre-quantized; no multi-level quantization comparison exists for
  it, so `quant_levels` is `None` — not a guess.
- `qwen_llamacpp` — Qwen2.5-0.5B-Instruct / llama.cpp (Metal) / Apple M3,
  from `model_calibration/llamacpp_prefill_fit.json` (10 points, LOOCV
  5.7%) and `baseline/results_quantization_tradeoff.json` (real fp16 /
  Q8_0 / Q4_K_M measurements — see `baseline/TIER2_LLAMACPP_FINDINGS.md`).
- `smollm2_360m_llamacpp_profile` — SmolLM2-360M-Instruct / llama.cpp
  (Metal) / Apple M3, auto-calibrated end-to-end by `core/calibrate.py`
  (10 points, LOOCV 2.76%) — see `TIER5_AUTO_ONBOARDING_FINDINGS.md`. No
  model-specific loader function exists for this one;
  `load_profile_from_calibration_json` is generic.
- `qwen25_05b_mlx_profile` — the *same* Qwen2.5-0.5B-Instruct model as
  `qwen_llamacpp`, this time on MLX, auto-calibrated by
  `core/calibrate_mlx.py` (LOOCV 2.05% — tighter than the llama.cpp
  version of the same model). A genuine same-model, cross-backend
  comparison; `kv_bytes_per_token` comes out identical (12,288) on both
  backends, as it should for the same real architecture.
- `smolvlm_256m_mlx_profile` — SmolVLM-256M-Instruct, same Idefics3 family
  as `smolvlm_mlx` but a much smaller LM backbone, auto-calibrated by
  `core/calibrate_mlx_vlm.py` (LOOCV mean 7.53%/max 37.93%, validating
  better than the original hand-built profile). Proves the vision side of
  Tier 5 generalizes too — `MLXExecutor` (originally written only for the
  2B `smolvlm_mlx`) ran a real closed-loop check against this model
  unmodified, 1.3% prediction error.

A `Technique` declares `applies_to(profile)` and `recommend(profile, ...)`.
Two are implemented, chosen specifically because they demonstrate
*different* applicability on the two real profiles above, not because
they're the only two techniques worth having:

- `PrefillBudgetTechnique` — applies to both profiles (both have a
  calibrated prefill cost model); the SLA-safe token count it computes is
  correctly different for each (and flags when a request falls outside
  the calibrated domain — see `core/demo.py`'s N=3619 extrapolation
  warning for Qwen at a 2000ms budget).
- `QuantizationLevelTechnique` — applies to `qwen_llamacpp` (three
  measured levels) but *not* to `smolvlm_mlx` (no such data exists) —
  the interface reports "not applicable" rather than guessing.
- `KVCacheBudgetTechnique` — max context length per sequence under a
  memory budget, from each profile's real `kv_bytes_per_token` (derived
  from actual checkpoint architecture: layers, KV heads, head dim, dtype).
  Applies to both profiles, and the real 16x difference between them
  (SmolVLM: 196,608 B/token, full MHA, 32 KV heads; Qwen2.5-0.5B: 12,288
  B/token, GQA, 2 KV heads) produces a genuinely different answer — e.g.
  10,923 vs 174,763 max context tokens under the same 2 GiB budget.
- `ConcurrencyThroughputTechnique` — max concurrent sequences under a
  decode (TBT) SLA. Explicitly labeled MODELED, not measured: every decode
  measurement in this repo is batch=1 only, so this extends the measured
  batch=1 `DecodeCostModel` with the same per-extra-sequence KV-read term
  `amio_constants.py` already documents for SmolVLM (~3.0ms/extra sequence
  at 1548 ctx), generalized to any profile's real `kv_bytes_per_token` and
  `bandwidth_gbps` — applied to both profiles with the same honest label,
  not asserted as a measurement for either.

## Usage

```bash
scope/venv_phase0/bin/python -m core.demo
scope/venv_phase0/bin/python -m pytest tests/test_core.py -v
```

## Tier 4: closing the loop (recommend → execute → verify)

`core/executor.py` + `core/executors/` add real execution: given a
Technique's recommendation, actually run it through the real backend
(`LlamaCppExecutor` loads the real recommended GGUF quant level and runs a
real `llm.eval()`; `MLXExecutor` runs a real image through the real vision
encoder + LM prefill) and compare MEASURED latency against the cost
model's PREDICTED value — not just against the historical LOOCV score.

```bash
scope/venv_phase0/bin/python -m core.optimize --profile qwen_llamacpp --sla-ms 500
scope/venv_phase0/bin/python -m core.optimize --profile smolvlm_mlx --sla-ms 500
```

This caught two things LOOCV alone would not have: (1) a recommendation
sitting exactly at the SLA boundary is fragile to real-world variance —
Qwen's recommendation predicted 500.1ms, measured 502.8ms, and technically
missed the SLA despite only 0.5% prediction error; (2) SmolVLM/MLX's cost
model, despite R²=0.9996 in-sample, is off by 41-72% against a fresh real
execution — reproducibly, and in the same direction as
`baseline/PREFILL_V3_FINDINGS.md`'s earlier session-dependent finding. See
`TIER4_CLOSED_LOOP_FINDINGS.md` for the full writeup. Deliberately not
"fixed" — this is reported as a finding about this profile's real-world
reliability, not smoothed over.

Executors are standalone (not wrapped in `tests/test_core.py`), matching
this repo's convention of keeping real, slow, model-loading measurements as
runnable scripts rather than part of the fast pytest suite.

`PrefillBudgetTechnique.recommend()` defaults `safety_margin_pct` to the
profile's own **max LOOCV AE** (the worst single held-out point, not the
mean) — added after the Qwen finding above (a boundary-exact
recommendation missing its SLA despite 0.5% prediction error), and
tightened further after a mean-based margin was tried and found
insufficient (see TIER5 below). Re-verified across all four profiles: every
one that returns a token budget now meets its SLA with real headroom, not
a near-miss (e.g. Qwen/llama.cpp: 356.7ms measured vs a 500ms SLA, 31.7%
margin). SmolVLM/MLX is the interesting case — its margin (59.9%) is now
large enough that the technique correctly reports **no feasible token
count** for a 500ms SLA, matching this project's own long-standing finding
that the SLA is infeasible on this hardware. Full detail in
`TIER4_CLOSED_LOOP_FINDINGS.md`.

## Tier 5: automated onboarding (no hand-written per-model code)

`core/calibrate.py` generalizes the measurement methodology behind
`qwen_llamacpp` into a reusable pipeline for any llama.cpp/GGUF model:
download, real prefill/decode calibration, weighted quadratic fit with
honest LOOCV, and real `kv_bytes_per_token` pulled from the checkpoint's
own `config.json` — emitted as one self-describing JSON.
`load_profile_from_calibration_json` turns that into a working profile
with no new code. Proven by onboarding a third model, SmolLM2-360M-Instruct,
end-to-end (LOOCV 2.76% — the best of the three profiles). The
`LlamaCppExecutor` was refactored alongside this so it resolves which GGUF
file to load from the profile itself (`resolve_gguf_filename`), not a
hardcoded per-model dict — necessary for it to work against a profile it
has never seen before. See `TIER5_AUTO_ONBOARDING_FINDINGS.md`.

```bash
scope/venv_phase0/bin/python -m core.calibrate \
  --model-id <hf-model-id> --gguf-repo <repo> --gguf-filename <file> \
  --base-config-repo <hf-model-id> --out model_calibration/<name>_llamacpp_profile.json
scope/venv_phase0/bin/python -m core.optimize --profile <name>_llamacpp_profile --sla-ms 500
```

`core/calibrate_mlx.py` extends the same automated approach to MLX-LM text
models, reusing `core/calibrate.py`'s fit/architecture functions unchanged
(they were already backend-agnostic). Proven by calibrating the *same*
Qwen2.5-0.5B-Instruct model already profiled on llama.cpp — a genuine
same-model, cross-backend comparison (LOOCV 2.05% on MLX vs 5.72% on
llama.cpp; identical `kv_bytes_per_token`, as expected for the same
architecture). `MLXLMExecutor` is the generic text-only counterpart to the
vision-specific `MLXExecutor`, resolving everything from `profile.model_id`.

This also surfaced two bugs, both now fixed rather than left as caveats:
(1) the margin-adjusted recommendation for `qwen25_05b_mlx_profile` missed
its SLA on two separate real runs — fixed by switching the default margin
basis from mean to max LOOCV AE (see above); (2) its decode fit had an
unphysical negative slope from only 5 narrow-range calibration points —
`core/calibrate.py`'s `fit_decode_linear` now clamps a negative slope to 0
(flat model) and keeps the raw, unclamped fit alongside it for
transparency. Both fixes and their before/after numbers are recorded in
`TIER5_AUTO_ONBOARDING_FINDINGS.md` and `TIER4_CLOSED_LOOP_FINDINGS.md`.

`core/calibrate_mlx_vlm.py` extends the same approach to vision-language
models, reusing `fetch_architecture` after generalizing it to read the
nested `text_config` VLM checkpoints use for their LM architecture (one
shared function now confirmed against both schema shapes, not a
VLM-specific duplicate). Calibrated SmolVLM-256M-Instruct — same Idefics3
family as `smolvlm_mlx` but a much smaller LM backbone — and the existing
`MLXExecutor` (written only for the original 2B model) ran a real
closed-loop check against it with **zero modification**, 1.3% prediction
error. Confirms Tier 5's automation covers vision models too, not just
text. Full detail in `TIER5_AUTO_ONBOARDING_FINDINGS.md`.

## Tier 6: real concurrency measurement exposes a bad model

`ConcurrencyThroughputTechnique` had never been checked against a real
multi-sequence run — it was a MODELED extension of batch=1 decode data
from day one. `baseline/measure_llamacpp_concurrency.py` measures real
concurrent-decode throughput (B independent llama.cpp contexts, Python
threads, SmolLM2-360M) and compares it against the technique's prediction
for the same B and context length. Result: the technique is **substantially
too optimistic** — it predicts near-flat TBT scaling (11.5ms → 11.8ms from
B=1 to B=4, since it only models KV-bandwidth contention) while the real
measurement showed close to linear degradation (~27ms at B=4, 2.3x worse
than predicted), most likely from GIL or GPU-command-queue serialization
the model doesn't account for. Not recalibrated to this one data point —
the technique's docstring and `core/TIER6_REAL_CONCURRENCY_FINDINGS.md`
now say plainly to treat its output as an optimistic lower bound, not an
estimate, until validated for a given real deployment.

## What this is not (yet)

This is a proof of the abstraction on four profiles, four techniques, and
three executors — not a general driver that loads arbitrary HuggingFace
models on arbitrary backends, and not a production autotuner (still no
automatic recalibration when a closed-loop check disagrees with the cost
model — the margin is a static, LOOCV-derived buffer, not a response to
live results). Extending it further (automatic margin tuning from
closed-loop history, a real CUDA backend/profile, real batching/scheduling
techniques measured against a real engine) is future work; see the main
README's roadmap.
