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

## What this is not (yet)

This is a proof of the abstraction on two profiles and two techniques —
not a general driver that loads arbitrary HuggingFace/GGUF models and
auto-calibrates them. Extending it further (more techniques, an
auto-calibration pipeline instead of hand-run scripts, more profiles)
is future work; see the main README's roadmap.
