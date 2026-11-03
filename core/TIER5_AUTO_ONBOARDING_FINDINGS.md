# Automated onboarding: a third model, zero new per-model code

Gap identified after Tier 4: onboarding `qwen_llamacpp` required a
hand-written measurement script (`baseline/measure_llamacpp.py`) and a
hand-written loader function with that model's architecture constants
inlined (`load_qwen_llamacpp_profile()`). That's the opposite of
"universal" -- every new model is bespoke.

`core/calibrate.py` fixes this for llama.cpp-backed models: given a GGUF
repo/file and the base checkpoint's HF repo (for its real `config.json`),
it downloads the model, runs the same real prefill/decode measurement
methodology as `baseline/measure_llamacpp.py` (warm-up, N trials, weighted
quadratic fit, honest LOOCV), derives `kv_bytes_per_token` from the
checkpoint's actual architecture (not assumed), and writes one
self-describing JSON. `core/profile.py`'s `load_profile_from_calibration_json`
turns that JSON into a working `ModelHardwareProfile` with no code specific
to that model.

## Proof: onboarding SmolLM2-360M-Instruct

```bash
scope/venv_phase0/bin/python -m core.calibrate \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --gguf-repo bartowski/SmolLM2-360M-Instruct-GGUF \
  --gguf-filename SmolLM2-360M-Instruct-Q4_K_M.gguf \
  --base-config-repo HuggingFaceTB/SmolLM2-360M-Instruct \
  --trials 8 \
  --out model_calibration/smollm2_360m_llamacpp_profile.json
```

Real architecture, pulled from the checkpoint's own `config.json` (not
guessed): hidden_size=960, 32 layers, 15 attention heads / 5 KV heads
(GQA), head_dim=64 -> **40,960 B/token** KV cache -- between Qwen2.5-0.5B's
12,288 (fewer KV heads, fewer layers) and SmolVLM's 196,608 (full MHA, no
GQA).

Calibration result (10 points, N in [32, 2048]):

| | SmolVLM/MLX | Qwen2.5-0.5B/llama.cpp | SmolLM2-360M/llama.cpp |
|---|---|---|---|
| points | 4 | 10 | 10 |
| R² (in-sample) | 0.9996 | 0.9996 | 0.99995 |
| MAPE (in-sample) | 3.01% | 2.30% | 1.61% |
| MAPE (LOOCV) | 20.7% | 5.72% | **2.76%** |

The third independently-calibrated model has the *best* LOOCV of the
three, reinforcing Tier 1/2's conclusion: data density and measurement
discipline (10 clean points vs. 4) is what drives validation quality here,
not anything specific to a particular model or backend.

`core/demo.py` and `core/optimize.py --profile smollm2_360m_llamacpp_profile`
both work against this profile with **zero new code** -- all four existing
Techniques correctly apply (or don't -- `QuantizationLevelTechnique`
reports "not applicable" since only one quant level was calibrated,
exactly as designed) and produce profile-specific, real numbers (e.g. KV
budget: 52,429 max context tokens/sequence at a 2GiB budget, vs Qwen's
174,763 and SmolVLM's 10,923 -- all three different, all from the same
`KVCacheBudgetTechnique` code).

**Closed-loop validation** (`core/executors/llamacpp_executor.py`, refactored
to resolve GGUF filenames from the profile itself rather than a
model-specific hardcoded dict -- see that file's `resolve_gguf_filename`):
predicted 486.4ms, measured 495.8ms at the margin-adjusted N=1132 token
budget under a 500ms SLA -- **1.9% error**, consistent with the 2.76%
LOOCV estimate and the SLA was met for real.

## Second proof: the same model, two backends, automated both times

`core/calibrate_mlx.py` generalizes the same pipeline to MLX-LM text
models, reusing `core/calibrate.py`'s `fetch_architecture` /
`fit_quadratic_with_loocv` / `fit_decode_linear` unchanged (those only
operate on plain records and a config.json URL — already backend-agnostic).
Only the real-call measurement differs (`model(input_ids, cache=...)` via
`mlx_lm`, vs `llm.eval(tokens)` via `llama-cpp-python`).

Calibrated `mlx-community/Qwen2.5-0.5B-Instruct-4bit` — the **same base
architecture** already on llama.cpp as `qwen_llamacpp` (hidden_size=896, 24
layers, 2 KV heads/GQA; `kv_bytes_per_token` comes out identical: 12,288 on
both backends, since it's the same real config, confirming the derivation
is backend-independent as it should be):

| | Qwen2.5-0.5B / llama.cpp | Qwen2.5-0.5B / MLX |
|---|---|---|
| LOOCV MAPE | 5.72% | **2.05%** |
| R² (in-sample) | 0.9996 | 0.9995 |

Same model, two backends, two independently automated calibration runs —
both validate well, with MLX's slightly tighter here. This is a genuine
same-model cross-backend comparison, not just two unrelated models that
happen to both be small.

**Closed-loop result, and an honest limit of the margin approach**: this
profile's margin-adjusted recommendation (2.0% margin from its own 2.05%
LOOCV) missed the 500ms SLA on two separate real runs (502.4ms and
501.6ms, 2.4-2.6% real error) — reproducibly, not as an outlier. See
`TIER4_CLOSED_LOOP_FINDINGS.md`'s second follow-up: a margin sized to the
*mean* LOOCV error will be breached about as often as real error exceeds
that mean, by construction. This is the clearest demonstration yet that
the current margin heuristic needs a more conservative basis (e.g. max
LOOCV error, not mean) — not a result specific to this profile or backend.

**A real flaw this run also surfaced**: the decode (TBT) fit for this
profile has a small *negative* slope (`TBT(ctx) ~= 9.661 - 0.011*ctx` ms),
which predicts negative latency by ctx≈2048 — physically impossible. This
comes from fitting a line to only 5 narrow-range points (ctx 32-384);
within that range the effect is tiny and plausibly just noise, but the
fit should never have been allowed to extrapolate to a sign flip.
`core/calibrate_mlx.py` and `core/calibrate.py` do not currently guard
against this (unlike `model_calibration/calibrate_cost_model.py`'s
existing alpha-clamping for the prefill fit) — a real gap to fix before
trusting any profile's decode numbers past its narrow calibrated range.

## What this still doesn't do

- Only handles llama.cpp-backed models. An equivalent generic pipeline for
  MLX (or any other backend) would need its own calibration script — MLX's
  real methodology is more bespoke (vision + real image + chat template
  for SmolVLM; a hypothetical new MLX-LM text model would differ again).
- Still requires knowing the GGUF repo/file and the base checkpoint's repo
  up front, as command-line arguments. Not "point this at any HF model
  name and it figures out the rest" — closer to "given these three
  identifiers, do the rest automatically."
- Only prefill + batch=1 decode are calibrated automatically. Quantization
  tradeoffs, vision-specific techniques, and anything else from
  `core/techniques/` still needs its own measurement path per profile.
