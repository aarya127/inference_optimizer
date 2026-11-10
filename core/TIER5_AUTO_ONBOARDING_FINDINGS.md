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

> **Recalibrated after the sync-bug audit ([SYNC_AUDIT_FINDINGS.md](SYNC_AUDIT_FINDINGS.md)).**
> `calibrate.py`'s decode sweep let the untimed prefill leak into the first
> timed decode step, so the SmolLM2 decode fit came out at 7.93 +
> 0.0139·ctx ms, ~3x too steep. The re-run gives a flat 8.59 ms (raw slope
> −0.005, clamped). The prefill side was not materially affected. The
> re-run's fit is LOOCV 2.98% mean / 6.64% max (R² 0.9986), and Qwen/llama.cpp's
> re-measured LOOCV is 1.68% (was 5.72%). The table above is kept as the
> original run's record. The closed-loop result below was **not**
> reproduced with the recalibrated profile: N=1088 passed its 500ms SLA
> on 3/3 runs, but by as little as 0.2ms. See SYNC_AUDIT_FINDINGS.md §6
> for why (n_batch chunk steps).

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
| LOOCV MAPE | 5.72% (1.68% after the sync-audit re-run) | **2.05%** |
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

## Both findings above are now fixed

See `TIER4_CLOSED_LOOP_FINDINGS.md`'s "Resolution" section for full
before/after numbers. Summary:

- **Margin basis**: `core/calibrate.py`'s `fit_quadratic_with_loocv` now
  also reports `loocv_max_ape_pct` (worst single held-out point), and
  `PrefillBudgetTechnique` defaults to it instead of the mean. Recomputed
  for this profile (a fresh calibration run, so the exact numbers differ
  slightly from the ones above by natural run-to-run variance): mean LOOCV
  2.68%, max LOOCV AE **7.74%**. Re-running the closed loop with the new
  margin: predicted 461.4ms, measured 468.7ms and 472.8ms across two runs
  — comfortably under the 500ms SLA both times, not a near-miss.
- **Decode fit**: `fit_decode_linear` now clamps a negative slope to 0 (a
  flat model) and retains the raw, unclamped fit in the output for
  transparency (`raw_fit_per_ctx_token_ms`, `clamped_negative_slope`). The
  fresh calibration run above happened to produce a (tiny) positive slope
  naturally, so the clamp didn't fire this time — it remains in place as a
  guard for whenever a future run reproduces the negative-slope artifact,
  verified directly against a reproduction of the original scenario in
  `tests/test_core.py::test_fit_decode_linear_clamps_negative_slope`.

## Third proof: the vision side generalizes too

`core/calibrate_mlx_vlm.py` extends the automated approach to
vision-language models, reusing `fetch_architecture` after generalizing it
to handle the nested `text_config` vision-language configs use (SmolVLM's
real LM architecture lives under `config["text_config"]`, not at the top
level like a plain text model) — one shared function, no duplication, now
confirmed against both schema shapes. The real-call measurement reuses the
exact methodology already proven in `core/executors/mlx_executor.py` and
the Tier-1 densification script: crops fixed at 1 (~70-100 real image
tokens via a synthetic seeded image), text length varied to reach each
target N.

Calibrated `mlx-community/SmolVLM-256M-Instruct-4bit` — same Idefics3
family as `smolvlm_mlx`, but a genuinely different, much smaller LM
backbone (hidden_size=576, 30 layers, 3 KV heads/GQA vs the original's
larger SmolLM2-1.7B-class one; `kv_bytes_per_token` comes out to 23,040,
between SmolLM2-360M's 40,960 and Qwen's 12,288):

| | smolvlm_mlx (original, hand-built) | smolvlm_256m_mlx_profile (auto) |
|---|---|---|
| points | 4 | 10 (two landed on the same floor N=80 — see below) |
| LOOCV mean / max | 20.7% / 59.9% | 7.53% / 37.93% |
| R² (in-sample) | 0.9996 | 0.9995 |

The smaller, auto-calibrated model again validates better than the
original hand-built one — consistent with every prior comparison in this
file: more points, cleaner conditions, better validation, independent of
model or backend.

**Closed-loop result**: `MLXExecutor` — written originally for the ~2B
`smolvlm_mlx` and never modified for this — correctly loaded and ran a
real prefill for this completely different model via `profile.model_id`.
Predicted 313.7ms, measured 317.8ms (**1.3% error**), comfortably under
the 500ms SLA with its 37.9% margin.

**A real quirk, disclosed not hidden**: this model's tokenization floor at
crops=1 is higher than the smallest two target token counts (32, 64) —
both searches landed on the same actual N=80, since the fixed image +
chat-template overhead alone exceeds those targets. The calibration still
ran (duplicate x-values don't break the quadratic fit), but this is
effectively a 9-point sweep with one point measured twice, not a full
10-point spread. `core/calibrate_mlx_vlm.py` does not currently detect or
warn about this.

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
