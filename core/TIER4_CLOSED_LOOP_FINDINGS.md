# Tier 4: closing the loop (recommend → execute → verify)

`core/optimize.py` takes a Technique's recommendation and actually runs it
through a real `Executor` (`core/executors/`), comparing the cost model's
PREDICTED latency against a fresh, live MEASURED one — not just against
the historical LOOCV score the model was calibrated with. Run both ways:

```bash
scope/venv_phase0/bin/python -m core.optimize --profile qwen_llamacpp --sla-ms 500
scope/venv_phase0/bin/python -m core.optimize --profile smolvlm_mlx --sla-ms 500
```

## Qwen2.5-0.5B / llama.cpp: validated tightly

`PrefillBudgetTechnique` recommends N=1151 tokens under a 500ms SLA;
`LlamaCppExecutor` actually loads `q4_k_m` and runs it for real:

- predicted: 500.1 ms
- **measured: 502.8 ms (0.5% error)**
- SLA met on this real run: **False** (missed by 2.8ms)

The 0.5% error is excellent — consistent with this profile's LOOCV MAPE of
5.7% (baseline/TIER2_LLAMACPP_FINDINGS.md). But because the recommended N
is, by construction, the exact token count that hits the SLA boundary, a
tiny real-world variance (0.5%) was enough to miss it. **Lesson: a
technique that recommends the exact predicted boundary is fragile in
practice — a production version should target a safety margin below the
SLA, not the boundary itself.** This is a genuine finding from actually
executing the recommendation, not something LOOCV on historical data alone
would surface.

## SmolVLM / MLX: the cost model does not survive a fresh execution

`PrefillBudgetTechnique` recommends N=180 tokens under the same 500ms SLA;
`MLXExecutor` actually runs a real image + real LM prefill (same code path
as `baseline/measure_v2.py`):

| N (actual) | predicted (ms) | measured (ms) | error |
|---|---|---|---|
| 181 | 501.5 | 291.4–292.5 (2 repeat runs) | **72%** |
| 481 | 1096.0 | 778.9 | **41%** |

Both errors are large and in the *same direction* (fresh measurement
consistently faster than predicted) and reproducible across repeated
invocations (291.4 and 292.5 ms on two separate runs at N=181) — this is
not per-call noise. It is also the same direction and rough magnitude as
`baseline/PREFILL_V3_FINDINGS.md`'s finding: a densification run of this
same crops=1 + text-padding methodology, run from inside this same live
agent session, measured ~1.8x faster than the original dedicated campaign
at nearly identical N. That finding was inconclusive on its own; this
closed-loop result, gathered independently and later, reproduces the same
direction and a comparable magnitude, which raises it from "maybe a fluke"
to "a real, session/machine-state-dependent effect on this profile" —
plausibly the GPU's power/clock state differing between a fresh, idle
session and this one, which has run hours of sustained MLX/Metal
compute. It is emphatically **not** proof that the original 4-point
campaign was wrong; both campaigns may be internally correct records of
different real machine states.

**This is the headline Tier-4 lesson:** SmolVLM/MLX's cost model has
excellent in-sample fit (R²=0.9996) and a known-mediocre LOOCV (20.7%,
`baseline/PREFILL_V3_FINDINGS.md`) — but closing the loop shows the real
generalization error, across sessions/machine-states, is *worse than
LOOCV predicted*, not better. Qwen/llama.cpp's cost model, by contrast,
validated within rounding error of its own LOOCV estimate. The difference
is exactly what Tier 1/2 already pointed at: 4 points from one session
vs. 10 points from one clean session — and this result is further
evidence that data density and measurement-condition stability matter
more than which functional form is used.

## Practical implication for the "universal tool" vision

A recommendation engine that only trusts its own calibration (LOOCV) can
be badly wrong in production. The closed loop here — predict, execute for
real, compare — is cheap to run and caught a problem LOOCV alone did not
expose. Any real deployment of `core/` should run this kind of live check
periodically, not assume a one-time calibration stays valid indefinitely.

## What this does not (yet) do

- No safety-margin logic in `PrefillBudgetTechnique` itself (the Qwen
  finding above suggests it should have one).
- No automatic recalibration when a closed-loop check disagrees with the
  cost model by more than some threshold.
- `MLXExecutor`'s text-length search lands close to, not exactly at, the
  requested N (tokenizer granularity) -- reported honestly in
  `ExecutionResult.note`, not silently rounded.
