# Prefill calibration densification (v3) — inconclusive, NOT adopted

**Attempt:** densify the 4-point LM-prefill calibration (fit to
`baseline/results_v2.json`, LOOCV MAPE 20.7%) by varying text-prompt length
instead of crop count, reusing the identical trusted real code path
(`get_input_embeddings` → `make_prompt_cache` → `language_model` prefill,
`mx.eval` on actual outputs — see `baseline/measure_prefill_text_calibration.py`).
This produced 13 new points spanning N ≈ 100–2381, run with the same
`mlx_version` (0.31.0) on the same machine/platform as the original campaign.

**Result:** combining the new points with the original 4 and refitting
(`model_calibration/refit_prefill_v3.py`) makes the fit *worse*, not better:

| | original 4-point | combined 17-point (median-weighted) |
|---|---|---|
| R² (in-sample) | 0.9996 | 0.6752 |
| MAPE (in-sample) | 3.01% | 14.93% |
| MAPE (LOOCV) | 20.7% | 15.68% |

LOOCV improves somewhat (more points means holding one out costs less
information), but the collapse in in-sample R² means the quadratic no
longer describes the data well at all — a worse model, not a better-
validated one.

**Root cause, not a modeling problem:** the new run's smallest point
(N=101, "repeats=0") should be almost identical to the original campaign's
`crops_1` point (N=100) — same image, same crop setting, same chat-template
text, same code path. Instead they disagree by ~1.8×:

- `crops_1` (original, 2026-07-28): trials `[388.3, 357.3, 379.4, 311.8, 345.3]` ms
- `repeats=0` (this run): trials `[195.3, 194.4, 194.5, 194.6, 194.7, 196.3, 194.6, 194.5]` ms

Both clusters are internally tight-ish; they are two different stable
regimes, not one noisy distribution. `mlx_version` and platform are
identical between the two runs, which rules out a software-version
confound. The same run also shows severe, non-monotonic variance at large N
(up to 67% coefficient of variation at N≈1981–2381 — see raw trials in
`baseline/prefill_text_calibration.json`), which was never observed in the
original campaign.

The most likely explanation: this measurement was run from inside a live
coding-agent session sharing the machine (this repository's ongoing Claude
Code session), not an idle, dedicated benchmarking environment. Concurrent
CPU/GPU/memory load from that session plausibly explains both the ~1.8×
baseline shift and the large-N instability.

**Disposition:** these coefficients are **not** adopted into
`amio_constants.py` / README / FINAL_REPORT. The original 4-point fit
(`PREFILL_GAMMA/BETA/ALPHA`, LOOCV 20.7%) remains the source of truth. The
new script and data are kept (not deleted) because the *methodology*
(varying text length to decouple N from crop count) is sound and reusable —
only this particular run's conditions were not.

**To actually resolve the 20.7% LOOCV gap:** re-run
`baseline/measure_prefill_text_calibration.py` on an idle machine with no
concurrent foreground load — not from inside an active agent/IDE session —
then re-run `model_calibration/refit_prefill_v3.py` and compare against the
numbers in this file before deciding whether to adopt.
