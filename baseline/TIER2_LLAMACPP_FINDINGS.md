# Tier 2: second model, second backend (llama.cpp / Metal) — findings

**Goal:** everything measured elsewhere in this repo is one model
(SmolVLM-Instruct-4bit) on one backend (MLX). This checks whether the
*measurement methodology* (real synchronous calls, warm-up, N trials,
honest mean/std reporting) and the *quadratic cost-model functional form*
`T_prefill(N) = gamma*N^2 + beta*N + alpha` generalize, or were artifacts of
that one (model, backend) pair.

**Setup:** Qwen2.5-0.5B-Instruct, Q4_K_M GGUF, served via `llama-cpp-python`
with full Metal GPU offload (`n_gpu_layers=-1`) on the same M3 used
throughout this study. A genuinely different model (different architecture,
size, and quantization scheme) on a genuinely different inference engine —
not a re-export of SmolVLM's LM. See `baseline/measure_llamacpp.py` for the
method (`llm.eval(tokens)` for prefill — synchronous, no MLX-style lazy-eval
trap to guard against — `llm.reset()` between trials, 1 discarded per-shape
warm-up, 8 timed trials/point) and `model_calibration/fit_llamacpp_prefill.py`
for the fit.

**This is NOT a speed comparison.** Different model size, architecture, and
quantization make raw latency numbers incomparable across the two studies.
What's comparable is whether the same functional form and methodology hold.

## Results

10 points, N in [32, 2048] tokens, all with tight variance (CV mostly <2%,
worst case 2.8%) — unlike the confounded run in `PREFILL_V3_FINDINGS.md`,
this measurement session had no other heavy process sharing the machine.

| | SmolVLM/MLX (4 points, N in [100,1560]) | Qwen2.5-0.5B/llama.cpp (10 points, N in [32,2048]) |
|---|---|---|
| R² (in-sample) | 0.9996 | 0.9996 |
| MAPE (in-sample) | 3.01% | 2.30% |
| MAPE (LOOCV) | **20.7%** | **5.72%** |
| alpha (intercept) | 244.60 ms (positive) | 5.58 ms (positive) |

Fitted: `T_prefill(N) = 4.9248e-05*N^2 + 0.37296*N + 5.58` ms.

Decode TBT (batch=1) also replicates the qualitative SmolVLM finding —
small fixed overhead + a small linear per-context-token term, i.e. decode
is memory-bandwidth-bound and roughly flat, not the binding constraint:

    Qwen2.5-0.5B/llama.cpp:  TBT(ctx) ~= 10.26 + 0.00446*ctx  ms
    SmolVLM/MLX:             TBT(ctx) ~= 18.33 + 0.00320*ctx  ms

## Interpretation

1. **The quadratic form generalizes.** A second model on a second backend
   fits the same functional form just as well in-sample (R²=0.9996) and,
   with enough well-conditioned points, validates far better under LOOCV
   (5.72% vs 20.7%). This corroborates `PREFILL_V3_FINDINGS.md`'s
   conclusion that the SmolVLM study's 20.7% LOOCV gap is a data-density /
   measurement-condition limitation (only 4 points, one run confounded by a
   shared machine on retry) — not evidence against the cost-model shape
   itself.
2. **The decode-is-roughly-flat finding also generalizes** across model
   size and backend, with model-specific constants as expected.
3. **What this does NOT prove:** portability across hardware (still one
   M3), portability to vision-language models specifically (this is a
   text-only model), or that any *technique* (quantization, batching, KV
   paging) transfers — only that the measurement/calibration methodology
   and the cost-model's functional form do. Extending Tier 2 to a second
   *technique* comparison (e.g. GGUF quant levels: Q4_K_M vs Q8_0) and to a
   second VLM would strengthen this further.

## Quantization-level tradeoff (real, measured)

A second, complementary Tier-2 check: does a *technique* — not just the
measurement methodology — transfer, and can it be measured honestly rather
than asserted? The original study's REVIEW_FINDINGS.md flagged that every
quantization "result" in that study was fabricated (a manufactured 2x FP8
speedup with no such GPU path on Apple Silicon; W4A8 applied on top of a
baseline that was already 4-bit, double-counting the same compression
twice). This repeats the comparison for real: Qwen2.5-0.5B-Instruct, three
real GGUF quantization levels (fp16, Q8_0, Q4_K_M), same backend
(llama.cpp/Metal), each measured for file size, prefill speed, decode
speed, and a real perplexity proxy (manual log-softmax over raw
per-position logits — see `baseline/measure_quantization_tradeoff.py` for
why the high-level `create_completion(echo=True, logprobs=1)` API was
tried and rejected: it returned 569 "token" logprobs for a 23-token
sentence, an unexplained discrepancy not worth trusting).

| level | size (MiB) | perplexity | prefill@1024 (ms) | decode TBT (ms) |
|---|---|---|---|---|
| fp16 | 1207.8 | 11.93 | 703.9 | 17.8 |
| q8_0 | 644.4 | 11.89 | 683.0 | 12.3 |
| q4_k_m | 468.6 | 12.59 | 658.4 | 10.1 |

Real mechanism, not a fabricated one: prefill (compute-bound) speeds up
only modestly (~6.5%, fp16→q4_k_m), while decode (memory-bandwidth-bound)
speeds up substantially (~43%) because lower-precision weights move less
data per token — this is why the original study's claim of a uniform ~2x
GEMM speedup from quantization was never physically plausible on this
hardware. Quality cost, measured rather than asserted: perplexity rises
only ~5.5% relative (11.93→12.59) on the fixed reference passage from fp16
to Q4_K_M, on this one model and one passage (not a task-accuracy
benchmark — same caveat as the rest of this repo's "quality" numbers).

This data backs `core/techniques/quantization_level.py`'s
`QuantizationLevelTechnique` (see `core/README.md`).

## Reproduction

```bash
# Model (curl, not huggingface_hub's hf_hub_download -- that downloader has
# no read-timeout and hung indefinitely on a dead connection during this
# project's own run):
curl -L -C - --retry 20 --retry-delay 3 --connect-timeout 15 \
  --speed-time 15 --speed-limit 500 \
  -o ~/.cache/inference_optimizer_models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

scope/venv_phase0/bin/python baseline/measure_llamacpp.py --trials 8
scope/venv_phase0/bin/python model_calibration/fit_llamacpp_prefill.py

# Quantization tradeoff (downloads fp16 + q8_0; q4_k_m reused from above):
curl -L -C - --retry 20 --retry-delay 3 --connect-timeout 15 \
  --speed-time 15 --speed-limit 500 \
  -o ~/.cache/inference_optimizer_models/qwen2.5-0.5b-instruct-fp16.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf"
curl -L -C - --retry 20 --retry-delay 3 --connect-timeout 15 \
  --speed-time 15 --speed-limit 500 \
  -o ~/.cache/inference_optimizer_models/qwen2.5-0.5b-instruct-q8_0.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"

scope/venv_phase0/bin/python baseline/measure_quantization_tradeoff.py --trials 8
```
