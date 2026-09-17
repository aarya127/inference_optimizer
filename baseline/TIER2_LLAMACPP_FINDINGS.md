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
```
