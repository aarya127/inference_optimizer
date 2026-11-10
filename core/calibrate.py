"""Automated calibration pipeline for llama.cpp-backed models.

This is the direct fix for the gap identified after Tier 4: onboarding a
new (model, backend) profile previously meant hand-writing a new
measurement script per model (baseline/measure_llamacpp.py was written
specifically for Qwen2.5-0.5B) and hand-wiring its results into
core/profile.py (load_qwen_llamacpp_profile() has Qwen's architecture
constants inlined). This module does the same real measurement --
download, real llm.eval() calls, warm-up, N trials, weighted quadratic fit,
honest LOOCV -- for ANY GGUF model + its real HF config.json, and emits one
self-describing JSON that core/profile.py's load_profile_from_calibration_json
can turn into a ModelHardwareProfile without any model-specific code.

It does NOT remove load_qwen_llamacpp_profile() or load_smolvlm_mlx_profile()
-- those stay as hand-built profiles from this project's own measurement
history. This is the path for the NEXT model, proven by onboarding a third
one (see core/onboard_smollm2.py) with no new per-model loader function.

Usage:
    scope/venv_phase0/bin/python -m core.calibrate \
        --gguf-repo bartowski/SmolLM2-360M-Instruct-GGUF \
        --gguf-filename SmolLM2-360M-Instruct-Q4_K_M.gguf \
        --base-config-repo HuggingFaceTB/SmolLM2-360M-Instruct \
        --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
        --out model_calibration/smollm2_360m_llamacpp_profile.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

MODEL_CACHE_DIR = Path.home() / ".cache" / "inference_optimizer_models"
DEFAULT_TOKEN_COUNTS = [32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
_FILLER = ("The quick brown fox jumps over the lazy dog near the river bank "
           "while the sun sets slowly behind the distant mountains. ")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def fetch_architecture(base_config_repo: str) -> dict:
    """Pull the REAL config.json for the base checkpoint and derive KV
    bytes/token generically -- no per-model hardcoding, no guessing.

    Handles both GQA (num_key_value_heads present) and plain MHA (falls
    back to num_attention_heads), and both plain text-LM configs (fields
    at the top level) and vision-language configs (the LM architecture
    nested under "text_config", e.g. Idefics3/SmolVLM) -- the nested case
    is what core/calibrate_mlx_vlm.py needs; this function is shared
    rather than duplicated because the KV-bytes-per-token derivation
    itself doesn't care whether a vision tower sits in front of the LM.

    kv_dtype_bytes=2 (F16) is the default KV cache dtype for both
    llama.cpp (core/calibrate.py) and MLX (core/calibrate_mlx.py /
    calibrate_mlx_vlm.py) unless explicitly overridden, which none of the
    measurement scripts in this repo do.
    """
    url = f"https://huggingface.co/{base_config_repo}/raw/main/config.json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        cfg = json.loads(resp.read())

    lm_cfg = cfg.get("text_config", cfg)  # vision-language configs nest the LM here

    hidden_size = lm_cfg["hidden_size"]
    num_layers = lm_cfg["num_hidden_layers"]
    num_attention_heads = lm_cfg["num_attention_heads"]
    num_kv_heads = lm_cfg.get("num_key_value_heads", num_attention_heads)
    head_dim = lm_cfg.get("head_dim", hidden_size // num_attention_heads)
    kv_dtype_bytes = 2  # F16, the default KV cache dtype on both backends here

    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes

    return {
        "source_config_url": url,
        "nested_under_text_config": "text_config" in cfg,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "kv_dtype_bytes": kv_dtype_bytes,
        "kv_bytes_per_token": kv_bytes_per_token,
        "uses_gqa": num_kv_heads != num_attention_heads,
    }


def download_gguf(gguf_repo: str, gguf_filename: str) -> Path:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_CACHE_DIR / gguf_filename
    if dest.exists():
        print(f"[download] {dest} already present, skipping")
        return dest

    url = f"https://huggingface.co/{gguf_repo}/resolve/main/{gguf_filename}"
    print(f"[download] {url} -> {dest}")
    # Same robust curl invocation developed during Tier 2: -C - to resume
    # instead of restarting on stall, --speed-time/--speed-limit to detect
    # and recover from a dead connection (hf_hub_download hung indefinitely
    # on exactly this during this project's own run).
    subprocess.run(
        ["curl", "-L", "-C", "-", "--retry", "20", "--retry-delay", "3",
         "--connect-timeout", "15", "--speed-time", "15", "--speed-limit", "500",
         "-o", str(dest), "--progress-bar", url],
        check=True,
    )
    return dest


def _build_tokens(llm, n_tokens: int) -> list[int]:
    text = _FILLER * (n_tokens // 8 + 5)
    toks = llm.tokenize(text.encode("utf-8"), add_bos=True)
    if len(toks) < n_tokens:
        raise ValueError(f"filler text too short for n_tokens={n_tokens}")
    return toks[:n_tokens]


def _sync(llm) -> None:
    """Block until the most recent eval()'s GPU work has finished.

    llama_decode() returns before its Metal work completes, and
    llama-cpp-python 0.3.x's eval() does not read logits (logits_all=False),
    so nothing inside eval() waits; llama_get_logits_ith() calls
    llama_synchronize() first. Without this, calibrate_decode()'s first
    timed step absorbed the whole untimed prefill before it, inflating mean
    TBT in proportion to ctx -- the SmolLM2 profile's decode slope came out
    ~3x too steep (0.0139 vs ~0.0044 ms/ctx-token). See
    core/SYNC_AUDIT_FINDINGS.md."""
    llm._ctx.get_logits_ith(-1)


def _eval_sync(llm, tokens: list[int]) -> None:
    llm.eval(tokens)
    _sync(llm)


def calibrate_prefill(llm, token_counts: list[int], trials: int) -> list[dict]:
    points = []
    warm = _build_tokens(llm, token_counts[0])
    llm.reset()
    _eval_sync(llm, warm)  # global warm-up, discarded

    for n_tok in token_counts:
        tokens = _build_tokens(llm, n_tok)
        llm.reset()
        _eval_sync(llm, tokens)  # per-shape warm-up, discarded

        times = []
        for _ in range(trials):
            llm.reset()
            t0 = _now_ms()
            _eval_sync(llm, tokens)
            times.append(_now_ms() - t0)

        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"  N={n_tok:5d}  prefill={mean_t:8.1f}ms  std={std_t:6.2f}")
        points.append({"n_tokens": n_tok, "mean_ms": mean_t, "std_ms": std_t})
    return points


def calibrate_decode(llm, ctx_tokens_list: list[int], n_decode: int = 32) -> list[dict]:
    results = []
    for ctx in ctx_tokens_list:
        tokens = _build_tokens(llm, ctx)
        llm.reset()
        _eval_sync(llm, tokens)  # untimed prefill; synced so it can't leak into step 1
        tbt = []
        for _ in range(n_decode):
            t0 = _now_ms()
            tok = llm.sample()
            _eval_sync(llm, [tok])
            tbt.append(_now_ms() - t0)
        results.append({"ctx_tokens": ctx, "mean_ms": statistics.mean(tbt)})
    return results


def fit_quadratic_with_loocv(points: list[dict]) -> dict:
    n_arr = np.array([p["n_tokens"] for p in points], dtype=float)
    t_arr = np.array([p["mean_ms"] for p in points], dtype=float)
    std_arr = np.array([max(p["std_ms"], 1e-3) for p in points], dtype=float)
    w = 1.0 / std_arr

    coeffs = np.polyfit(n_arr, t_arr, 2, w=w)
    gamma, beta, alpha = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    y_pred = np.polyval(coeffs, n_arr)
    ss_res = np.sum((t_arr - y_pred) ** 2)
    ss_tot = np.sum((t_arr - np.mean(t_arr)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
    insample_mape = float(np.mean(np.abs(y_pred - t_arr) / t_arr) * 100)

    loocv_apes = []
    for i in range(len(n_arr)):
        mask = np.arange(len(n_arr)) != i
        c_i = np.polyfit(n_arr[mask], t_arr[mask], 2, w=w[mask])
        pred_i = float(np.polyval(c_i, n_arr[i]))
        loocv_apes.append(abs(pred_i - t_arr[i]) / t_arr[i] * 100)
    loocv_mape = float(np.mean(loocv_apes))
    # The worst single held-out point, not just the average -- see
    # core/TIER4_CLOSED_LOOP_FINDINGS.md's second follow-up: a safety
    # margin sized to the MEAN LOOCV error gets breached about as often as
    # real error exceeds that mean, by construction (confirmed: Qwen/MLX's
    # margin, built from its 2.05% mean LOOCV, missed its SLA on 2/2 real
    # runs with 2.4-2.6% real error). PrefillBudgetTechnique prefers this
    # field when present.
    loocv_max_ape = float(np.max(loocv_apes))

    return {
        "gamma": gamma, "beta": beta, "alpha": alpha, "r2": r2,
        "insample_mape_pct": insample_mape, "loocv_mape_pct": loocv_mape,
        "loocv_max_ape_pct": loocv_max_ape,
        "domain_n": [float(n_arr.min()), float(n_arr.max())],
        "n_points": len(points), "points": points,
    }


def fit_decode_linear(points: list[dict]) -> dict:
    """Linear TBT(ctx) = overhead_ms + per_ctx_token_ms*ctx.

    Clamps a negative per_ctx_token_ms to 0 (flat model): more context
    cannot physically make a token cheaper to decode, so a negative slope
    from a handful of narrow-range points is noise, not a real effect --
    left unclamped it can predict negative latency outside that narrow
    range (observed for qwen25_05b_mlx_profile: -0.011ms/token from 5
    points in ctx [32,384], implying negative TBT by ctx~2048). The raw,
    unclamped fit is kept in the returned dict for transparency.
    """
    ctx = np.array([p["ctx_tokens"] for p in points], dtype=float)
    tbt = np.array([p["mean_ms"] for p in points], dtype=float)
    coeffs = np.polyfit(ctx, tbt, 1)
    raw_slope, raw_overhead = float(coeffs[0]), float(coeffs[1])

    result = {
        "overhead_ms": raw_overhead,
        "per_ctx_token_ms": raw_slope,
        "raw_fit_overhead_ms": raw_overhead,
        "raw_fit_per_ctx_token_ms": raw_slope,
        "domain_ctx": [float(ctx.min()), float(ctx.max())],
        "clamped_negative_slope": False,
    }
    if raw_slope < 0:
        result["per_ctx_token_ms"] = 0.0
        result["overhead_ms"] = float(np.mean(tbt))
        result["clamped_negative_slope"] = True
    return result


def run_calibration(
    model_id: str,
    gguf_repo: str,
    gguf_filename: str,
    base_config_repo: str,
    token_counts: Optional[list[int]] = None,
    trials: int = 8,
    n_ctx: int = 4096,
) -> dict:
    from llama_cpp import Llama

    token_counts = token_counts or DEFAULT_TOKEN_COUNTS

    print(f"=== Auto-calibrating {model_id} (llama.cpp/Metal) ===")
    arch = fetch_architecture(base_config_repo)
    print(f"[architecture] {arch}")

    gguf_path = download_gguf(gguf_repo, gguf_filename)

    print(f"[load] {gguf_path}")
    llm = Llama(model_path=str(gguf_path), n_gpu_layers=-1, n_ctx=n_ctx, verbose=False)

    print("[calibrate] prefill sweep")
    prefill_points = calibrate_prefill(llm, token_counts, trials)
    prefill_fit = fit_quadratic_with_loocv(prefill_points)
    print(f"  gamma={prefill_fit['gamma']:.6e} beta={prefill_fit['beta']:.6f} "
          f"alpha={prefill_fit['alpha']:.2f}  R^2={prefill_fit['r2']:.4f}  "
          f"LOOCV={prefill_fit['loocv_mape_pct']:.2f}%")

    print("[calibrate] decode sweep")
    decode_points = calibrate_decode(llm, token_counts[:5])
    decode_fit = fit_decode_linear(decode_points)
    print(f"  TBT(ctx) ~= {decode_fit['overhead_ms']:.3f} + "
          f"{decode_fit['per_ctx_token_ms']:.6f}*ctx  ms")

    profile = {
        "model_id": model_id,
        "backend": "llama.cpp",
        "hardware": "Apple M3 base (10-core GPU, 8GB unified, 100GB/s), Metal offload",
        "gguf_repo": gguf_repo,
        "gguf_filename": gguf_filename,
        "supports_vision": False,
        "architecture": arch,
        "kv_bytes_per_token": arch["kv_bytes_per_token"],
        "bandwidth_gbps": 100.0,
        "prefill": prefill_fit,
        "decode": decode_fit,
    }
    return profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--gguf-repo", required=True)
    ap.add_argument("--gguf-filename", required=True)
    ap.add_argument("--base-config-repo", required=True)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    profile = run_calibration(
        model_id=args.model_id,
        gguf_repo=args.gguf_repo,
        gguf_filename=args.gguf_filename,
        base_config_repo=args.base_config_repo,
        trials=args.trials,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2))
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
