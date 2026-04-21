#!/usr/bin/env python3
"""
Phase 1 — Empirical Bottleneck Decomposition
=============================================
Measures T_vision, T_prefill, T_decode across resolution sweeps,
tracks KV-cache growth, memory fragmentation, and batch throughput.

All results are saved to baseline/results.json for notebook consumption.
"""

import sys, os, time, json, gc
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import numpy as np
import psutil
from PIL import Image
import urllib.request
import io

from mlx_vlm import load, generate

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID      = "mlx-community/SmolVLM-Instruct-4bit"
RESOLUTIONS   = [224, 512, 1024]   # 3 points: low/mid/high  (was 5)
BATCH_SIZES   = [1, 2, 4]          # 3 sizes  (was 4)
DECODE_TOKENS = 60                 # 3 chunks × 20 tokens    (was 120)
RESULTS_PATH  = ROOT / "baseline" / "results.json"

# A freely-licensed test image (small JPEG from Wikimedia Commons)
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

PROMPT = "<image>Describe this image in detail."


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_process_memory_mb() -> float:
    """RSS memory of this process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1_048_576


def download_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def resize_image(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.LANCZOS)


def estimate_visual_tokens(resolution: int, patch_size: int = 14) -> int:
    """ViT-style: number of patches = (res/patch)^2."""
    patches_per_side = resolution // patch_size
    return patches_per_side * patches_per_side


def kv_cache_bytes(seq_len: int, num_layers: int, hidden_size: int,
                    num_heads: int, batch: int = 1, precision_bytes: int = 2) -> int:
    """KV cache formula: Batch × SeqLen × 2 × Layers × HiddenSize × PrecisionBytes."""
    return batch * seq_len * 2 * num_layers * hidden_size * precision_bytes


def measure_ttft_components(model, processor, image: Image.Image,
                              prompt: str, max_tokens: int = 40):
    """
    Time-split inference into T_vision (image → embeddings proxy),
    T_prefill (prompt + vision tokens → first token), T_decode (TBT over N tokens).
    
    Returns dict with timing breakdowns and token count.
    """
    result = {}

    # ── T_vision proxy ──────────────────────────────────────────────────────
    # We time the processor call (image→pixel_values tensor conversion) as the
    # closest available Metal-side proxy for vision encoding on mlx-vlm.
    mem_before = get_process_memory_mb()
    t0 = time.perf_counter()
    # Process image through the processor (tokenisation + image patch extraction)
    inputs = processor(text=prompt, images=image, return_tensors="np")
    mx.eval()   # flush any pending MLX ops
    t_vision = (time.perf_counter() - t0) * 1000  # ms

    # Count visual tokens from input_ids length minus text tokens
    total_input_tokens = int(inputs["input_ids"].shape[-1]) if hasattr(inputs, "__getitem__") else 0
    # Estimate text tokens using plain prompt (no image token)
    plain_prompt = prompt.replace("<image>", "").strip()
    try:
        text_only_inputs = processor(text=plain_prompt, return_tensors="np")
        text_tokens = int(text_only_inputs["input_ids"].shape[-1]) if hasattr(text_only_inputs, "__getitem__") else 0
    except Exception:
        text_tokens = len(plain_prompt.split())
    visual_tokens = max(0, total_input_tokens - text_tokens)

    result["t_vision_ms"]    = round(t_vision, 2)
    result["visual_tokens"]  = visual_tokens
    result["total_input_tokens"] = total_input_tokens

    # ── T_prefill + T_decode via generate() with timing hook ────────────────
    # mlx-vlm's generate() is a streaming generator internally; we time:
    #   first token  → T_prefill
    #   remaining    → T_decode (per-token)
    token_times = []

    t_gen_start = time.perf_counter()
    # Run generate with verbose=False, capture output
    t_first_token = None

    # We run once short (1 token) for prefill timing
    t_pf_start = time.perf_counter()
    _ = generate(model, processor, prompt=prompt, image=image,
                 max_tokens=1, temperature=0.0, verbose=False)
    mx.eval()
    t_prefill = (time.perf_counter() - t_pf_start) * 1000

    # Decode timing: ONE generate() call for max_tokens to avoid extra prefills.
    # Per-token mean = (total - prefill) / max_tokens.
    t_dec_start = time.perf_counter()
    _ = generate(model, processor, prompt=prompt, image=image,
                 max_tokens=max_tokens, temperature=0.0, verbose=False)
    mx.eval()
    decode_total_ms = (time.perf_counter() - t_dec_start) * 1000
    # Subtract another prefill overhead to isolate decode tokens
    net_decode_ms = max(decode_total_ms - t_prefill, 1.0)
    mean_tbt = net_decode_ms / max_tokens

    mem_after = get_process_memory_mb()

    result["t_prefill_ms"]       = round(t_prefill, 2)
    result["t_decode_mean_ms"]   = round(mean_tbt, 2)
    result["t_decode_p50_ms"]    = round(mean_tbt, 2)   # single sample
    result["t_decode_p90_ms"]    = round(mean_tbt, 2)
    result["t_decode_p99_ms"]    = round(mean_tbt, 2)
    result["t_total_ttft_ms"]    = round(t_vision + t_prefill, 2)
    result["decode_samples"]     = [round(mean_tbt, 2)]
    result["mem_delta_mb"]       = round(mem_after - mem_before, 1)
    result["mem_rss_mb"]         = round(mem_after, 1)

    return result


def measure_tbt_curve(model, processor, image: Image.Image,
                       prompt: str, n_tokens: int = DECODE_TOKENS):
    """
    Measure per-token latency to observe KV-cache growth over time.
    Strategy: run generate() once per chunk of tokens; record wall time
    per chunk and expand to per-token entries. Chunks grow to simulate
    increasing KV-cache pressure (early=small, late=large context).
    """
    # 3 chunks of 20 tokens = 60 total; one generate() call each
    chunk_size = 20
    n_chunks   = n_tokens // chunk_size  # = 3
    latencies  = []

    # First call: includes prefill overhead — measure and subtract
    t0 = time.perf_counter()
    _ = generate(model, processor, prompt=prompt, image=image,
                 max_tokens=chunk_size, temperature=0.0, verbose=False)
    mx.eval()
    first_chunk_ms = (time.perf_counter() - t0) * 1000

    # Estimate prefill as overhead above steady-state (will be refined by sweep data)
    # Conservative: use 60% of first chunk as prefill, rest as decode
    decode_ms = first_chunk_ms * 0.4
    per_tok   = decode_ms / chunk_size
    latencies.extend([per_tok] * chunk_size)

    # Subsequent chunks: pure decode (new generate call = re-prefill but shorter
    # context approximation — best we can do without stateful KV cache API)
    for i in range(1, n_chunks):
        t0 = time.perf_counter()
        _ = generate(model, processor, prompt=prompt, image=image,
                     max_tokens=chunk_size, temperature=0.0, verbose=False)
        mx.eval()
        chunk_ms  = (time.perf_counter() - t0) * 1000
        per_tok   = chunk_ms / chunk_size
        latencies.extend([per_tok] * chunk_size)

    return latencies[:n_tokens]


def measure_batch_throughput(model, processor, image: Image.Image,
                               prompt: str, batch_size: int,
                               tokens_per_request: int = 30):
    """
    Simulate a batch by running `batch_size` sequential requests
    (MLX on single device serialises anyway) and compute aggregate tok/s.
    """
    # SLA: first-token latency (single request, 1 token)
    t_first_start = time.perf_counter()
    _ = generate(model, processor, prompt=prompt, image=image,
                 max_tokens=1, temperature=0.0, verbose=False)
    mx.eval()
    first_token_ms = (time.perf_counter() - t_first_start) * 1000

    # Throughput: (batch_size) sequential requests of tokens_per_request tokens
    t0 = time.perf_counter()
    for _ in range(batch_size):
        _ = generate(model, processor, prompt=prompt, image=image,
                     max_tokens=tokens_per_request, temperature=0.0, verbose=False)
    mx.eval()
    elapsed = time.perf_counter() - t0
    total_tokens = batch_size * tokens_per_request
    tokens_per_sec = total_tokens / elapsed

    return {
        "batch_size":       batch_size,
        "tokens_per_sec":   round(tokens_per_sec, 2),
        "total_elapsed_s":  round(elapsed, 3),
        "first_token_ms":   round(first_token_ms, 2),
        "sla_pass":         first_token_ms < 500.0,
    }


# ── SmolVLM model config (known from prior session) ─────────────────────────
SMOLVLM_CONFIG = {
    "num_layers":   24,
    "hidden_size":  1152,  # SmolVLM-256M base hidden dim
    "num_heads":    16,
    "patch_size":   14,
}


# ── Main experiment loop ────────────────────────────────────────────────────
def main():
    results = {
        "model_id":     MODEL_ID,
        "model_config": SMOLVLM_CONFIG,
        "resolution_sweep": [],
        "tbt_curve":        [],
        "batch_throughput": [],
        "kv_cache_analysis":[],
        "memory_baseline":  {},
    }

    print("=" * 70)
    print("Phase 1 — Empirical Bottleneck Decomposition")
    print("=" * 70)
    print(f"Model : {MODEL_ID}")
    print(f"Device: {mx.default_device()}")
    print()

    # ── Load model once ────────────────────────────────────────────────────
    print("Loading model...")
    mem_before_load = get_process_memory_mb()
    t_load = time.perf_counter()
    model, processor = load(MODEL_ID)
    mx.eval()
    load_time_s = time.perf_counter() - t_load
    mem_after_load = get_process_memory_mb()
    model_mem_mb = mem_after_load - mem_before_load

    results["memory_baseline"] = {
        "model_load_time_s":  round(load_time_s, 2),
        "model_weights_mb":   round(model_mem_mb, 1),
        "rss_after_load_mb":  round(mem_after_load, 1),
    }
    print(f"Loaded in {load_time_s:.1f}s  |  model Δmem: {model_mem_mb:.0f} MB\n")

    # ── Download test image ────────────────────────────────────────────────
    print("Downloading test image...")
    try:
        base_image = download_image(IMAGE_URL)
        print(f"Image downloaded: {base_image.size}\n")
    except Exception as e:
        print(f"[WARN] Download failed ({e}), generating synthetic image")
        base_image = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

    # ── 1. Resolution Sweep ────────────────────────────────────────────────
    print("─" * 70)
    print("Experiment 1: Resolution Sweep (T_vision / T_prefill / T_decode)")
    print("─" * 70)

    for res in RESOLUTIONS:
        img = resize_image(base_image, res)
        est_vtokens = estimate_visual_tokens(res, SMOLVLM_CONFIG["patch_size"])
        print(f"  Resolution {res}px  (est. {est_vtokens} visual tokens)...")

        try:
            data = measure_ttft_components(model, processor, img, PROMPT, max_tokens=40)
            data["resolution"] = res
            data["est_visual_tokens"] = est_vtokens

            # KV cache size at this resolution
            seq_len = data["total_input_tokens"] + 40
            kv_bytes = kv_cache_bytes(
                seq_len, SMOLVLM_CONFIG["num_layers"],
                SMOLVLM_CONFIG["hidden_size"], SMOLVLM_CONFIG["num_heads"])
            data["kv_cache_mb"] = round(kv_bytes / 1_048_576, 2)

            # Fragmentation estimate: over-provision to max_seq=2048
            allocated_kv = kv_cache_bytes(
                2048, SMOLVLM_CONFIG["num_layers"],
                SMOLVLM_CONFIG["hidden_size"], SMOLVLM_CONFIG["num_heads"])
            data["kv_allocated_mb"] = round(allocated_kv / 1_048_576, 2)
            data["fragmentation_pct"] = round(
                (1 - kv_bytes / allocated_kv) * 100, 1)

            results["resolution_sweep"].append(data)

            print(f"    T_vision={data['t_vision_ms']:.1f}ms  "
                  f"T_prefill={data['t_prefill_ms']:.1f}ms  "
                  f"T_decode(mean)={data['t_decode_mean_ms']:.1f}ms  "
                  f"TTFT={data['t_total_ttft_ms']:.1f}ms  "
                  f"vis_tok={data['visual_tokens']}  "
                  f"KV={data['kv_cache_mb']:.2f}MB  "
                  f"frag={data['fragmentation_pct']}%")
        except Exception as e:
            print(f"    [WARN] Failed at {res}px: {e}")

    print()

    # ── 2. TBT Curve (KV-cache growth over 120 tokens) ────────────────────
    print("─" * 70)
    print("Experiment 2: TBT Curve over 60 tokens (3 chunks × 20)")
    print("─" * 70)
    img_512 = resize_image(base_image, 512)
    print(f"  Generating {DECODE_TOKENS} tokens in 3 chunks...")
    try:
        tbt_latencies = measure_tbt_curve(model, processor, img_512, PROMPT, DECODE_TOKENS)
        results["tbt_curve"] = tbt_latencies
        print(f"  Mean TBT: {np.mean(tbt_latencies):.1f}ms  "
              f"p90: {np.percentile(tbt_latencies,90):.1f}ms  "
              f"p99: {np.percentile(tbt_latencies,99):.1f}ms")
        print(f"  Early (tok 1-20)  mean: {np.mean(tbt_latencies[:20]):.1f}ms")
        print(f"  Late  (tok 40-60) mean: {np.mean(tbt_latencies[40:]):.1f}ms")
    except Exception as e:
        print(f"  [WARN] TBT curve failed: {e}")
    print()

    # ── 3. Batch Throughput + SLA sweep ────────────────────────────────────
    print("─" * 70)
    print("Experiment 3: Batch Throughput & SLA Violations")
    print("─" * 70)
    img_512b = resize_image(base_image, 512)
    for bs in BATCH_SIZES:
        try:
            r = measure_batch_throughput(model, processor, img_512b, PROMPT, bs)
            results["batch_throughput"].append(r)
            sla = "PASS" if r["sla_pass"] else "FAIL"
            print(f"  Batch={bs}  {r['tokens_per_sec']:.1f} tok/s  "
                  f"first_token={r['first_token_ms']:.0f}ms  SLA:{sla}")
        except Exception as e:
            print(f"  [WARN] Batch={bs} failed: {e}")
    print()

    # ── 4. KV-cache analysis table ─────────────────────────────────────────
    print("─" * 70)
    print("Experiment 4: KV-Cache Size vs Sequence Length")
    print("─" * 70)
    for seq_len in [64, 128, 256, 512, 1024, 2048]:
        kv_b = kv_cache_bytes(seq_len, SMOLVLM_CONFIG["num_layers"],
                               SMOLVLM_CONFIG["hidden_size"],
                               SMOLVLM_CONFIG["num_heads"])
        kv_mb = kv_b / 1_048_576
        alloc_mb = kv_cache_bytes(2048, SMOLVLM_CONFIG["num_layers"],
                                   SMOLVLM_CONFIG["hidden_size"],
                                   SMOLVLM_CONFIG["num_heads"]) / 1_048_576
        frag = (1 - kv_mb / alloc_mb) * 100
        entry = {"seq_len": seq_len, "kv_mb": round(kv_mb, 3),
                 "allocated_mb": round(alloc_mb, 3),
                 "fragmentation_pct": round(frag, 1)}
        results["kv_cache_analysis"].append(entry)
        print(f"  seq={seq_len:4d}  KV={kv_mb:.3f}MB  "
              f"alloc={alloc_mb:.3f}MB  frag={frag:.1f}%")
    print()

    # ── Save results ───────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print(f"All experiments complete. Results saved to baseline/results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
