# Phase 4 integration smoke-test — delete this file after use
from simulation.kv_manager import ContiguousBackend, PagedBackend, kv_cache_size_mb
from simulation.w4a8_quantizer import W4A8Analyzer, SCHEMES
from simulation.vision_offloader import VisionOffloader
from simulation.sm_orchestrator import decode_starvation_analysis, batch_expansion_summary

print("All Phase 4 imports OK")

cb = ContiguousBackend()
pb = PagedBackend()
r_c = cb.allocate(1548)
r_p = pb.allocate(1548)
assert r_c.fragmentation_pct > 20, f"Contiguous frag={r_c.fragmentation_pct:.1f}% should be >20%"
assert r_p.fragmentation_pct < 4.0, f"Paged frag={r_p.fragmentation_pct:.2f}% should be <4%"
print(f"  KV backends: contiguous frag={r_c.fragmentation_pct:.1f}%  paged frag={r_p.fragmentation_pct:.2f}%  OK")

off = VisionOffloader(k_physical_layers=2)
result = off.simulate()
assert result.zero_overhead, "Should be zero-overhead on M3"
assert result.memory_saved_mb > 180, f"Expected >180 MB freed, got {result.memory_saved_mb:.1f}"
print(f"  Vision offloader: {result.memory_saved_mb:.1f} MB freed, zero_overhead={result.zero_overhead}  OK")

an = W4A8Analyzer()
w4a8 = an.analyze(SCHEMES["w4a8_gar"])
assert w4a8.tbt_gain_vs_fp16 > 2.0, f"W4A8 gain={w4a8.tbt_gain_vs_fp16:.2f}x should be >2x"
assert w4a8.memory_reduction_x == 4.0, "W4 should give 4x memory reduction"
print(f"  W4A8+GAR: TBT={w4a8.tbt_predicted_ms:.1f}ms  gain={w4a8.tbt_gain_vs_fp16:.1f}x  OK")

star = decode_starvation_analysis(max_batch=4)
assert 1 in star["paged_w4a8_gar"].safe_batch_range, "W4A8 should be safe at batch=1"
print(f"  Starvation: paged_w4a8_gar safe batches={star['paged_w4a8_gar'].safe_batch_range}  OK")

kv_mb = kv_cache_size_mb(1548, batch_size=1, quantization_bits=16)
print(f"  KV size seq=1548 FP16 batch=1: {kv_mb:.2f} MB  (baseline=168.12 MB)")

exp = batch_expansion_summary()
labels = [r.strategy for r in exp]
print(f"  Batch expansion strategies: {labels}")
print()
print("All Phase 4 assertions PASSED")
