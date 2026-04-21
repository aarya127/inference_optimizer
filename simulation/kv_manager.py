"""
KV Cache Manager — Dual-Backend Simulation (Phase 4)

Implements and compares two KV cache allocation strategies:

  ContiguousBackend  — static pre-allocation to max_seq_len.  Simple but
                       wastes 60–80% of memory via fragmentation and over-
                       reservation.  Mirrors SmolVLM baseline (results.json:
                       kv_allocated_mb=216, fragmentation=22.2% at 1548 tokens).

  PagedBackend       — PagedAttention-style block table.  KV cache is split
                       into fixed-size physical blocks (PAGE_SIZE tokens each).
                       A logical→physical block table maps sequence positions
                       to non-contiguous blocks.  Target: fragmentation < 4%.

Hardware / model constants (derived from Phase 1 measurements)
--------------------------------------------------------------
  n_layers    = 24        (SmolVLM LM depth)
  hidden_size = 1152      (KV head dimension: calibrated from kv_cache_mb data)
  dtype_bytes = 2         (FP16 baseline)
  max_seq_len = 2048      (pre-allocation ceiling for contiguous backend)
  kv_per_tok  = 110,592 bytes/token (= 2 × 24 × 1152 × 2)

Verification:  kv_per_tok × 2048 = 226,492,416 B = 216.0 MB  (matches results.json)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Model / hardware constants
# ---------------------------------------------------------------------------
LM_N_LAYERS: int = 24
LM_HIDDEN_SIZE: int = 1152          # KV hidden dimension (empirically derived)
LM_DTYPE_BYTES: int = 2             # FP16
MAX_SEQ_LEN: int = 2048             # contiguous backend ceiling
PAGE_SIZE: int = 16                 # tokens per paged block (standard vLLM default)
M3_TOTAL_MEMORY_MB: float = 8192.0  # M3 8 GB unified memory
MODEL_WEIGHTS_MB: float = 250.0     # SmolVLM 4-bit weights (≈ 500 M params × 0.5 B)
VISION_WEIGHTS_MB: float = 200.0    # SigLIP 4-bit vision encoder
OS_OVERHEAD_MB: float = 2048.0      # OS + framework overhead

# Bytes required to store KV for one token across all layers (FP16)
KV_BYTES_PER_TOKEN: int = 2 * LM_N_LAYERS * LM_HIDDEN_SIZE * LM_DTYPE_BYTES
# = 2 × 24 × 1152 × 2 = 110,592 bytes / token

# Available memory budget for KV pool
KV_POOL_BUDGET_MB: float = (
    M3_TOTAL_MEMORY_MB - MODEL_WEIGHTS_MB - VISION_WEIGHTS_MB - OS_OVERHEAD_MB
)


# ---------------------------------------------------------------------------
# Shared result dataclass
# ---------------------------------------------------------------------------

@dataclass
class KVAllocationResult:
    """Memory accounting for one sequence under a given backend."""
    backend: str                  # "contiguous" or "paged"
    seq_len: int                  # actual sequence length (tokens)
    kv_used_mb: float             # memory actually needed for this sequence
    kv_allocated_mb: float        # memory reserved by the allocator
    fragmentation_pct: float      # (allocated - used) / allocated × 100
    n_blocks: int                 # physical blocks in use (paged) or 1 (contiguous)
    waste_bytes: int              # allocated - used, in bytes
    notes: str = ""

    @property
    def efficiency_pct(self) -> float:
        return 100.0 - self.fragmentation_pct


# ---------------------------------------------------------------------------
# ContiguousBackend
# ---------------------------------------------------------------------------

class ContiguousBackend:
    """
    Baseline static allocator.  One contiguous buffer per sequence, sized to
    max_seq_len regardless of actual usage.

    Fragmentation formula from the spec:
        Frag = (Allocated − Used) / Allocated
    """

    def __init__(
        self,
        max_seq_len: int = MAX_SEQ_LEN,
        kv_bytes_per_token: int = KV_BYTES_PER_TOKEN,
    ):
        self.max_seq_len = max_seq_len
        self.kv_bytes_per_token = kv_bytes_per_token
        self._allocated_mb_per_seq = (
            max_seq_len * kv_bytes_per_token / (1024 ** 2)
        )
        self._sequences: Dict[int, int] = {}   # seq_id → seq_len
        self._next_id = 0

    @property
    def kv_allocated_mb_per_seq(self) -> float:
        return self._allocated_mb_per_seq

    def allocate(self, seq_len: int) -> KVAllocationResult:
        """Allocate a contiguous KV buffer for a sequence of `seq_len` tokens."""
        assert 0 < seq_len <= self.max_seq_len, (
            f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}"
        )
        used_bytes = seq_len * self.kv_bytes_per_token
        alloc_bytes = self.max_seq_len * self.kv_bytes_per_token
        waste_bytes = alloc_bytes - used_bytes
        frag_pct = waste_bytes / alloc_bytes * 100.0

        seq_id = self._next_id
        self._sequences[seq_id] = seq_len
        self._next_id += 1

        return KVAllocationResult(
            backend="contiguous",
            seq_len=seq_len,
            kv_used_mb=used_bytes / (1024 ** 2),
            kv_allocated_mb=alloc_bytes / (1024 ** 2),
            fragmentation_pct=frag_pct,
            n_blocks=1,
            waste_bytes=waste_bytes,
            notes=f"pre-allocated {self.max_seq_len} tokens, used {seq_len}",
        )

    def max_concurrent_seqs(self, pool_budget_mb: float = KV_POOL_BUDGET_MB) -> int:
        """How many sequences fit in the available KV pool?"""
        return int(pool_budget_mb / self._allocated_mb_per_seq)

    def sweep(self, seq_lens: List[int] | None = None) -> List[KVAllocationResult]:
        """Return allocations for a list of representative sequence lengths."""
        lengths = seq_lens or [64, 128, 256, 512, 1024, 1548, 2048]
        return [self.allocate(min(s, self.max_seq_len)) for s in lengths]


# ---------------------------------------------------------------------------
# PagedBackend
# ---------------------------------------------------------------------------

@dataclass
class PhysicalBlock:
    """One physical memory block holding PAGE_SIZE token KV pairs."""
    block_id: int
    page_size: int          # tokens this block can hold
    tokens_used: int = 0    # tokens currently stored
    ref_count: int = 0      # number of sequences pointing here

    @property
    def is_full(self) -> bool:
        return self.tokens_used >= self.page_size

    @property
    def waste_tokens(self) -> int:
        return self.page_size - self.tokens_used


@dataclass
class BlockTable:
    """Maps logical block indices (0, 1, 2 …) to physical block IDs."""
    seq_id: int
    seq_len: int
    logical_to_physical: Dict[int, int] = field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return len(self.logical_to_physical)


class PagedBackend:
    """
    PagedAttention-style block allocator.

    Memory is split into a pool of fixed-size physical blocks.  Each sequence
    maintains a BlockTable that maps its logical token positions to physical
    blocks.  Blocks are allocated on demand (no over-reservation).

    Fragmentation source: only the last block of each sequence may be
    partially filled.  Maximum waste = PAGE_SIZE − 1 tokens per sequence.

        Frag = sum(waste_per_seq) / sum(allocated_per_seq)
    """

    def __init__(
        self,
        page_size: int = PAGE_SIZE,
        kv_bytes_per_token: int = KV_BYTES_PER_TOKEN,
        pool_budget_mb: float = KV_POOL_BUDGET_MB,
    ):
        self.page_size = page_size
        self.kv_bytes_per_token = kv_bytes_per_token
        self.bytes_per_block = page_size * kv_bytes_per_token
        self.mb_per_block = self.bytes_per_block / (1024 ** 2)

        # Total physical blocks available
        self.total_blocks = int(pool_budget_mb * (1024 ** 2) / self.bytes_per_block)
        self._free_blocks: List[int] = list(range(self.total_blocks))
        self._physical_blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i, page_size=page_size)
            for i in range(self.total_blocks)
        }
        self._block_tables: Dict[int, BlockTable] = {}
        self._next_seq_id = 0

    @property
    def free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - self.free_blocks

    def _alloc_block(self) -> int:
        if not self._free_blocks:
            raise MemoryError("KV pool exhausted — no free physical blocks left.")
        block_id = self._free_blocks.pop(0)
        return block_id

    def allocate(self, seq_len: int) -> KVAllocationResult:
        """
        Allocate physical blocks for `seq_len` tokens.

        Returns a KVAllocationResult tracking actual usage vs allocated bytes.
        """
        n_blocks_needed = math.ceil(seq_len / self.page_size)

        # Allocate blocks and build block table
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        table = BlockTable(seq_id=seq_id, seq_len=seq_len)

        for logical_idx in range(n_blocks_needed):
            phys_id = self._alloc_block()
            table.logical_to_physical[logical_idx] = phys_id
            block = self._physical_blocks[phys_id]
            block.ref_count += 1
            # Fill tokens: all blocks full except (possibly) the last
            if logical_idx < n_blocks_needed - 1:
                block.tokens_used = self.page_size
            else:
                block.tokens_used = seq_len - logical_idx * self.page_size

        self._block_tables[seq_id] = table

        # Accounting
        alloc_bytes = n_blocks_needed * self.bytes_per_block
        used_bytes = seq_len * self.kv_bytes_per_token
        waste_bytes = alloc_bytes - used_bytes
        frag_pct = waste_bytes / alloc_bytes * 100.0 if alloc_bytes > 0 else 0.0

        return KVAllocationResult(
            backend="paged",
            seq_len=seq_len,
            kv_used_mb=used_bytes / (1024 ** 2),
            kv_allocated_mb=alloc_bytes / (1024 ** 2),
            fragmentation_pct=frag_pct,
            n_blocks=n_blocks_needed,
            waste_bytes=waste_bytes,
            notes=(
                f"PageSize={self.page_size}: {n_blocks_needed} blocks "
                f"({waste_bytes // self.kv_bytes_per_token} wasted tokens in last block)"
            ),
        )

    def free(self, seq_id: int) -> None:
        """Return all blocks held by a sequence back to the free pool."""
        if seq_id not in self._block_tables:
            return
        table = self._block_tables.pop(seq_id)
        for phys_id in table.logical_to_physical.values():
            block = self._physical_blocks[phys_id]
            block.ref_count -= 1
            block.tokens_used = 0
            if block.ref_count == 0:
                self._free_blocks.append(phys_id)

    def max_concurrent_seqs(self, seq_len: int) -> int:
        """How many sequences of `seq_len` tokens fit in the full pool?"""
        blocks_per_seq = math.ceil(seq_len / self.page_size)
        return self.total_blocks // blocks_per_seq

    def sweep(self, seq_lens: List[int] | None = None) -> List[KVAllocationResult]:
        """Return allocations for representative lengths (resets state first)."""
        self._free_blocks = list(range(self.total_blocks))
        for b in self._physical_blocks.values():
            b.tokens_used = 0
            b.ref_count = 0
        self._block_tables.clear()
        self._next_seq_id = 0

        lengths = seq_lens or [64, 128, 256, 512, 1024, 1548, 2048]
        results = []
        for s in lengths:
            try:
                results.append(self.allocate(s))
            except MemoryError:
                break
        return results


# ---------------------------------------------------------------------------
# KV Cache Size Calculator  (M3 scenario matrix)
# ---------------------------------------------------------------------------

def kv_cache_size_mb(
    seq_len: int,
    batch_size: int = 1,
    n_layers: int = LM_N_LAYERS,
    hidden_size: int = LM_HIDDEN_SIZE,
    dtype_bytes: int = LM_DTYPE_BYTES,
    quantization_bits: int = 16,
) -> float:
    """
    Compute KV cache memory in MB for given config.

    Parameters
    ----------
    seq_len           : sequence length (visual + text tokens)
    batch_size        : number of concurrent requests
    n_layers, hidden_size, dtype_bytes : model architecture
    quantization_bits : 16 (FP16), 8 (INT8/FP8), or 4 (INT4)

    Returns
    -------
    KV cache size in MB.
    """
    quant_factor = 16 / quantization_bits
    bytes_per_token = 2 * n_layers * hidden_size * dtype_bytes / quant_factor
    total_bytes = bytes_per_token * seq_len * batch_size
    return total_bytes / (1024 ** 2)


def kv_scenario_matrix(
    seq_lens: List[int] | None = None,
    batch_sizes: List[int] | None = None,
    quant_bits: List[int] | None = None,
) -> List[dict]:
    """
    Print a matrix of KV cache sizes for M3 planning.

    Returns a list of dicts with keys:
        seq_len, batch_size, quant_bits, kv_mb, fits_in_budget
    """
    seqs = seq_lens or [256, 512, 1024, 1548, 2048]
    batches = batch_sizes or [1, 2, 4, 8, 16]
    qtypes = quant_bits or [16, 8, 4]
    rows = []
    for s in seqs:
        for b in batches:
            for q in qtypes:
                kv_mb = kv_cache_size_mb(s, b, quantization_bits=q)
                rows.append(dict(
                    seq_len=s,
                    batch_size=b,
                    quant_bits=q,
                    kv_mb=round(kv_mb, 2),
                    fits_in_budget=kv_mb <= KV_POOL_BUDGET_MB,
                ))
    return rows


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_backends(
    seq_lens: List[int] | None = None,
) -> List[Tuple[KVAllocationResult, KVAllocationResult]]:
    """
    Return (contiguous_result, paged_result) pairs for each seq_len.
    """
    contiguous = ContiguousBackend()
    paged = PagedBackend()
    lengths = seq_lens or [64, 128, 256, 512, 1024, 1548, 2048]
    pairs = []
    for s in lengths:
        c_res = contiguous.allocate(s)
        p_res = paged.allocate(s)
        pairs.append((c_res, p_res))
    return pairs


# ---------------------------------------------------------------------------
# Self-test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("AMIO Phase 4 — KV Manager Dual-Backend Comparison")
    print("=" * 72)

    # --- Hardware budget summary ---
    print(f"\nM3 KV Pool Budget: {KV_POOL_BUDGET_MB:.0f} MB")
    print(f"  Total memory   : {M3_TOTAL_MEMORY_MB:.0f} MB")
    print(f"  Model weights  : {MODEL_WEIGHTS_MB:.0f} MB")
    print(f"  Vision encoder : {VISION_WEIGHTS_MB:.0f} MB")
    print(f"  OS overhead    : {OS_OVERHEAD_MB:.0f} MB")
    print(f"  KV bytes/token : {KV_BYTES_PER_TOKEN:,} B  "
          f"(2 × {LM_N_LAYERS} layers × {LM_HIDDEN_SIZE} hidden × {LM_DTYPE_BYTES} B FP16)")

    # --- Backend comparison table ---
    print()
    print(f"  {'seq_len':>7}  "
          f"{'Contiguous alloc':>16}  {'Contig frag%':>12}  "
          f"{'Paged alloc':>11}  {'Paged frag%':>11}")
    print("  " + "-" * 65)
    pairs = compare_backends()
    for c, p in pairs:
        match = "OK" if p.fragmentation_pct < 4.0 else "WARN"
        print(
            f"  {c.seq_len:>7}  "
            f"{c.kv_allocated_mb:>14.2f} MB  {c.fragmentation_pct:>10.1f}%  "
            f"{p.kv_allocated_mb:>9.2f} MB  {p.fragmentation_pct:>9.2f}%  {match}"
        )

    # --- Max concurrent sequences ---
    print()
    cb = ContiguousBackend()
    pb = PagedBackend()
    print(f"  Max concurrent seqs (seq_len=1548, FP16 KV):")
    print(f"    Contiguous : {cb.max_concurrent_seqs()}")
    print(f"    Paged      : {pb.max_concurrent_seqs(1548)}")
    print(f"    Paged (W4) : {PagedBackend(kv_bytes_per_token=KV_BYTES_PER_TOKEN//4).max_concurrent_seqs(1548)}")

    # --- KV scenario matrix (M3, selected rows) ---
    print()
    print("  KV Cache Size Matrix (M3, key scenarios)")
    print(f"  {'seq_len':>7}  {'batch':>5}  {'quant':>5}  {'kv_mb':>8}  {'fits?':>6}")
    print("  " + "-" * 40)
    interesting = [
        (1548, 1,  16),  # Phase 1 baseline
        (1548, 1,   4),  # W4A8
        (1548, 4,  16),  # batch=4, FP16
        (1548, 4,   4),  # batch=4, W4
        (1548, 16,  4),  # batch=16, W4
        (2048, 1,  16),  # max seq FP16
        (2048, 8,   4),  # max seq, W4, batch=8
        ( 256, 32,  4),  # short seqs, W4, large batch
    ]
    for s, b, q in interesting:
        kv_mb = kv_cache_size_mb(s, b, quantization_bits=q)
        flag = "PASS" if kv_mb <= KV_POOL_BUDGET_MB else "FAIL"
        print(f"  {s:>7}  {b:>5}  {q:>4}b  {kv_mb:>7.2f}  {flag}")

    print()
    print("KV manager self-test complete")
