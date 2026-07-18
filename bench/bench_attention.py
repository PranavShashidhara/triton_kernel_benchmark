"""Benchmark: Triton fused attention vs PyTorch SDPA vs naive.

Covers seq lengths 512-8K and head dims 64/128, matching the CUDA
attention benchmark grid for direct comparison.

Run:  python -m bench.bench_attention
Writes results/data/attention_bench.csv
"""

import csv
import os

import torch
import torch.nn.functional as F

from kernels.attention_triton import fused_attention
from bench.utils import attention_tflops, bench, check_close, device_report

BATCH = 4
HEADS = 8
SEQ_LENS = [512, 1024, 2048, 4096, 8192]
HEAD_DIMS = [64, 128]
OUT_CSV = "results/data/attention_bench.csv"


def naive_attention(q, k, v, causal=False):
    """O(N^2)-memory baseline that materializes the full attention matrix."""
    scale = q.shape[-1] ** -0.5
    s = (q @ k.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(s.shape[-2:], device=s.device, dtype=torch.bool), 1)
        s = s.masked_fill(mask, float("-inf"))
    return F.softmax(s.float(), dim=-1).to(v.dtype) @ v


def main():
    print("Device:")
    device_report()

    rows = []
    for head_dim in HEAD_DIMS:
        for seq in SEQ_LENS:
            for causal in (False, True):
                q, k, v = (
                    torch.randn((BATCH, HEADS, seq, head_dim), device="cuda", dtype=torch.float16)
                    for _ in range(3)
                )

                ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                out = fused_attention(q, k, v, causal=causal)
                check_close(out, ref, atol=1e-2,
                            label=f"attn seq={seq} d={head_dim} causal={causal}")

                ms_sdpa = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
                ms_triton = bench(lambda: fused_attention(q, k, v, causal=causal))

                # Naive OOMs at long seq — record that fact instead of crashing
                try:
                    torch.cuda.reset_peak_memory_stats()
                    ms_naive = bench(lambda: naive_attention(q, k, v, causal=causal), warmup=3, iters=10)
                    naive_peak_gb = torch.cuda.max_memory_allocated() / 1e9
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    ms_naive, naive_peak_gb = None, None

                torch.cuda.reset_peak_memory_stats()
                fused_attention(q, k, v, causal=causal)
                triton_peak_gb = torch.cuda.max_memory_allocated() / 1e9

                row = {
                    "seq": seq, "head_dim": head_dim, "causal": causal,
                    "sdpa_ms": round(ms_sdpa, 4),
                    "triton_ms": round(ms_triton, 4),
                    "naive_ms": round(ms_naive, 4) if ms_naive else "OOM",
                    "triton_tflops": round(attention_tflops(BATCH, HEADS, seq, head_dim, ms_triton, causal), 2),
                    "triton_vs_sdpa_pct": round(100 * ms_sdpa / ms_triton, 1),
                    "triton_peak_gb": round(triton_peak_gb, 3),
                    "naive_peak_gb": round(naive_peak_gb, 3) if naive_peak_gb else "OOM",
                }
                rows.append(row)
                print(row)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
