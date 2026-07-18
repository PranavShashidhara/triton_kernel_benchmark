"""Benchmark: Triton GEMM vs cuBLAS (and CUDA WMMA kernels if installed).

Run:  python -m bench.bench_gemm
Writes results/data/gemm_bench.csv
"""

import csv
import os

import torch

from kernels.matmul_triton import matmul as triton_matmul
from bench.utils import bench, check_close, device_report, gemm_tflops

SIZES = [512, 1024, 2048, 4096]
OUT_CSV = "results/data/gemm_bench.csv"


def try_import_cuda_ext():
    """Import CUDA WMMA extension if available on this machine."""
    try:
        import my_kernels  # built via torch.utils.cpp_extension
        return my_kernels
    except ImportError:
        return None


def main():
    print("Device:")
    device_report()
    cuda_ext = try_import_cuda_ext()
    if cuda_ext is None:
        print("\n(CUDA extension not found — benchmarking Triton vs cuBLAS only.)\n")

    rows = []
    for n in SIZES:
        a = torch.randn((n, n), device="cuda", dtype=torch.float16)
        b = torch.randn((n, n), device="cuda", dtype=torch.float16)

        ref = a @ b  # cuBLAS
        out = triton_matmul(a, b)
        check_close(out, ref, label=f"triton gemm {n}x{n}")

        ms_cublas = bench(lambda: a @ b)
        ms_triton = bench(lambda: triton_matmul(a, b))
        row = {
            "N": n,
            "cublas_ms": round(ms_cublas, 4),
            "cublas_tflops": round(gemm_tflops(n, n, n, ms_cublas), 2),
            "triton_ms": round(ms_triton, 4),
            "triton_tflops": round(gemm_tflops(n, n, n, ms_triton), 2),
            "triton_vs_cublas_pct": round(100 * ms_cublas / ms_triton, 1),
        }

        if cuda_ext is not None:
            ms_cuda = bench(lambda: cuda_ext.wmma_gemm(a, b))
            row["cuda_wmma_ms"] = round(ms_cuda, 4)
            row["cuda_wmma_tflops"] = round(gemm_tflops(n, n, n, ms_cuda), 2)

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
