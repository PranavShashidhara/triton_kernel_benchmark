"""Week 7: PTX extraction for Triton kernels + helper notes for CUDA side.

Triton side runs anywhere Triton runs (Colab included).
CUDA side (nvcc -ptx / cuobjdump --dump-sass) needs the CUDA toolkit — run
those on the RunPod box where the Project 1 kernels build.

Run:  python -m analysis.extract_ptx
Writes results/ptx/matmul_N{size}.ptx and attention_seq{len}.ptx
"""

import os
import re

import torch

from kernels.matmul_triton import matmul, matmul_kernel
from kernels.attention_triton import fused_attention, fused_attention_kernel

OUT_DIR = "results/ptx"


def _ptx_from_cache(jit_fn) -> str | None:
    """Pull PTX for the most recently compiled specialization of a @triton.jit fn."""
    cache = getattr(jit_fn, "cache", None) or {}
    for _device, kernels in cache.items():
        for _key, compiled in kernels.items():
            asm = getattr(compiled, "asm", None)
            if asm and "ptx" in asm:
                return asm["ptx"]
    return None


def summarize_ptx(ptx: str) -> dict:
    """Quick stats worth citing in the README before reading the PTX by hand."""
    return {
        "lines": ptx.count("\n"),
        "registers_decl": len(re.findall(r"\.reg \.\w+", ptx)),
        "cp_async": len(re.findall(r"cp\.async", ptx)),
        "mma_sync": len(re.findall(r"mma\.sync", ptx)),
        "shared_ld": len(re.findall(r"ld\.shared", ptx)),
        "shared_st": len(re.findall(r"st\.shared", ptx)),
        "global_ld": len(re.findall(r"ld\.global", ptx)),
        "bar_sync": len(re.findall(r"bar\.sync", ptx)),
    }


def dump(name: str, ptx: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.ptx")
    with open(path, "w") as f:
        f.write(ptx)
    stats = summarize_ptx(ptx)
    print(f"{path}  |  " + "  ".join(f"{k}={v}" for k, v in stats.items()))


def main():
    # Compile at a representative size, then pull PTX from the JIT cache
    n = 2048
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    matmul(a, b)
    ptx = _ptx_from_cache(matmul_kernel)
    if ptx:
        dump(f"matmul_N{n}", ptx)
    else:
        print("No PTX found for matmul — check Triton version's cache layout")

    seq, head_dim = 2048, 64
    q, k, v = (torch.randn((1, 8, seq, head_dim), device="cuda", dtype=torch.float16)
               for _ in range(3))
    fused_attention(q, k, v, causal=True)
    ptx = _ptx_from_cache(fused_attention_kernel)
    if ptx:
        dump(f"attention_seq{seq}", ptx)
    else:
        print("No PTX found for attention — check Triton version's cache layout")

    print(
        "\nCUDA-side extraction (run on the RunPod box with the toolkit):\n"
        "  nvcc -ptx -arch=sm_80 your_kernel.cu -o results/ptx/cuda_kernel.ptx\n"
        "  nvcc -cubin -arch=sm_80 your_kernel.cu -o kernel.cubin\n"
        "  cuobjdump --dump-sass kernel.cubin > results/ptx/cuda_kernel.sass\n"
    )


if __name__ == "__main__":
    main()
