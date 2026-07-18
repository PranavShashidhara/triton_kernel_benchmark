"""torch.compile integration.

Registers the Triton kernels as PyTorch custom ops, checks for graph breaks
under torch.compile, and benchmarks against Inductor's max-autotune output.

Run:  python -m analysis.torch_compile_check
"""

import torch
from torch.library import Library, impl

from kernels.matmul_triton import matmul as triton_matmul
from kernels.attention_triton import fused_attention
from bench.utils import bench, device_report

# ------------------------------------------------------- custom op registration

mylib = Library("mylib", "DEF")
mylib.define("triton_gemm(Tensor a, Tensor b) -> Tensor")
mylib.define("triton_attention(Tensor q, Tensor k, Tensor v, bool causal) -> Tensor")


@impl(mylib, "triton_gemm", "CUDA")
def _gemm_cuda(a, b):
    return triton_matmul(a, b)


@impl(mylib, "triton_gemm", "Meta")
def _gemm_meta(a, b):
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)


@impl(mylib, "triton_attention", "CUDA")
def _attn_cuda(q, k, v, causal):
    return fused_attention(q, k, v, causal=causal)


@impl(mylib, "triton_attention", "Meta")
def _attn_meta(q, k, v, causal):
    return torch.empty_like(q)


# ------------------------------------------------------------------ graph checks

def check_graph_breaks():
    @torch.compile
    def fwd(x, w):
        return torch.ops.mylib.triton_gemm(x, w)

    x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    w = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    explanation = torch._dynamo.explain(fwd)(x, w)
    print(f"graph breaks: {explanation.graph_break_count}")
    print(f"graphs captured: {explanation.graph_count}")
    if explanation.graph_break_count > 0:
        for reason in explanation.break_reasons:
            print(f"  break: {reason}")
    return explanation


def bench_vs_inductor():
    n = 2048
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)

    def eager(x, w):
        return x @ w

    inductor_max = torch.compile(eager, mode="max-autotune")
    inductor_max(a, b)  # warm compile

    custom = torch.compile(lambda x, w: torch.ops.mylib.triton_gemm(x, w))
    custom(a, b)

    print(f"eager cuBLAS:           {bench(lambda: eager(a, b)):.4f} ms")
    print(f"inductor max-autotune:  {bench(lambda: inductor_max(a, b)):.4f} ms")
    print(f"custom triton op:       {bench(lambda: custom(a, b)):.4f} ms")


def main():
    print("Device:")
    device_report()
    print("\n=== Graph break check ===")
    check_graph_breaks()
    print("\n=== Benchmark vs Inductor ===")
    bench_vs_inductor()


if __name__ == "__main__":
    main()
