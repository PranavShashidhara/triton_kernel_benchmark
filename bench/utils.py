"""Shared benchmarking and correctness utilities.

Uses CUDA-event timing, warmup iterations, and median-of-N latency.
Error metrics: max absolute error, mean relative error, cosine similarity.
"""

import torch


# ---------------------------------------------------------------- correctness

def error_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    out_f = out.float().flatten()
    ref_f = ref.float().flatten()
    abs_err = (out_f - ref_f).abs()
    denom = ref_f.abs().clamp_min(1e-6)
    return {
        "max_abs_error": abs_err.max().item(),
        "mean_rel_error": (abs_err / denom).mean().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            out_f.unsqueeze(0), ref_f.unsqueeze(0)
        ).item(),
    }


def check_close(out, ref, atol=1e-2, rtol=1e-2, label=""):
    ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    metrics = error_metrics(out, ref)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: max_abs={metrics['max_abs_error']:.2e} "
          f"mean_rel={metrics['mean_rel_error']:.2e} cos={metrics['cosine_similarity']:.6f}")
    return ok, metrics


# -------------------------------------------------------------------- timing

def bench(fn, warmup: int = 10, iters: int = 50) -> float:
    """Median latency in milliseconds via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def gemm_tflops(M: int, N: int, K: int, ms: float) -> float:
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


def attention_tflops(batch: int, heads: int, seq: int, head_dim: int, ms: float,
                     causal: bool = False) -> float:
    # QK^T + PV: 2 * (2 * seq^2 * head_dim) per head; causal halves the work
    flops = 4.0 * batch * heads * seq * seq * head_dim
    if causal:
        flops /= 2
    return flops / (ms * 1e-3) / 1e12


def device_report() -> dict:
    props = torch.cuda.get_device_properties(0)
    info = {
        "name": props.name,
        "sm": f"sm_{props.major}{props.minor}",
        "sm_count": props.multi_processor_count,
        "memory_gb": round(props.total_memory / 1e9, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info
