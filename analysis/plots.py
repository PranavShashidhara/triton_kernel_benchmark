"""Performance analysis charts.

Run:  python -m analysis.plots
Reads results/data/*.csv, writes results/charts/*.png
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = "results/data"
CHARTS = "results/charts"


def _read(path):
    with open(path, newline="") as f:
        return [row for row in csv.DictReader(f)]


def plot_performance_surface():
    """Heatmap per matrix size: (num_warps x num_stages) -> TFLOPS.

    One heatmap per (size, block shape) keeps each panel readable.
    """
    rows = [r for r in _read(f"{DATA}/sweep.csv") if r["tflops"] != "FAIL"]
    sizes = sorted({int(r["N"]) for r in rows})
    blocks = sorted({(int(r["block_m"]), int(r["block_n"])) for r in rows})

    fig, axes = plt.subplots(len(blocks), len(sizes),
                             figsize=(4 * len(sizes), 3.5 * len(blocks)),
                             squeeze=False)
    for bi, (bm, bn) in enumerate(blocks):
        for si, n in enumerate(sizes):
            sub = [r for r in rows
                   if int(r["N"]) == n and int(r["block_m"]) == bm and int(r["block_n"]) == bn]
            warps = sorted({int(r["num_warps"]) for r in sub})
            stages = sorted({int(r["num_stages"]) for r in sub})
            grid = np.full((len(warps), len(stages)), np.nan)
            for r in sub:
                grid[warps.index(int(r["num_warps"])),
                     stages.index(int(r["num_stages"]))] = float(r["tflops"])
            ax = axes[bi][si]
            im = ax.imshow(grid, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(stages)), stages)
            ax.set_yticks(range(len(warps)), warps)
            ax.set_xlabel("num_stages")
            ax.set_ylabel("num_warps")
            ax.set_title(f"N={n}, block=({bm},{bn})")
            for wi in range(len(warps)):
                for sti in range(len(stages)):
                    if not np.isnan(grid[wi, sti]):
                        ax.text(sti, wi, f"{grid[wi, sti]:.0f}",
                                ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(im, ax=ax, label="TFLOPS")
    fig.suptitle("Triton config performance surface (TFLOPS)")
    fig.tight_layout()
    out = f"{CHARTS}/performance_surface.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def plot_gemm_comparison():
    rows = _read(f"{DATA}/gemm_bench.csv")
    ns = [int(r["N"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, [float(r["cublas_tflops"]) for r in rows], "o-", label="cuBLAS")
    ax.plot(ns, [float(r["triton_tflops"]) for r in rows], "s-", label="Triton (autotuned)")
    if "cuda_wmma_tflops" in rows[0]:
        ax.plot(ns, [float(r["cuda_wmma_tflops"]) for r in rows], "^-", label="CUDA WMMA")
    ax.set_xlabel("Matrix size N (NxN)")
    ax.set_ylabel("TFLOPS")
    ax.set_xscale("log", base=2)
    ax.set_title("GEMM throughput: Triton vs cuBLAS" +
                 (" vs hand-written CUDA" if "cuda_wmma_tflops" in rows[0] else ""))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = f"{CHARTS}/gemm_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def plot_attention():
    rows = [r for r in _read(f"{DATA}/attention_bench.csv")
            if r["causal"] == "False" and r["head_dim"] == "64"]
    seqs = [int(r["seq"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(seqs, [float(r["sdpa_ms"]) for r in rows], "o-", label="PyTorch SDPA")
    ax1.plot(seqs, [float(r["triton_ms"]) for r in rows], "s-", label="Triton fused")
    naive = [(s, float(r["naive_ms"])) for s, r in zip(seqs, rows) if r["naive_ms"] != "OOM"]
    if naive:
        ax1.plot([s for s, _ in naive], [m for _, m in naive], "^-", label="Naive (3-kernel)")
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_title("Attention latency (head_dim=64)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(seqs, [float(r["triton_peak_gb"]) for r in rows], "s-", label="Triton fused (O(N))")
    naive_mem = [(s, float(r["naive_peak_gb"])) for s, r in zip(seqs, rows)
                 if r["naive_peak_gb"] != "OOM"]
    if naive_mem:
        ax2.plot([s for s, _ in naive_mem], [m for _, m in naive_mem], "^-", label="Naive (O(N^2))")
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("Peak memory (GB)")
    ax2.set_xscale("log", base=2)
    ax2.set_title("Peak memory vs sequence length")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = f"{CHARTS}/attention_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def main():
    os.makedirs(CHARTS, exist_ok=True)
    for fn in (plot_performance_surface, plot_gemm_comparison, plot_attention):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"Skipping {fn.__name__}: {e.filename} not found (run the benchmark first)")


if __name__ == "__main__":
    main()
