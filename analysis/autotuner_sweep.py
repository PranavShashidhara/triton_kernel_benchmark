"""Systematic autotuner sweep.

Enumerates the full config space (num_stages x num_warps x block shapes)
across matrix sizes 512-4096, timing every config explicitly. Produces
the (size x config -> TFLOPS) performance surface and compares the
autotuner's chosen config against a hand-tuned CUDA pipeline depth
reference at each size.

Run:  python -m analysis.autotuner_sweep
Writes results/data/sweep.csv and results/data/autotuner_choices.csv
"""

import csv
import os

import torch
import triton

from kernels.matmul_triton import matmul, matmul_kernel, matmul_with_config
from bench.utils import bench, device_report, gemm_tflops

SIZES = [512, 1024, 2048, 4096]

# Sweep axes: stages 2-5, warps 2/4/8
STAGES = [2, 3, 4, 5]
WARPS = [2, 4, 8]
BLOCKS = [(64, 64, 32), (128, 128, 32)]

# Hand-tuned CUDA pipeline depth per size (for comparison only, not timing).
HAND_TUNED_STAGES = {512: 3, 1024: 3, 2048: 3, 4096: 4}

SWEEP_CSV = "results/data/sweep.csv"
CHOICES_CSV = "results/data/autotuner_choices.csv"


def run_sweep():
    rows = []
    for n in SIZES:
        a = torch.randn((n, n), device="cuda", dtype=torch.float16)
        b = torch.randn((n, n), device="cuda", dtype=torch.float16)
        for bm, bn, bk in BLOCKS:
            for w in WARPS:
                for s in STAGES:
                    try:
                        ms = bench(
                            lambda: matmul_with_config(a, b, bm, bn, bk, w, s),
                            warmup=5, iters=20,
                        )
                        tf = gemm_tflops(n, n, n, ms)
                    except Exception as e:  # some configs exceed shared mem / registers
                        ms, tf = None, None
                        print(f"  config failed N={n} block=({bm},{bn},{bk}) "
                              f"warps={w} stages={s}: {type(e).__name__}")
                    rows.append({
                        "N": n, "block_m": bm, "block_n": bn, "block_k": bk,
                        "num_warps": w, "num_stages": s,
                        "ms": round(ms, 4) if ms else "FAIL",
                        "tflops": round(tf, 2) if tf else "FAIL",
                    })
                    if tf:
                        print(f"  N={n} block=({bm},{bn},{bk}) warps={w} "
                              f"stages={s}: {tf:.1f} TFLOPS")
    return rows


def record_autotuner_choices():
    """Trigger autotuning per size and record which config the autotuner picked."""
    choices = []
    for n in SIZES:
        a = torch.randn((n, n), device="cuda", dtype=torch.float16)
        b = torch.randn((n, n), device="cuda", dtype=torch.float16)
        matmul(a, b)  # triggers autotune for this key
        best = matmul_kernel.best_config
        chosen_stages = best.num_stages
        hand = HAND_TUNED_STAGES.get(n)
        choices.append({
            "N": n,
            "chosen_block_m": best.kwargs["BLOCK_M"],
            "chosen_block_n": best.kwargs["BLOCK_N"],
            "chosen_block_k": best.kwargs["BLOCK_K"],
            "chosen_num_warps": best.num_warps,
            "chosen_num_stages": chosen_stages,
            "hand_tuned_stages": hand,
            "stages_match": chosen_stages == hand,
        })
        print(f"N={n}: autotuner chose {best} | hand-tuned stages: {hand}")
    return choices


def main():
    print("Device:")
    device_report()

    print("\n=== Full config sweep ===")
    sweep_rows = run_sweep()

    print("\n=== Autotuner choices vs hand-tuned ===")
    choice_rows = record_autotuner_choices()

    os.makedirs("results/data", exist_ok=True)
    for path, rows in ((SWEEP_CSV, sweep_rows), (CHOICES_CSV, choice_rows)):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
