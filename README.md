# triton-kernel-comparison

Project 2 of the GPU portfolio: hand-tuned CUDA kernels (from
[`cuda-memory-hierarchy-benchmarks`](../cuda-memory-hierarchy-benchmarks))
ported to Triton, with a systematic characterization of Triton's autotuner
against manual pipeline scheduling.

**Centerpiece finding (Week 7):** _[fill in after the sweep — e.g. "Triton's
autotuner converges to a 3-stage pipeline at N=2048, matching my hand-tuned
choice; at N=512 it picks 2 stages and underperforms my hand-tuned 3-stage
config by X%"]_

## Layout

```
kernels/                    Triton kernel implementations
  matmul_triton.py          autotuned GEMM + fixed-config variant for sweeps
  attention_triton.py       fused attention w/ online softmax + causal masking
bench/                      benchmarking (same methodology as Project 1 harness)
  utils.py                  CUDA-event timing, error metrics, TFLOPS math
  bench_gemm.py             Triton vs cuBLAS (vs Project 1 CUDA ext if present)
  bench_attention.py        Triton vs SDPA vs naive; latency + peak-memory grid
analysis/                   Week 7-8 analysis
  autotuner_sweep.py        full (size x config) sweep -> performance surface
  extract_ptx.py            Triton PTX dump + instruction-count stats
  plots.py                  performance-surface heatmaps + comparison charts
  torch_compile_check.py    custom ops, graph-break check, vs Inductor
results/
  data/                     CSVs from every run (committed — they are the record)
  ptx/                      dumped PTX/SASS
  charts/                   PNGs for this README
main.ipynb                  Colab driver — thin orchestration, no logic
```

## Running

Everything runs from the repo root as modules, on Colab (Pro, A100/H100) or
any CUDA box:

```bash
python -m bench.bench_gemm
python -m bench.bench_attention
python -m analysis.autotuner_sweep      # edit HAND_TUNED_STAGES first
python -m analysis.extract_ptx
python -m analysis.plots
python -m analysis.torch_compile_check
```

Or open `main.ipynb` in Colab and run top to bottom.

**Hardware note:** benchmark numbers in `results/data/` are tagged with the
device via `device_report()`. Comparisons against Project 1 CUDA kernels are
only valid on matching hardware (A100 / sm_80).

## Sweep design

The autotuner sweep (`analysis/autotuner_sweep.py`) does two things the
autotuner alone doesn't:

1. **Times every config**, not just the winner — producing the full
   (size × stages × warps × blocks) performance surface, including configs
   that fail to compile (shared-memory / register overruns are recorded, not
   hidden).
2. **Records the autotuner's per-size choice** and compares its pipeline
   depth against the hand-tuned depth chosen in the Week 5 CUDA multistage
   kernel — the direct "compiler vs manual scheduling" comparison.

## Findings

_[Written per the Week 7 plan: 3-4 findings on register allocation,
pipelining depth, and where the compiler wins/loses vs manual scheduling.
Backed by `results/charts/performance_surface.png` and the PTX stats from
`extract_ptx.py`.]_

1.
2.
3.

## Profiling note

Nsight Compute (`ncu`) needs GPU performance-counter permissions that Colab
VMs typically block. The `ncu` passes from the roadmap run on the RunPod A100
used for Project 1; everything else in this repo runs on Colab.

## Related

- Project 1: `cuda-memory-hierarchy-benchmarks` — the CUDA kernels these ports
  are compared against
- Project 3: `parallelism-ladder` — distributed training/inference comms
  characterization
