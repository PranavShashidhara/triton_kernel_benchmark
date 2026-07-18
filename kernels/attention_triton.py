"""Triton fused attention with online softmax.

Week 6 deliverable: port of the Week 3-4 CUDA fused attention kernel.
Same algorithm (single-pass online softmax, fp16 compute / fp32 accumulate,
optional causal masking) so the CUDA-vs-Triton comparison is algorithm-
for-algorithm, not just "two attention implementations."

Reference for comparison: triton-lang/triton tutorials/06-fused-attention.py
(this is a from-scratch port of *your* kernel, kept intentionally close to
your CUDA structure rather than to the tutorial).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=s)
        for s in [2, 3, 4]
    ]
    + [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=s)
        for s in [2, 3, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    SEQ_LEN, HEAD_DIM: tl.constexpr,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)          # query tile
    pid_bh = tl.program_id(1)         # (batch, head) flattened

    q_ptr += pid_bh * stride_qh
    k_ptr += pid_bh * stride_kh
    v_ptr += pid_bh * stride_vh
    o_ptr += pid_bh * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)

    # Online softmax state — mirrors the CUDA kernel's per-row (m_i, l_i, acc)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Causal: query tile pid_m only attends to K/V tiles up to its own position.
    # Fully-masked tiles are skipped entirely by bounding the loop — this is the
    # same "skip fully-masked tiles" optimization from the Week 4 CUDA kernel.
    if IS_CAUSAL:
        hi = (pid_m + 1) * BLOCK_M
    else:
        hi = SEQ_LEN

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < SEQ_LEN, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale            # (BLOCK_M, BLOCK_N), fp32
        s = tl.where(offs_n[None, :] < SEQ_LEN, s, float("-inf"))
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < SEQ_LEN, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < SEQ_LEN)


def fused_attention(q, k, v, causal: bool = False) -> torch.Tensor:
    """Shapes: (batch, heads, seq, head_dim), fp16. Returns same shape, fp16."""
    assert q.dtype == torch.float16, "fp16 in / fp32 accumulate, matching the CUDA kernel"
    batch, heads, seq_len, head_dim = q.shape
    assert head_dim in (64, 128), "benchmark grid covers head dims 64 and 128"
    o = torch.empty_like(q)
    sm_scale = head_dim ** -0.5
    grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_M"]), batch * heads)
    fused_attention_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seq_len, head_dim,
        sm_scale,
        IS_CAUSAL=causal,
    )
    return o
