import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

from typing import Tuple


# Torch reference implementation
# q, k, v: [B, H, N, D]
def torch_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [batch, heads, seq_len, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v must have same shape, got {q.shape}, {k.shape}, {v.shape}")

    _, _, n, d = q.shape
    scale = 1.0 / math.sqrt(d)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    causal_mask = torch.tril(
        torch.ones((n, n), device=q.device, dtype=torch.bool)
    )
    scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.float())
    return out.to(q.dtype)

# Forward kernel
@triton.jit
def _flash_attn_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
                           stride_qb, stride_qh, stride_qn, stride_qd,
                           stride_kb, stride_kh, stride_kn, stride_kd,
                           stride_vb, stride_vh, stride_vn, stride_vd,
                           stride_ob, stride_oh, stride_on, stride_od,
                           n_heads: tl.constexpr, seq_len: tl.constexpr,
                           head_dim: tl.constexpr, scale: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    off_b = pid_bh // n_heads
    off_h = pid_bh - off_b * n_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_d = offs_d < head_dim
    mask_m = offs_m < seq_len

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh

    q = tl.load(
        q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # Для causal attention блоки K/V, начинающиеся правее max(offs_m), полностью замаскированы.
    # Их не считаем.
    # max row index inside this Q-block is:
    #   pid_m * BLOCK_M + BLOCK_M - 1
    # поэтому K-блоки с start_n > этот индекс полностью запрещены.
    max_visible_col = (pid_m + 1) * BLOCK_M

    for start_n in range(0, seq_len, BLOCK_N):
        if start_n < max_visible_col:
            cur_n = start_n + offs_n
            mask_n = cur_n < seq_len

            k = tl.load(
                k_base + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )

            v = tl.load(
                v_base + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )

            scores = tl.dot(q, tl.trans(k)) * scale

            # causal mask: разрешены только cur_n <= offs_m
            causal_mask = cur_n[None, :] <= offs_m[:, None]

            scores = tl.where(
                mask_m[:, None] & mask_n[None, :] & causal_mask,
                scores,
                -float("inf"),
            )

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)

            # Для невалидных строк последнего блока избегаем -inf - -inf.
            m_new = tl.where(mask_m, m_new, 0.0)

            p = tl.exp(scores - m_new[:, None])
            p = tl.where(mask_m[:, None], p, 0.0)

            alpha = tl.exp(m_i - m_new)
            alpha = tl.where(mask_m, alpha, 0.0)

            l_new = alpha * l_i + tl.sum(p, axis=1)

            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v.to(tl.float32))

            m_i = m_new
            l_i = l_new

    out = acc / l_i[:, None]

    tl.store(
        o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
        out,
        mask=mask_m[:, None] & mask_d[None, :],
    )

    # lse = logsumexp(scores) = m + log(l)
    # shape lse: [B, H, N], contiguous
    tl.store(
        lse_ptr + (off_b * n_heads + off_h) * seq_len + offs_m,
        m_i + tl.log(l_i),
        mask=mask_m,
    )


def _flash_attention_forward( q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                              block_m: int = 64, block_n: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [B, H, N, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v must have same shape, got {q.shape}, {k.shape}, {v.shape}")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, v must be CUDA tensors")

    batch, n_heads, seq_len, head_dim = q.shape

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)
    lse = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)

    block_d = triton.next_power_of_2(head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    grid = (triton.cdiv(seq_len, block_m), batch * n_heads)

    _flash_attn_fwd_kernel[grid](q, k, v, out, lse,
                                 q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                                 k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                                 v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                                 out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                                 n_heads=n_heads, seq_len=seq_len, head_dim=head_dim, scale=scale,
                                 BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d, num_warps=4)
    return out, lse

# Backward kernel
@triton.jit
def _flash_attn_bwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, lse_ptr, dq_ptr, dk_ptr, dv_ptr,
                           stride_qb, stride_qh, stride_qn, stride_qd,
                           stride_kb, stride_kh, stride_kn, stride_kd,
                           stride_vb, stride_vh, stride_vn, stride_vd,
                           stride_ob, stride_oh, stride_on, stride_od,
                           stride_dob, stride_doh, stride_don, stride_dod,
                           stride_dqb, stride_dqh, stride_dqn, stride_dqd,
                           stride_dkb, stride_dkh, stride_dkn, stride_dkd,
                           stride_dvb, stride_dvh, stride_dvn, stride_dvd,
                           n_heads: tl.constexpr, seq_len: tl.constexpr,
                           head_dim: tl.constexpr, scale: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    off_b = pid_bh // n_heads
    off_h = pid_bh - off_b * n_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < seq_len
    mask_d = offs_d < head_dim

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh

    dq_base = dq_ptr + off_b * stride_dqb + off_h * stride_dqh
    dk_base = dk_ptr + off_b * stride_dkb + off_h * stride_dkh
    dv_base = dv_ptr + off_b * stride_dvb + off_h * stride_dvh

    q = tl.load(
        q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    o = tl.load(
        o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    do = tl.load(
        do_base + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    lse = tl.load(
        lse_ptr + (off_b * n_heads + off_h) * seq_len + offs_m,
        mask=mask_m,
        other=0.0,
    )

    # D_i = sum_c dO_ic * O_ic
    d_i = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    dq_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    max_visible_col = (pid_m + 1) * BLOCK_M

    for start_n in range(0, seq_len, BLOCK_N):
        if start_n < max_visible_col:
            cur_n = start_n + offs_n
            mask_n = cur_n < seq_len

            k = tl.load(
                k_base + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )

            v = tl.load(
                v_base + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )

            scores = tl.dot(q, tl.trans(k)) * scale

            causal_mask = cur_n[None, :] <= offs_m[:, None]

            scores = tl.where(
                mask_m[:, None] & mask_n[None, :] & causal_mask,
                scores,
                -float("inf"),
            )

            # P = softmax(scores)
            p = tl.exp(scores - lse[:, None])
            p = tl.where(mask_m[:, None] & mask_n[None, :] & causal_mask, p, 0.0)

            # dV += P^T @ dO
            dv = tl.dot(
                tl.trans(p.to(tl.float32)),
                do.to(tl.float32),
            )

            # dP = dO @ V^T
            dp = tl.dot(
                do.to(tl.float32),
                tl.trans(v.to(tl.float32)),
            )

            # dS = P * (dP - D)
            ds = p * (dp - d_i[:, None])
            ds = tl.where(mask_m[:, None] & mask_n[None, :] & causal_mask, ds, 0.0)

            # dQ += dS @ K * scale
            dq_acc += tl.dot(
                ds.to(tl.float32),
                k.to(tl.float32),
            ) * scale

            # dK += dS^T @ Q * scale
            dk = tl.dot(
                tl.trans(ds.to(tl.float32)),
                q.to(tl.float32),
            ) * scale

            tl.atomic_add(
                dv_base + cur_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd,
                dv,
                mask=mask_n[:, None] & mask_d[None, :],
                sem="relaxed",
            )

            tl.atomic_add(
                dk_base + cur_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
                dk,
                mask=mask_n[:, None] & mask_d[None, :],
                sem="relaxed",
            )

    tl.store(
        dq_base + offs_m[:, None] * stride_dqn + offs_d[None, :] * stride_dqd,
        dq_acc,
        mask=mask_m[:, None] & mask_d[None, :],
    )


def _flash_attention_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                              out: torch.Tensor, grad_out: torch.Tensor, lse: torch.Tensor,
                              block_m: int = 64, block_n: int = 64
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, n_heads, seq_len, head_dim = q.shape

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = out.contiguous()
    grad_out = grad_out.contiguous()

    dq = torch.empty_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    block_d = triton.next_power_of_2(head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    grid = (triton.cdiv(seq_len, block_m), batch * n_heads)

    _flash_attn_bwd_kernel[grid]( q, k, v, out, grad_out, lse, dq, dk, dv,
                                  q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                                  k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                                  v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                                  out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                                  grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
                                  dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                                  dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                                  dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                                  n_heads=n_heads, seq_len=seq_len, head_dim=head_dim, scale=scale,
                                  BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d, num_warps=4)

    return dq, dk, dv

# Autograd Function and Module
class FlashCausalAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                block_m: int = 64, block_n: int = 64) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out, lse = _flash_attention_forward(q, k, v, block_m=block_m, block_n=block_n)

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.block_m = block_m
        ctx.block_n = block_n

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors

        dq, dk, dv = _flash_attention_backward(q, k, v, out, grad_out, lse, block_m=ctx.block_m, block_n=ctx.block_n)

        return dq, dk, dv, None, None


def flash_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    return FlashCausalAttentionFunction.apply(q, k, v, block_m, block_n)


class FlashCausalAttention(nn.Module):
    def __init__(self, block_m: int = 64, block_n: int = 64):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return flash_causal_attention(q, k, v, block_m=self.block_m, block_n=self.block_n)