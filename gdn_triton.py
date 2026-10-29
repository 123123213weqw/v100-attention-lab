"""GDN (Gated DeltaNet) chunk 化前向的 Triton fused kernel（面向 V100 / SM70）。

设计要点（SM70 约束）：
- 无 cp.async（SM80 才有）、无 TMA/WGMMA（SM90+），因此 num_stages 取小值；
- 每个 program 负责 (batch, head, V 的一个 BV 切片)，顺序遍历 chunk，
  状态切片 b_h (BV×K) 全程驻留寄存器；
- 三角求解用逐行前代（forward substitution）代替显式矩阵求逆；
- fp16 输入 + fp32 累加（张量核路径），对数域衰减 exp(b_t - b_j)（j<=t 时 <=1，不溢出）。

语义与 gdn_ref.py 完全一致，见该文件注释。
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    # SM70 实测：w4 的重配置会因寄存器/本地内存压力启动 OOM，重配置一律 w8
    configs=[
        triton.Config({'C': 64, 'BV': 64}, num_warps=8, num_stages=1),
        triton.Config({'C': 64, 'BV': 32}, num_warps=8, num_stages=1),
        triton.Config({'C': 32, 'BV': 64}, num_warps=8, num_stages=1),
        triton.Config({'C': 32, 'BV': 32}, num_warps=8, num_stages=1),
        triton.Config({'C': 32, 'BV': 32}, num_warps=4, num_stages=1),
    ],
    key=['T', 'DK', 'DV'],
)
@triton.jit
def _gdn_fwd_kernel(
    Q, K, V, Beta, G, O, Hf,
    sqb, sqt, sqh,
    skb, skt, skh,
    svb, svt, svh,
    sbb, sbt,          # Beta: [B,T,H]
    sgb, sgt,          # G:    [B,T,H]
    sob, sot, soh,
    shb, shh,          # Hf: [B,H,V,K]，行主序，K 连续
    T, H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    C: tl.constexpr, BV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    i_b = pid_bh // H
    i_h = pid_bh % H

    offs_k = tl.arange(0, DK)
    offs_c = tl.arange(0, C)
    offs_v = pid_v * BV + tl.arange(0, BV)

    # 状态切片（V×K 的 BV 行），fp32 驻留
    b_h = tl.zeros((BV, DK), dtype=tl.float32)

    for i_t in range(0, tl.cdiv(T, C)):
        t0 = i_t * C
        tq = t0 + offs_c

        p_q = Q + i_b * sqb + i_h * sqh + tq[:, None] * sqt + offs_k[None, :]
        q = tl.load(p_q)                                   # (C,K) fp16
        p_k = K + i_b * skb + i_h * skh + tq[:, None] * skt + offs_k[None, :]
        k = tl.load(p_k)                                   # (C,K) fp16
        p_v = V + i_b * svb + i_h * svh + tq[:, None] * svt + offs_v[None, :]
        v = tl.load(p_v)                                   # (C,BV) fp16
        beta = tl.load(Beta + i_b * sbb + tq * sbt + i_h).to(tl.float32)   # (C,)
        g = tl.load(G + i_b * sgb + tq * sgt + i_h).to(tl.float32)         # (C,)

        b = tl.cumsum(g, 0)                                # chunk 内累计对数衰减

        # A[t,j] = beta_t * exp(b_t-b_j) * (k_t·k_j)，严格下三角
        kk = tl.dot(k, tl.trans(k), out_dtype=tl.float32)  # (C,C)
        dec = tl.exp(b[:, None] - b[None, :])              # j<=t 时 <=1
        A = kk * dec * beta[:, None]
        A = tl.where(offs_c[:, None] > offs_c[None, :], A, 0.0)

        # 前代求解 U = (I+A)^{-1}(beta*v)，W = (I+A)^{-1}(beta*exp(b)*k)
        U = v.to(tl.float32) * beta[:, None]
        W = k.to(tl.float32) * (beta * tl.exp(b))[:, None]
        for i in range(1, C):
            mi = (offs_c == i)
            arow = tl.sum(tl.where(mi[:, None], A, 0.0), 0)      # A 第 i 行（仅 j<i 非零）
            cu = tl.sum(arow[:, None] * U, 0)                    # (BV,)
            cw = tl.sum(arow[:, None] * W, 0)                    # (K,)
            U = tl.where(mi[:, None], U - cu, U)
            W = tl.where(mi[:, None], W - cw, W)

        # u = U - h_0 W  →  (C,BV)
        u = U - tl.dot(W.to(tl.float16), tl.trans(b_h).to(tl.float16), out_dtype=tl.float32)

        # chunk 内输出：Bout[t,j] = exp(b_t-b_j)(q_t·k_j)，j<=t（含对角线）
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        bout = qk * dec
        bout = tl.where(offs_c[:, None] >= offs_c[None, :], bout, 0.0)
        o_intra = tl.dot(bout.to(tl.float16), u.to(tl.float16), out_dtype=tl.float32)
        o_inter = tl.dot(q, tl.trans(b_h.to(tl.float16)), out_dtype=tl.float32) \
            * tl.exp(b)[:, None]
        p_o = O + i_b * sob + i_h * soh + tq[:, None] * sot + offs_v[None, :]
        tl.store(p_o, (o_intra + o_inter).to(O.dtype.element_ty))

        # 状态更新：h = h (exp(b_C) I - W̄ᵀK) + ŪᵀK
        bC = tl.sum(tl.where(offs_c == C - 1, b, 0.0))
        wb = (W * tl.exp(bC - b)[:, None]).to(tl.float16)
        ub = (U * tl.exp(bC - b)[:, None]).to(tl.float16)
        wtk = tl.dot(tl.trans(wb), k, out_dtype=tl.float32)      # (K,K)
        b_h = b_h * tl.exp(bC) \
            - tl.dot(b_h.to(tl.float16), wtk.to(tl.float16), out_dtype=tl.float32) \
            + tl.dot(tl.trans(ub), k, out_dtype=tl.float32)

    # 末状态（只写各自 V 切片）
    p_h = Hf + i_b * shb + i_h * shh + offs_v[:, None] * DK + offs_k[None, :]
    tl.store(p_h, b_h)


def gdn_chunk_fwd(q, k, v, beta, g):
    """Triton fused chunked GDN 前向。

    q,k: [B,T,H,K] fp16；v: [B,T,H,V] fp16；beta,g: [B,T,H] fp32。
    返回 o [B,T,H,V] fp16、h_final [B,H,V,K] fp32。T 会被 pad 到 64 的倍数。
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    Cmax = 64
    if T % Cmax != 0:
        qp, kp, vp, bp, gp = _pad(q, k, v, beta, g, Cmax)
        o, hf = gdn_chunk_fwd(qp, kp, vp, bp, gp)
        return o[:, :T], hf
    o = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    hf = torch.empty(B, H, V, K, dtype=torch.float32, device=q.device)
    grid = lambda META: (B * H, triton.cdiv(V, META['BV']))
    _gdn_fwd_kernel[grid](
        q, k, v, beta, g, o, hf,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        beta.stride(0), beta.stride(1),
        g.stride(0), g.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        hf.stride(0), hf.stride(1),
        T, H=H, DK=K, DV=V,
    )
    return o, hf


def _pad(q, k, v, beta, g, C):
    B, T, H, K = q.shape
    V = v.shape[-1]
    Tp = (T + C - 1) // C * C
    p = Tp - T
    dev, dt = q.device, q.dtype
    return (
        torch.cat([q, torch.zeros(B, p, H, K, dtype=dt, device=dev)], 1),
        torch.cat([k, torch.zeros(B, p, H, K, dtype=dt, device=dev)], 1),
        torch.cat([v, torch.zeros(B, p, H, V, dtype=dt, device=dev)], 1),
        torch.cat([beta, torch.zeros(B, p, H, dtype=torch.float32, device=dev)], 1),
        torch.cat([g, torch.zeros(B, p, H, dtype=torch.float32, device=dev)], 1),
    )
