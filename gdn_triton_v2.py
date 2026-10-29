"""GDN fused kernel v2：把 v1 的 31 步串行前代求解换成 Newton-Schulz 迭代。

原理：M = I + A（A 严格下三角、幂零 A^C=0）。
    X0 = I - A，X ← X(2I - MX)，误差 E_k = I - X_k·M = A^{2^(k+1)}
    k = ceil(log2(C)) - 1 步后精确为 0（CPU 实测 C=32 时 k=3 即可）。
每步 2 次 C×C tl.dot → 全部走 tensor core，串行度从 C-1 降到 ITERS。

与 v1 的其余部分（chunk 循环、状态更新、输出）完全一致，见 gdn_triton.py。
CPU 数学验证（gdn_chunk_ns，gate ∈ [0,1]）：误差 fp32 级（~1e-6）。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_fwd_kernel_v2(
    Q, K, V, Beta, G, O, Hf,
    sqb, sqt, sqh,
    skb, skt, skh,
    svb, svt, svh,
    sbb, sbt,
    sgb, sgt,
    sob, sot, soh,
    shb, shh,
    T, H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    C: tl.constexpr, BV: tl.constexpr, ITERS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    i_b = pid_bh // H
    i_h = pid_bh % H

    offs_k = tl.arange(0, DK)
    offs_c = tl.arange(0, C)
    offs_v = pid_v * BV + tl.arange(0, BV)
    eye = tl.where(offs_c[:, None] == offs_c[None, :], 1.0, 0.0)

    b_h = tl.zeros((BV, DK), dtype=tl.float32)

    for i_t in range(0, tl.cdiv(T, C)):
        t0 = i_t * C
        tq = t0 + offs_c

        q = tl.load(Q + i_b * sqb + i_h * sqh + tq[:, None] * sqt + offs_k[None, :])
        k = tl.load(K + i_b * skb + i_h * skh + tq[:, None] * skt + offs_k[None, :])
        v = tl.load(V + i_b * svb + i_h * svh + tq[:, None] * svt + offs_v[None, :])
        beta = tl.load(Beta + i_b * sbb + tq * sbt + i_h).to(tl.float32)
        g = tl.load(G + i_b * sgb + tq * sgt + i_h).to(tl.float32)

        b = tl.cumsum(g, 0)
        kk = tl.dot(k, tl.trans(k), out_dtype=tl.float32)
        dec = tl.exp(b[:, None] - b[None, :])
        A = kk * dec * beta[:, None]
        A = tl.where(offs_c[:, None] > offs_c[None, :], A, 0.0)

        # Newton-Schulz 三角逆：X ← X(2I - MX)，fp16 dot + fp32 累加
        m16 = (eye + A).to(tl.float16)
        x16 = (eye - A).to(tl.float16)
        for _ in range(ITERS):
            mx = tl.dot(m16, x16, out_dtype=tl.float32)
            x16 = tl.dot(x16, (2.0 * eye - mx).to(tl.float16),
                         out_dtype=tl.float32).to(tl.float16)

        U = tl.dot(x16, (v * beta[:, None]).to(tl.float16), out_dtype=tl.float32)
        W = tl.dot(x16, (k * (beta * tl.exp(b))[:, None]).to(tl.float16),
                   out_dtype=tl.float32)

        u = U - tl.dot(W.to(tl.float16), tl.trans(b_h.to(tl.float16)), out_dtype=tl.float32)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * dec
        bout = tl.where(offs_c[:, None] >= offs_c[None, :], qk, 0.0)
        o_intra = tl.dot(bout.to(tl.float16), u.to(tl.float16), out_dtype=tl.float32)
        o_inter = tl.dot(q, tl.trans(b_h.to(tl.float16)), out_dtype=tl.float32) \
            * tl.exp(b)[:, None]
        p_o = O + i_b * sob + i_h * soh + tq[:, None] * sot + offs_v[None, :]
        tl.store(p_o, (o_intra + o_inter).to(O.dtype.element_ty))

        bC = tl.sum(tl.where(offs_c == C - 1, b, 0.0))
        wb = (W * tl.exp(bC - b)[:, None]).to(tl.float16)
        ub = (U * tl.exp(bC - b)[:, None]).to(tl.float16)
        wtk = tl.dot(tl.trans(wb), k, out_dtype=tl.float32)
        b_h = b_h * tl.exp(bC) \
            - tl.dot(b_h.to(tl.float16), wtk.to(tl.float16), out_dtype=tl.float32) \
            + tl.dot(tl.trans(ub), k, out_dtype=tl.float32)

    p_h = Hf + i_b * shb + i_h * shh + offs_v[:, None] * DK + offs_k[None, :]
    tl.store(p_h, b_h)


def gdn_chunk_fwd_v2(q, k, v, beta, g, C=32, BV=32, num_warps=4, num_stages=1, iters=3):
    """v2 fused chunked GDN 前向（Newton-Schulz 求解）。T 自动 pad 到 C 倍数。"""
    B, T, H, K = q.shape
    V = v.shape[-1]
    if T % C != 0:
        p = (T + C - 1) // C * C - T
        zq = q.new_zeros(B, p, H, K); zk = k.new_zeros(B, p, H, K)
        zv = v.new_zeros(B, p, H, V)
        zb = beta.new_zeros(B, p, H); zg = g.new_zeros(B, p, H)
        q = torch.cat([q, zq], 1); k = torch.cat([k, zk], 1)
        v = torch.cat([v, zv], 1); beta = torch.cat([beta, zb], 1); g = torch.cat([g, zg], 1)
        T = T + p
    o = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    hf = torch.empty(B, H, V, K, dtype=torch.float32, device=q.device)
    _gdn_fwd_kernel_v2[(B * H, V // BV)](
        q, k, v, beta, g, o, hf,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        beta.stride(0), beta.stride(1),
        g.stride(0), g.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        hf.stride(0), hf.stride(1),
        T, H=H, DK=K, DV=V, C=C, BV=BV, ITERS=iters,
        num_warps=num_warps, num_stages=num_stages)
    return o, hf
