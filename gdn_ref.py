"""GDN (Gated DeltaNet) 参考实现。

语义（每个 head，状态 h 为 V×K 矩阵，k/q 为 K 维行向量，v 为 V 维行向量）：

    h_t = exp(g_t) * h_{t-1} + u_t ⊗ k_t
    u_t = beta_t * (v_t - exp(g_t) * h_{t-1} · k_t)
    o_t = h_t · q_t

等价形式（K×V 状态 S）：S_t = γ_t S_{t-1} (I - β_t k_t k_tᵀ) + β_t k_t v_tᵀ。

输入：
    q, k: [B, T, H, K]   v: [B, T, H, V]
    beta: [B, T, H]，取值 (0, 1)
    g:    [B, T, H]，对数衰减门控（<= 0），γ_t = exp(g_t)

提供两个实现：
    gdn_recurrent_ref  —— 逐步递推，定义级精度（fp32/fp64）
    gdn_chunk_ref      —— chunk 化 UT 变换（与 Triton kernel 相同算法），用于先在 CPU 上验证数学
"""

import torch


def _l2norm(x, eps=1e-6):
    return x / torch.sqrt(x.float().pow(2).sum(-1, keepdim=True) + eps).to(x.dtype)


def gdn_recurrent_ref(q, k, v, beta, g, initial_state=None, dtype=torch.float32):
    """逐步递推参考实现。返回 o [B,T,H,V]（dtype 精度）、h_final [B,H,V,K]。"""
    B, T, H, K = q.shape
    V = v.shape[-1]
    q = q.to(dtype); k = k.to(dtype); v = v.to(dtype)
    beta = beta.to(dtype); g = g.to(dtype)
    if initial_state is not None:
        h = initial_state.to(dtype).clone()
    else:
        h = torch.zeros(B, H, V, K, dtype=dtype, device=q.device)
    o = torch.empty(B, T, H, V, dtype=dtype, device=q.device)
    for t in range(T):
        ht = torch.exp(g[:, t]).to(dtype)[:, :, None, None] * h          # 衰减
        hk = torch.einsum('bhvk,bhk->bhv', ht, k[:, t])                  # 状态沿 k 的投影
        u = beta[:, t, :, None] * (v[:, t] - hk)                          # 残差加权
        h = ht + u[:, :, :, None] * k[:, t][:, :, None, :]                # 秩 1 更新
        o[:, t] = torch.einsum('bhvk,bhk->bhv', h, q[:, t])
    return o, h


def gdn_chunk_ref(q, k, v, beta, g, C=64, initial_state=None):
    """chunk 化参考实现（UT 变换），fp32。要求 T 是 C 的倍数。

    每个 chunk 内：
        b_t = cumsum(g)                     # chunk 内累计衰减
        A[t,j] = beta_t * exp(b_t - b_j) * (k_t·k_j)，严格下三角
        U = (I + A)^{-1} (beta * v)          # chunk 内值残差
        W = (I + A)^{-1} (beta * exp(b) * k) # 状态修正
        u_t = U_t - h_0 W_t                  # 真实更新量
        o_t = exp(b_t) q_t·h_0 + Σ_{j<=t} exp(b_t-b_j) u_j (k_j·q_t)
        h_next = h_0 (exp(b_C) I - W̄ᵀ K) + Ūᵀ K，其中 W̄_t = W_t exp(b_C-b_t)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert T % C == 0, "T 需为 C 的倍数（测试输入自行 pad）"
    q = q.float(); k = k.float(); v = v.float()
    beta = beta.float(); g = g.float()
    h = initial_state.float().clone() if initial_state is not None else \
        torch.zeros(B, H, V, K, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)
    eye = torch.eye(C, dtype=torch.float32, device=q.device)
    for c0 in range(0, T, C):
        # 注意：输入是 [B,T,H,*]，沿 T 切片后要转到 [B,H,C,*]
        qc = q[:, c0:c0+C].transpose(1, 2)      # [B,H,C,K]
        kc = k[:, c0:c0+C].transpose(1, 2)
        vc = v[:, c0:c0+C].transpose(1, 2)      # [B,H,C,V]
        bc = beta[:, c0:c0+C].transpose(1, 2)   # [B,H,C]
        gc = g[:, c0:c0+C].transpose(1, 2)
        b = gc.cumsum(-1)       # [B,H,C]
        kk = kc @ kc.transpose(-1, -2)                    # [B,H,C,C] [t,j]=k_t·k_j
        dec = torch.exp(b[..., :, None] - b[..., None, :])
        A = kk * dec * bc[..., :, None]
        A = torch.tril(A, -1)
        M = eye + A                                       # 单位下三角
        rhsU = vc * bc[..., None]
        rhsW = kc * (bc * torch.exp(b))[..., None]
        U = torch.linalg.solve_triangular(M, rhsU, upper=False, unitriangular=True)
        W = torch.linalg.solve_triangular(M, rhsW, upper=False, unitriangular=True)
        u = U - torch.einsum('bhck,bhvk->bhcv', W, h)     # u_t = U_t - h_0 W_t
        qk = qc @ kc.transpose(-1, -2) * dec              # [t,j]=exp(b_t-b_j) q_t·k_j
        Bout = torch.tril(qk)                             # 含对角线 j<=t
        oc = Bout @ u + \
            torch.einsum('bhck,bhvk->bhcv', qc * torch.exp(b)[..., None], h)
        o[:, c0:c0+C] = oc.transpose(1, 2)              # 转回 [B,C,H,V]
        # 状态更新
        bC = b[..., -1:]                                  # [B,H,1]
        Wb = W * torch.exp(bC - b)[..., None]             # W̄
        Ub = U * torch.exp(bC - b)[..., None]             # Ū
        WtK = torch.einsum('bhck,bhcl->bhkl', Wb, kc)     # [B,H,K,K]
        h = torch.einsum('bhvk,bhkl->bhvl', h, torch.exp(bC)[..., None] * torch.eye(K, device=q.device) - WtK) \
            + torch.einsum('bhcv,bhck->bhvk', Ub, kc)
    return o, h


def pad_to_multiple(q, k, v, beta, g, C):
    """把 T pad 到 C 的倍数：pad 步 q=0（输出 0）、beta=0（状态不变）、g=0。"""
    B, T, H, K = q.shape
    V = v.shape[-1]
    Tp = (T + C - 1) // C * C
    if Tp == T:
        return q, k, v, beta, g
    pad = Tp - T
    zq = q.new_zeros(B, pad, H, K); zk = k.new_zeros(B, pad, H, K)
    zv = v.new_zeros(B, pad, H, V)
    zb = beta.new_zeros(B, pad, H); zg = g.new_zeros(B, pad, H)
    return (torch.cat([q, zq], 1), torch.cat([k, zk], 1), torch.cat([v, zv], 1),
            torch.cat([beta, zb], 1), torch.cat([g, zg], 1))


if __name__ == "__main__":
    # CPU 自检：chunk 化实现必须与递推参考一致（fp32 容差）
    torch.manual_seed(0)
    B, T, H, K, V, C = 2, 128, 3, 64, 64, 32
    q = _l2norm(torch.randn(B, T, H, K))
    k = _l2norm(torch.randn(B, T, H, K))
    v = torch.randn(B, T, H, V)
    beta = torch.rand(B, T, H) * 0.5 + 0.4
    g = -torch.rand(B, T, H) * 0.1
    o1, h1 = gdn_recurrent_ref(q, k, v, beta, g)
    o2, h2 = gdn_chunk_ref(q, k, v, beta, g, C=C)
    print("chunked vs recurrent: o max err =", (o1 - o2).abs().max().item())
    print("chunked vs recurrent: h max err =", (h1 - h2).abs().max().item())
    assert (o1 - o2).abs().max().item() < 1e-3, "chunk 化实现与递推参考不一致"
    assert (h1 - h2).abs().max().item() < 1e-3, "chunk 化实现与递推参考不一致"
    print("OK: gdn_chunk_ref 与 gdn_recurrent_ref 数学一致")
