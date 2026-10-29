"""注意力动物园：V100 (SM70) 上的线性/softmax 注意力横向基准。

覆盖：
- GDN (fused Triton, 本项目)  +  GDN (FLA)
- DeltaNet (FLA chunk_delta_rule)  —— 语义上 = GDN with g=0
- GLA (FLA chunk_gla)
- RetNet / retention (FLA chunk_retention)
- torch SDPA MHA（full / causal 两种 mask）

所有 FLA 调用均 scale=1.0（与其内部默认 K^-0.5 区分），单条失败/OOM 记录后继续。
另做一项交叉验证：我们的递推参考（g=0）vs FLA delta_rule，确认框架语义一致。

用法：CUDA_VISIBLE_DEVICES=1 python bench_zoo.py
"""

import json
import time
import traceback

import torch
import triton
import triton.testing as tbench

triton.runtime.driver.active.get_empty_cache_for_benchmark = (
    lambda: torch.empty(int(8e6 // 4), dtype=torch.int, device='cuda'))

from gdn_ref import gdn_recurrent_ref, _l2norm
from gdn_triton import _gdn_fwd_kernel
from bench_gdn import make_inputs, wait_free

H, K, V = 16, 128, 128
SHAPES = [(2, 1024), (2, 4096), (2, 16384)]


def fla_fns():
    fns = {}
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        fns['gdn_fla'] = lambda q, k, v: chunk_gated_delta_rule(
            q=q, k=k, v=v, g=q.new_zeros(q.shape[:-1], dtype=torch.float32),
            beta=(torch.rand(q.shape[:3], device=q.device) * 0.5 + 0.4).float(),
            scale=1.0)
    except Exception:
        traceback.print_exc()
    try:
        from fla.ops.delta_rule import chunk_delta_rule
        fns['deltanet_fla'] = lambda q, k, v: chunk_delta_rule(
            q=q, k=k, v=v,
            beta=(torch.rand(q.shape[:3], device=q.device) * 0.5 + 0.4).float(),
            scale=1.0)
    except Exception:
        traceback.print_exc()
    try:
        from fla.ops.gla import chunk_gla
        fns['gla_fla'] = lambda q, k, v: chunk_gla(
            q=q, k=k, v=v,
            g=(-torch.rand(q.shape, device=q.device) * 0.1).float(),
            scale=1.0)
    except Exception:
        traceback.print_exc()
    try:
        from fla.ops.retention import chunk_retention
        fns['retention_fla'] = lambda q, k, v: chunk_retention(q=q, k=k, v=v)
    except Exception:
        traceback.print_exc()
    return fns


def gdn_ours(q, k, v):
    B, T, _, _ = q.shape
    beta = (torch.rand(B, T, H, device=q.device) * 0.5 + 0.4).float()
    g = (-torch.rand(B, T, H, device=q.device) * 0.1).float()
    o = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    hf = torch.empty(B, H, V, K, dtype=torch.float32, device=q.device)
    _gdn_fwd_kernel.fn[(B * H, V // 32)](
        q, k, v, beta, g, o, hf,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        beta.stride(0), beta.stride(1), g.stride(0), g.stride(1),
        o.stride(0), o.stride(1), o.stride(2), hf.stride(0), hf.stride(1),
        T, H=H, DK=K, DV=V, C=32, BV=32, num_warps=8, num_stages=1)
    return o, hf


def sdpa(q, k, v, causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)


def bench_one(name, fn, q, k, v):
    for attempt in range(2):
        try:
            wait_free(min_gb=2.0)
            torch.cuda.empty_cache()
            ms = tbench.do_bench(lambda: fn(q, k, v), warmup=15, rep=60)
            return round(ms, 4)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt == 1:
                return 'OOM'
        except Exception as e:
            return f"{type(e).__name__}: {str(e)[:120]}"
    return 'OOM'


def cross_check_deltanet():
    """我们的递推参考（g=0）应与 FLA delta_rule 一致。"""
    from fla.ops.delta_rule import chunk_delta_rule
    q, k, v, beta, _ = make_inputs(1, 256, 4, K, V, torch.device('cuda'), seed=7)
    zero_g = torch.zeros_like(beta)
    o_ref, _ = gdn_recurrent_ref(q, k, v, beta, zero_g)
    o_fla = chunk_delta_rule(q=q, k=k, v=v, beta=beta, scale=1.0)[0]
    rel = ((o_ref - o_fla.float()).abs().max() / o_ref.abs().max()).item()
    print(f"[cross-check] delta_rule (FLA) vs 递推参考(g=0): rel err = {rel:.4e}")
    return rel


def main():
    dev = torch.device('cuda')
    z = {}
    try:
        z['cross_check_deltanet_rel'] = cross_check_deltanet()
    except Exception as e:
        z['cross_check_deltanet_rel'] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    runners = {'gdn_ours': gdn_ours}
    runners.update(fla_fns())
    runners['mha_sdpa_full'] = lambda q, k, v: sdpa(q, k, v, causal=False)
    runners['mha_sdpa_causal'] = lambda q, k, v: sdpa(q, k, v, causal=True)

    rows = []
    for (B, T) in SHAPES:
        wait_free(min_gb=2.0)
        q, k, v, _, _ = make_inputs(B, T, H, K, V, dev, seed=5)
        row = {'B': B, 'T': T}
        for name, fn in runners.items():
            row[name] = bench_one(name, fn, q, k, v)
            print(f"  {name}: {row[name]}", flush=True)
        rows.append(row)
        del q, k, v
        torch.cuda.empty_cache()

    z['rows'] = rows
    with open('zoo_results.json', 'w') as f:
        json.dump(z, f, indent=2)
    print(json.dumps(z, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
