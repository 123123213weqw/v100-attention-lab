"""GDN kernel 在 V100 上的正确性校验与基准测试。

用法（在 V100_222 上）：
    CUDA_VISIBLE_DEVICES=1 python bench_gdn.py --correct        # 正确性
    CUDA_VISIBLE_DEVICES=1 python bench_gdn.py --bench          # 基准

对比对象：
- gdn_recurrent_ref（fp32 递推，语义参考）
- flash-linear-attention 的 chunk_gated_delta_rule（社区实现，如环境可用）
- torch SDPA（fp16 MHA，作为标准注意力 O(T^2) 的参照）

共享卡环境（显存被其他任务弹性占用）的适应性：
- 每次测量前等待空闲显存 >= 3GB；OOM 后清缓存、等待、重试一次；
- triton do_bench 的 L2 清洗缓冲从 256MB 减到 8MB（V100 L2 只有 6MB）。
"""

import argparse
import json
import time

import torch
import triton
import triton.testing as tbench

# 共享卡环境：triton do_bench 默认 256MB 清洗缓冲容易 OOM，改 8MB
triton.runtime.driver.active.get_empty_cache_for_benchmark = (
    lambda: torch.empty(int(8e6 // 4), dtype=torch.int, device='cuda'))

from gdn_ref import gdn_recurrent_ref, _l2norm          # noqa: E402
from gdn_triton import gdn_chunk_fwd, _gdn_fwd_kernel   # noqa: E402


def make_inputs(B, T, H, K, V, device, seed=0, gate_scale=0.1):
    torch.manual_seed(seed)
    q = _l2norm(torch.randn(B, T, H, K, device=device)).to(torch.float16)
    k = _l2norm(torch.randn(B, T, H, K, device=device)).to(torch.float16)
    v = torch.randn(B, T, H, V, device=device).to(torch.float16)
    beta = (torch.rand(B, T, H, device=device) * 0.5 + 0.4).float()
    g = (-torch.rand(B, T, H, device=device) * gate_scale).float()
    return q, k, v, beta, g


def gdn_chunk_fwd_fixed(q, k, v, beta, g, C=32, BV=32, num_warps=4, num_stages=1):
    """绕过 autotune，用固定配置直调 kernel。"""
    B, T, H, K = q.shape
    V = v.shape[-1]
    o = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    hf = torch.empty(B, H, V, K, dtype=torch.float32, device=q.device)
    _gdn_fwd_kernel.fn[(B * H, V // BV)](
        q, k, v, beta, g, o, hf,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        beta.stride(0), beta.stride(1),
        g.stride(0), g.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        hf.stride(0), hf.stride(1),
        T, H=H, DK=K, DV=V, C=C, BV=BV, num_warps=num_warps, num_stages=num_stages)
    return o, hf


def wait_free(min_gb=3.0, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        free, _ = torch.cuda.mem_get_info()
        if free >= min_gb * 1e9:
            return True
        time.sleep(5)
    return False


def bench_fn(fn, warmup=25, rep=100, tries=2):
    """带 OOM 重试的 do_bench：失败则清缓存、等显存、重试。"""
    for i in range(tries):
        try:
            return tbench.do_bench(fn, warmup=warmup, rep=rep)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if not wait_free():
                break
    return None


def run_correct(T=256, B=2, H=4, K=128, V=128, gate_scale=0.1):
    q, k, v, beta, g = make_inputs(B, T, H, K, V, torch.device('cuda'),
                                   gate_scale=gate_scale)
    o_ref, h_ref = gdn_recurrent_ref(q, k, v, beta, g)
    o_tri, h_tri = gdn_chunk_fwd_fixed(q, k, v, beta, g)
    o_tri = o_tri.float()
    err_o = (o_ref - o_tri).abs().max().item()
    rel_o = err_o / o_ref.abs().max().item()
    err_h = (h_ref - h_tri).abs().max().item()
    rel_h = err_h / h_ref.abs().max().item()
    ok = rel_o < 2e-2 and rel_h < 2e-2
    print(f"[correct] T={T} B={B} H={H} K={K} V={V} gate={gate_scale}")
    print(f"  output: max abs err={err_o:.4e}  rel={rel_o:.4e}")
    print(f"  state : max abs err={err_h:.4e}  rel={rel_h:.4e}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def try_import_fla():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        return chunk_gated_delta_rule
    except Exception as e:
        print(f"[fla] 不可用: {type(e).__name__}: {e}")
        return None


def mha_sdpa(q, k, v):
    # 标准 softmax 注意力（无因果 mask，同形状，仅作成本参照）
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))


def run_sweep():
    """逐配置直接计时，选出能启动且最快的配置写入 sweep_results.json。"""
    import itertools
    B, T, H, K, V = 2, 2048, 8, 128, 128
    q, k, v, beta, g = make_inputs(B, T, H, K, V, torch.device('cuda'), seed=3)
    out = []
    for C, BV, w, ns in itertools.product((32, 64), (32, 64), (4, 8), (1, 2, 3)):
        try:
            wait_free(min_gb=1.5)
            torch.cuda.empty_cache()
            ms = tbench.do_bench(
                lambda: gdn_chunk_fwd_fixed(q, k, v, beta, g, C=C, BV=BV, num_warps=w, num_stages=ns),
                warmup=10, rep=50)
            out.append({'C': C, 'BV': BV, 'num_warps': w, 'num_stages': ns, 'ms': round(ms, 4)})
            print(out[-1], flush=True)
        except Exception as e:
            print(f"C={C} BV={BV} w={w}: {type(e).__name__}")
    out.sort(key=lambda r: r['ms'])
    with open('sweep_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("best:", out[0])


def run_bench():
    fla_fn = try_import_fla()
    results = []
    H, K, V = 16, 128, 128
    for (B, T) in [(2, 1024), (2, 4096), (2, 16384), (4, 4096)]:
        wait_free()
        q, k, v, beta, g = make_inputs(B, T, H, K, V, torch.device('cuda'), seed=1)
        row = {'B': B, 'T': T, 'H': H, 'K': K, 'V': V}

        try:
            ms = bench_fn(lambda: gdn_chunk_fwd(q, k, v, beta, g))
            row['ours_mode'] = 'autotune'
        except Exception as e:
            print(f"[ours] autotune 失败({type(e).__name__})，回退固定配置")
            best = {}
            try:
                best = json.load(open('sweep_results.json'))[0]
            except Exception:
                pass
            ms = bench_fn(lambda: gdn_chunk_fwd_fixed(
                q, k, v, beta, g,
                C=best.get('C', 32), BV=best.get('BV', 32),
                num_warps=best.get('num_warps', 4),
                num_stages=best.get('num_stages', 1)))
            row['ours_mode'] = f"fixed(C{best.get('C',32)},BV{best.get('BV',32)},w{best.get('num_warps',4)},s{best.get('num_stages',1)})" 
        row['ours_ms'] = round(ms, 4) if ms else None

        if fla_fn is not None:
            try:
                # FLA 签名是 (q, k, v, g, beta)，务必用关键字传参
                ms_fla = bench_fn(lambda: fla_fn(q=q, k=k, v=v, g=g, beta=beta, scale=1.0))
                row['fla_ms'] = round(ms_fla, 4) if ms_fla else None
                o_fla = fla_fn(q=q, k=k, v=v, g=g, beta=beta, scale=1.0)[0]
                o_ours = gdn_chunk_fwd_fixed(q, k, v, beta, g)[0]
                d = (o_fla.float() - o_ours.float()).abs()
                row['ours_vs_fla_rel'] = round(
                    (d.max() / o_fla.float().abs().max()).item(), 6)
            except Exception as e:
                row['fla_ms'] = None
                row['fla_error'] = f"{type(e).__name__}: {str(e)[:150]}"

        try:
            ms_mha = bench_fn(lambda: mha_sdpa(q, k, v))
            row['sdpa_mha_ms'] = round(ms_mha, 4) if ms_mha else None
        except Exception as e:
            row['sdpa_mha_ms'] = None
            row['sdpa_error'] = f"{type(e).__name__}: {str(e)[:150]}"

        results.append(row)
        print(json.dumps(row, ensure_ascii=False))
        del q, k, v, beta, g
        torch.cuda.empty_cache()

    with open('bench_results.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print("已写入 bench_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--correct', action='store_true')
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    args = ap.parse_args()
    if args.correct:
        allok = True
        for gs in (0.05, 0.3):
            allok &= run_correct(gate_scale=gs)
        allok &= run_correct(T=512, K=64, V=64, gate_scale=0.1)
        print("全部正确性:", "PASS" if allok else "FAIL")
    if args.sweep:
        run_sweep()
    if args.bench:
        run_bench()
    if not (args.correct or args.bench):
        ap.print_help()
