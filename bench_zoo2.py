"""zoo2：FLA 20 个线性注意力变体在 V100 (SM70) 上的横评 + 我们的 v2 首测。

只做基准（各变体语义不同，正确性由 FLA 自身测试套件负责）；
ours v1/v2 单独做正确性验证（vs 递推参考）。

用法：CUDA_VISIBLE_DEVICES=1 python bench_zoo2.py
"""

import json
import traceback

import torch
import triton
import triton.testing as tbench

triton.runtime.driver.active.get_empty_cache_for_benchmark = (
    lambda: torch.empty(int(8e6 // 4), dtype=torch.int, device='cuda'))

from bench_gdn import make_inputs, wait_free, gdn_chunk_fwd_fixed   # noqa: E402
from gdn_triton_v2 import gdn_chunk_fwd_v2                          # noqa: E402
from gdn_ref import gdn_recurrent_ref                               # noqa: E402

B, H, K, V = 2, 16, 128, 128
SHAPES = [1024, 4096]


def neg(T, dev, scale=0.1):
    return (-torch.rand(B, T, H, device=dev).float() * scale)


def pos01(T, dev):
    return torch.rand(B, T, H, device=dev).float()


def mat(T, dev):
    return torch.randn(B, T, H, K, device=dev, dtype=torch.float16)


def import_fn(mod, fn):
    m = __import__(mod, fromlist=[fn])
    return getattr(m, fn)


def call_scale1(fn, **kw):
    """优先 scale=1.0 调用，入口无 scale 参数则去掉。"""
    try:
        return fn(**kw, scale=1.0)
    except TypeError:
        return fn(**kw)


def build_zoo(dev):
    """返回 {名字: (描述, 输入构造函数 fn(T)->dict, 调用包装)}"""
    zoo = {}

    def reg(name, mod, fn, mk):
        zoo[name] = (import_fn(mod, fn), mk)

    reg("kda", "fla.ops.kda", "chunk_kda",
        lambda T, dev: dict(q=make_inputs(B, T, H, K, V, dev)[0], **_qkv(T, dev)))
    # 简化：每个条目直接给完整 kwargs
    zoo.clear()

    def qkvg(T, dev):
        q, k, v, beta, g = make_inputs(B, T, H, K, V, dev, seed=5)
        return dict(q=q, k=k, v=v, g=g, beta=beta)

    zoo["kda"] = (import_fn("fla.ops.kda", "chunk_kda"),
                lambda T, dev: {**_qkv(T, dev), "g": (-torch.rand(B, T, H, K, device=dev) * 0.1).float(), "beta": pos01(T, dev)})
    zoo["gated_delta_product"] = (
        import_fn("fla.ops.gated_delta_product", "chunk_gated_delta_product"),
        lambda T, dev: {**qkvg(T, dev), "num_householder": 2})
    zoo["momentum_delta_rule"] = (
        import_fn("fla.ops.momentum_delta_rule", "chunk_momentum_delta_rule"),
        lambda T, dev: {**_qkv(T, dev), "log_alpha": neg(T, dev), "log_mu": neg(T, dev)})
    zoo["precond_kda"] = (
        import_fn("fla.ops.precond_kda", "chunk_precond_kda"),
        lambda T, dev: {**_qkv(T, dev), "g": (-torch.rand(B, T, H, K, device=dev) * 0.1).float(), "g_atk": neg(T, dev), "beta_atk": pos01(T, dev), "beta": pos01(T, dev)})
    zoo["gdn2"] = (
        import_fn("fla.ops.gdn2", "chunk_gdn2"),
        lambda T, dev: {**_qkv(T, dev), "g": (-torch.rand(B, T, H, K, device=dev) * 0.1).float(), "b": pos01(T, dev), "w": mat(T, dev)})
    zoo["linear_attn"] = (
        import_fn("fla.ops.linear_attn", "chunk_linear_attn"), lambda T, dev: _qkv(T, dev))
    zoo["simple_gla"] = (
        import_fn("fla.ops.simple_gla", "chunk_simple_gla"),
        lambda T, dev: {**_qkv(T, dev), "g": neg(T, dev)})
    zoo["based"] = (
        import_fn("fla.ops.based", "fused_chunk_based"), lambda T, dev: _qkv(T, dev))
    zoo["rebased"] = (
        import_fn("fla.ops.rebased", "parallel_rebased"), lambda T, dev: _qkv(T, dev))
    zoo["hgrn"] = (
        import_fn("fla.ops.hgrn", "chunk_hgrn"),
        lambda T, dev: dict(x=mat(T, dev).reshape(B, T, H * K), g=(-torch.rand(B, T, H * K, device=dev) * 0.1).float()))
    zoo["lightning_attn"] = (
        import_fn("fla.ops.lightning_attn", "chunk_lightning_attn"),
        lambda T, dev: {**_qkv(T, dev), "layer_idx": 0, "num_layers": 32})
    zoo["rwkv6"] = (
        import_fn("fla.ops.rwkv6", "chunk_rwkv6"),
        lambda T, dev: {**_qkv(T, dev), "r": mat(T, dev), "w": (-torch.rand(B, T, H, K, device=dev)).float(), "u": torch.rand(B, T, H, K, device=dev).float()})
    zoo["rwkv7"] = (
        import_fn("fla.ops.rwkv7", "chunk_rwkv7"),
        lambda T, dev: {**_kv(T, dev), "r": mat(T, dev), "w": (-torch.rand(B, T, H, K, device=dev)).float(),
                   "a": mat(T, dev), "b": mat(T, dev)})
    zoo["mesa_net"] = (
        import_fn("fla.ops.mesa_net", "chunk_mesa_net"),
        lambda T, dev: {**qkvg(T, dev), "lamb": torch.randn(H, K, device=dev).float()})
    # gated_oja_rule: SM70 illegal memory access（真 crash，记录进 README，不进基准）
    # based: 特征维度设计上限 16（K=128 不适用）
    # ttt: SM70 CompilationError；log_linear_attn: level_scales 形状不明
    zoo["comba"] = (
        import_fn("fla.ops.comba", "chunk_comba"),
        lambda T, dev: {**_qkv(T, dev), "p": mat(T, dev), "g": neg(T, dev)})
    zoo["gdn_ours_v1"] = "OURS_V1"
    zoo["gdn_ours_v2"] = "OURS_V2"
    zoo["mha_sdpa"] = "MHA"
    return zoo


def _kv(T, dev):
    q, k, v, beta, g = make_inputs(B, T, H, K, V, dev, seed=5)
    return dict(k=k, v=v)

def _qkv(T, dev):
    q, k, v, beta, g = make_inputs(B, T, H, K, V, dev, seed=5)
    return dict(q=q, k=k, v=v)


def main():
    dev = torch.device('cuda')

    # --- ours v2 在 V100 上的首测：正确性 + 与 v1 对比 ---
    print("=== ours v2 on V100 correctness ===", flush=True)
    v2_ok = True
    for gs in (0.05, 0.3):
        q, k, v, beta, g = make_inputs(2, 256, 4, K, V, dev, seed=9, gate_scale=gs)
        o_ref, h_ref = gdn_recurrent_ref(q, k, v, beta, g)
        o2, h2 = gdn_chunk_fwd_v2(q, k, v, beta, g, num_warps=8, iters=2)
        eo = ((o_ref - o2.float()).abs().max() / o_ref.abs().max()).item()
        eh = ((h_ref - h2).abs().max() / h_ref.abs().max()).item()
        v2_ok &= eo < 2e-2 and eh < 2e-2
        print(f"gate={gs}: o_rel={eo:.2e} h_rel={eh:.2e} "
              f"{'PASS' if eo < 2e-2 and eh < 2e-2 else 'FAIL'}", flush=True)

    zoo = build_zoo(dev)
    rows = []
    for T in SHAPES:
        wait_free(2.0)
        row = {"T": T}
        for name, entry in zoo.items():
            if entry == "OURS_V1":
                def run():
                    q, k, v, beta, g = make_inputs(B, T, H, K, V, dev, seed=5)
                    gdn_chunk_fwd_fixed(q, k, v, beta, g, C=32, BV=32, num_warps=8)
            elif entry == "OURS_V2":
                def run():
                    q, k, v, beta, g = make_inputs(B, T, H, K, V, dev, seed=5)
                    gdn_chunk_fwd_v2(q, k, v, beta, g, num_warps=8, iters=2)
            elif entry == "MHA":
                def run():
                    q, k, v, _, _, _ = make_inputs(B, T, H, K, V, dev, seed=5)
                    torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            else:
                fn, mk = entry
                kw = mk(T, dev)

                def run(fn=fn, kw=kw):
                    call_scale1(fn, **kw)
            try:
                run()  # 编译+首跑，单独捕获
                torch.cuda.synchronize()
                ms = tbench.do_bench(run, warmup=10, rep=40)
                row[name] = round(ms, 4)
            except Exception as e:
                row[name] = f"{type(e).__name__}: {str(e)[:60]}"
                print(f"  [{name}] FAIL {type(e).__name__}: {str(e)[:100]}", flush=True)
                torch.cuda.empty_cache()
            print(f"  T={T} {name}: {row[name]}", flush=True)
        rows.append(row)
        torch.cuda.empty_cache()

    with open('zoo2_results.json', 'w') as f:
        json.dump({"v2_correct_on_v100": v2_ok, "rows": rows}, f, indent=2)
    print(json.dumps(rows, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
