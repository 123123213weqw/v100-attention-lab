"""探测 GDN kernel 各配置的寄存器 spill：只编译不 launch，读 PTX local depot。

背景：w4 重配置在 V100 上 cuModuleLoad 阶段报 CUDA OOM。假设是 ptxas spill
产生的巨大 local memory 帧导致驱动加载模块时预留显存失败。本脚本用
JITFunction.warmup（只编译）拿 CompiledKernel，从 .asm['ptx'] 里解析
__local_depot 声明大小，量化各配置的 spill。
"""

import glob
import os
import re
import sys
import torch
from gdn_triton import _gdn_fwd_kernel

B, T, H, K, V = 1, 64, 2, 128, 128
q = torch.randn(B, T, H, K, device="cuda", dtype=torch.float16)
k = torch.randn(B, T, H, K, device="cuda", dtype=torch.float16)
v = torch.randn(B, T, H, V, device="cuda", dtype=torch.float16)
beta = torch.rand(B, T, H, device="cuda").float()
g = (-torch.rand(B, T, H, device="cuda") * 0.1).float()
o = torch.empty_like(v)
hf = torch.empty(B, H, V, K, dtype=torch.float32, device="cuda")

args = (q, k, v, beta, g, o, hf,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        beta.stride(0), beta.stride(1),
        g.stride(0), g.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        hf.stride(0), hf.stride(1), T)

jit_fn = _gdn_fwd_kernel.fn   # 剥掉 autotuner

print(f"{'C':>3} {'BV':>3} {'warps':>5} {'threads':>7}   SASS 资源占用")
for C, BV, w in [(32, 32, 4), (32, 32, 8),
                 (32, 64, 4), (32, 64, 8),
                 (64, 32, 4), (64, 32, 8),
                 (64, 64, 4), (64, 64, 8)]:
    ck = jit_fn.warmup(*args, H=H, DK=K, DV=V, C=C, BV=BV,
                       num_warps=w, num_stages=1, grid=(1,))
    ptx = ck.asm['ptx']
    m = re.search(r'\.local \.align \d+ \.b8 __local_depot(\d+)', ptx)
    depot = int(m.group(1)) if m else 0
    # ptxas 在 SASS 层的 spill：dump cubin 交给 cuobjdump -res-usage
    cubin_path = f"/tmp/probe_c{C}_bv{BV}_w{w}.cubin"
    with open(cubin_path, 'wb') as f:
        f.write(ck.asm['cubin'])
    import subprocess, shutil, glob, sys, os
    site = os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                        'lib', 'python*/site-packages/nvidia/cu13/bin/cuobjdump')
    cands = [shutil.which('cuobjdump'), '/usr/local/cuda/bin/cuobjdump',
             '/usr/local/cuda-12.8/bin/cuobjdump'] + glob.glob(site)
    cuobjdump = next((p for p in cands if p and os.path.exists(p)), None)
    out = subprocess.run([cuobjdump, '-res-usage', cubin_path],
                         capture_output=True, text=True).stdout
    m_reg = re.search(r'REG:(\d+)', out)
    m_loc = re.search(r'LOCAL:(\d+)', out)
    nreg = int(m_reg.group(1)) if m_reg else -1
    nloc = int(m_loc.group(1)) if m_loc else -1   # SASS 实际 local(栈) 字节/线程
    threads = w * 32
    print(f"{C:>3} {BV:>3} {w:>5} {threads:>7}  REG={nreg:<4} LOCAL={nloc:>6}B/th  "
          f"block_local={nloc * threads / 1024:.1f}KB  {'<<< SPILL' if nloc > 0 else ''}")
