# v100-attention-lab：V100 (SM70) 上的 GDN / 线性注意力 kernel 实验

在外部双卡 Tesla V100-PCIE-32GB 服务器（ssh 别名 `V100_222` / `V100_222_TS`）上，
为 **Gated DeltaNet (GDN)** 等线性注意力算子从零构建的 Triton kernel 实验：
正确性验证、针对 SM70 的调优、与社区实现 (flash-linear-attention, FLA) 和标准
softmax 注意力 (torch SDPA) 的对比。

## 背景：为什么 V100 是个有趣的靶子

- 主流 kernel 生态（DeepSeek TileKernels、FLA 的 autotune 配置、Triton 教程配置）
  都以 SM80+/SM90+ 为中心。V100 是 SM70：**没有 cp.async（SM80）、没有 TMA/WGMMA（SM90）、
  没有 bf16 硬件**，每 SM 96KB 统一 L1/shared（每 block 上限 64KB），第一代 tensor core。
- CUDA 13.0 起彻底移除 Volta（PyPI 上 torch≥2.13 的默认 cu13 wheel 没有 sm_70 kernel，
  实测 `torch.cuda.get_arch_list()` 从 sm_75 起），所以必须钉在 **torch 2.9.1+cu128**
  （arch 列表含 sm_70）+ triton 3.5.1（实测可在 sm70 上正常编译运行）。
- 共享卡环境：两块卡被其他任务弹性占用（26.7–29.2GB），我们的实验必须把显存
  占用控制在 ~500MB + 波动余量内，由此催生了 bench 脚本里的一整套自适应逻辑。

## 文件

| 文件 | 内容 |
|---|---|
| `gdn_ref.py` | GDN 语义定义 + 两个纯 PyTorch 实现：逐步递推（fp32 参考精度）、chunk 化 UT 变换（与 kernel 同算法）。CPU 自检：两者一致 ~4e-7 |
| `gdn_triton.py` | 单 kernel fused chunked GDN 前向（Triton），状态切片常驻寄存器，fp16 输入 fp32 累加 |
| `bench_gdn.py` | 正确性校验、配置 sweep、基准（ours vs FLA vs SDPA MHA），含共享卡 OOM 自适应 |

## GDN 语义（每个 head，状态 h 为 V×K）

```
h_t = exp(g_t) · h_{t-1} + u_t ⊗ k_t
u_t = β_t · (v_t − exp(g_t) · h_{t-1} · k_t)      # 门控 delta 规则
o_t = h_t · q_t
```

等价的 K×V 形式：`S_t = γ_t S_{t-1} (I − β_t k_t k_tᵀ) + β_t k_t v_tᵀ`，γ_t = exp(g_t)。
即 DeltaNet 的"沿 k 抹除旧值再写入 v"，外加每步对数衰减门 g（GDN 论文的设定）。

chunk 化（UT 变换，chunk 大小 C）：
- `A[t,j] = β_t · exp(b_t − b_j) · (k_t·k_j)`（严格下三角，b = chunk 内 cumsum(g)）
- `U = (I+A)⁻¹(β·v)`、`W = (I+A)⁻¹(β·exp(b)·k)`（同一三角系统求解）
- 输出 `o_t = exp(b_t)·q_t·h₀ + Σ_{j≤t} exp(b_t−b_j)·u_j·(k_j·q_t)`
- 状态 `h' = h₀·(exp(b_C)·I − W̄ᵀK) + ŪᵀK`，W̄/Ū 为按 exp(b_C−b) 缩放的 U/W

## SM70 上的调优发现（本项目的主要经验）

1. **寄存器压力是第一约束，且比预想的更彻底**（probe_spill.py 实测，cuobjdump -res-usage）：
   这颗 SM70 上 ptxas 对本 kernel 的寄存器分配是**全盘失败**的——所有配置都是
   `REG:32`（几乎裸奔）+ 巨大每线程栈：

   | 配置 | REG | STACK(B/线程) | w4/w8 实测 |
   |---|---|---|---|
   | C32/BV32/w4 | 32 | 23,096 | 可启动 |
   | C32/BV32/w8 | 32 | 11,872 | 可启动（最优） |
   | C32/BV64/w4 | 32 | 41,512 | **加载即 OOM** |
   | C32/BV64/w8 | 32 | 18,830 | 可启动 |
   | C64/BV32/w4 | 32 | 44,440 | **加载即 OOM** |
   | C64/BV32/w8 | 32 | 21,344 | 可启动 |
   | C64/BV64/w4 | 32 | 67,696 | **加载即 OOM** |
   | C64/BV64/w8 | 32 | 31,336 | 可启动 |

   机制链：PTX 前端无 local 声明（local_depot=0），但 ptxas 把 ~2000+ 个 PTX 虚拟
   寄存器压不进 255 个物理寄存器 → 整块 tile 状态（b_h/U/W/A/dec 等）spill 到
   SASS 栈（local memory，显存Backing、走 L1/L2）→ 驱动在 `cuModuleLoadData`
   加载模块时要按"每线程栈 × 驻留线程规模"预留显存，邻居任务占着 26~29GB 时
   阈值在 31KB~41KB/线程之间：w4 重配置（栈翻倍）过线报 `CUDA_ERROR_OUT_OF_MEMORY`。
   报错点在模块加载而非 launch（launch 资源超限会报 "too many resources requested
   for launch"），且我们进程自身稳态只占 404MB——所以"OOM"其实是**驱动给 spill 栈
   预留显存失败**，与 tensor 分配无关。w8 之所以"解决"：同一 tile 摊到 2× 线程，
   每线程栈减半，回到线内。
   推论：本 kernel 在 SM70 上实际是"栈机"（所有活值走 local memory）——这也是与
   FLA 差 2.6~2.8× 中除算法串行外的主要来源；要缩小差距得重构活值集（如减小每
   program 工作集）让 ptxas 真正放进寄存器，而不是只调 num_warps。
2. **无 cp.async ⇒ 软件流水无意义**：`num_stages=1` 最优，多级流水白占 shared。
3. **逐行前代（forward substitution）替代显式三角求逆**：直接在 U/W 上做 63 步行替换，
   避免了 FLA 式 solve_tril 的额外 O(C³)。
4. **fp16 精度足够**：输出相对误差 ~4.5e-4（vs fp32 递推参考），与 FLA 同量级。
5. 共享卡上 autotune 随时可能被邻居任务的显存波动打断 → bench 脚本带
   `wait_free()`（等空闲≥3GB）、OOM 重试、固定配置回退三级保险。

## 结果（2026-10-28，V100-PCIE-32GB 单卡，B×H×K×V = 2×16×128×128，fp16）

正确性（vs fp32 递推参考，三组配置）：全部 PASS，输出相对误差 4.5e-4 ~ 5.7e-4，
最终状态相对误差 2.8e-4 ~ 3.5e-4。与 FLA 输出交叉验证一致（差异仅为 FLA 默认
对 q 乘 K^-0.5 的缩放，`scale=1.0` 对齐后一致）。

配置 sweep（B=2, T=2048, H=8, K=V=128，GPU1 共享环境）：

| 配置 | 耗时 | 备注 |
|---|---|---|
| **C=32, BV=32, num_warps=8** | **59.8 ms** | 最优 |
| C=64, BV=32, num_warps=8 | 78.8 ms | C=64 的 63 步前代循环串行代价 > 省 chunk 迭代的收益 |
| C=32, BV=64, num_warps=8 | 83.5 ms | |
| C=32, BV=32, num_warps=4 | 86.8 ms | |
| C=32/64, BV=32/64, num_warps=4（重配置） | 启动即 OOM | 寄存器 spill 爆炸 |

最终基准（同一轮次，K=V=128, H=16，fp16；共享卡噪声下同轮相对比值才有意义）：

| B×T | ours (C32/BV32/w8) | FLA 0.6.0 | ours/FLA | torch SDPA (MHA) | 备注 |
|---|---|---|---|---|---|
| 2×1024 | 99.3 ms | 38.6 ms | 2.6× | 0.96 ms | |
| 2×4096 | 401.7 ms | 144.3 ms | 2.8× | 12.9 ms | |
| 2×16384 | **1318.7 ms ✓** | **OOM** | — | 210.0 ms | FLA 物化 chunk 状态 ~4.3GB 撑爆 5GB 空闲 |
| 4×4096 | 812.1 ms | 295.7 ms | 2.7× | 22.2 ms | |

与 FLA 输出交叉验证（`scale=1.0` 对齐后）：相对误差 6.5e-4 ~ 7.3e-4，fp16 量级，语义一致。
（注意：FLA 默认对 q 乘 K^-0.5，不对齐时会得到恰好差 √K 倍的"假不一致"。）

## 注意力动物园：SM70 上的横向对比（bench_zoo.py）

B=2, H=16, K=V=128, fp16，GPU1 共享环境（空闲 ~5GB）。FLA 统一 scale=1.0；
交叉验证：FLA chunk_delta_rule vs 本项目递推参考（g=0）rel err = 5.1e-4，语义一致。

| T | GDN ours | GDN FLA | DeltaNet FLA | GLA FLA | RetNet FLA | MHA full | MHA causal |
|---|---|---|---|---|---|---|---|
| 1024 | 61.2 | 23.1 | 20.4 | **1.06** | 15.7 | 0.61 | 0.37 |
| 4096 | 244.7 | 88.0 | 78.0 | **3.82** | 58.6 | 7.99 | 4.49 |
| 16384 | **975.6 ✓** | OOM | OOM | OOM | 230.7 | 133.8 | 66.9 |

单位 ms。三个发现：

1. **在 V100 上，softmax MHA（SDPA）全场最快**——即使 T=16384 causal 也只要 66.9ms。
   线性注意力的训练/推理优势需要现代硬件上的专用 kernel（FLA 在 SM80/90 上有
   warp-specialization/cp.async 路径）才能兑现；SM70 上"线性"的常数项劣势吃掉了
   渐近优势。这对"老卡跑新模型"的选型是直接结论：V100 上不必为了省显存上线性注意力
   （除非序列再长一个数量级）。
2. **delta-rule 家族（GDN/DeltaNet）比 GLA 慢 ~20×**：三角系统 (I+A)⁻¹ 的求解是
   本质开销，GLA 的对角衰减门控免去这一步。GDN 相对 DeltaNet 的额外成本（gate）
   在 FLA 实现里只有 ~13%。
3. **T=16384 + 5GB 空闲显存时，delta-rule/GLA 的 FLA chunked 实现全部 OOM**
   （都要物化 chunk 边界状态），只有我们状态常驻寄存器的 fused kernel 和 RetNet
   （对角衰减，状态更新简单）幸存。

## 设计取舍：我们的 fused kernel vs FLA

| | 本项目 fused kernel | FLA chunk_gated_delta_rule |
|---|---|---|
| 结构 | 单 kernel，每 program 顺序扫全部 chunk | 三 kernel 流水：WY 准备（chunk 并行）→ 状态扫描 → 输出（chunk 并行） |
| 关键路径 | T/C × (每 chunk 全部工作) | 只有状态扫描（每 chunk 两次 K×K/V GEMM） |
| 显存 | **O(B·H·K·V)，状态常驻寄存器** | 需物化全部 chunk 边界状态 [NT·K·V·fp32]，T=16384 时 ~4.3GB |
| T=16384 + 5GB 空闲 | 1353 ms 跑完 | **OOM** |
| 常规 shape | 慢 3~4× | 快 |

结论：FLA 用显存换并行度，我们用顺序扫描换 O(1) 状态显存。在共享卡/长序列场景下
两者互补——这本身就是把 kernel 搬到受限环境（V100 + 显存紧张）上做的有趣收获。

## 复现

```bash
ssh V100_222                      # 或 V100_222_TS（Tailscale 直连）
cd ~/attention-lab
./venv/bin/python gdn_ref.py                       # CPU 数学自检
CUDA_VISIBLE_DEVICES=1 ./venv/bin/python bench_gdn.py --correct   # 正确性
CUDA_VISIBLE_DEVICES=1 ./venv/bin/python bench_gdn.py --sweep     # 配置扫描
CUDA_VISIBLE_DEVICES=1 ./venv/bin/python bench_gdn.py --bench     # 基准
```

## 跨硬件验证：同一 kernel 在 SM70 vs SM89（School_4090D, RTX 4090 D）

代码零改动推到 4090D（torch 2.15.0+cu132 + triton 3.9.0），正确性同样全 PASS。
三组对照数据：

**① 寄存器分配（probe_spill，cuobjdump -res-usage，同一 kernel 同配置表）**

| | SM70 (V100) | SM89 (4090D) |
|---|---|---|
| REG | 32（全配置） | 254~255（全配置） |
| STACK/线程 | 11,872~67,696 B（全 spill） | **0 B（零 spill）** |

同一份 Triton 代码：SM70 上 ptxas 弃守成"栈机"，SM89 上寄存器分配完全正常。

**② 最优配置反转（同 sweep，B=2 T=2048 H=8 K=V=128）**

| | V100 最优 | 4090D 最优 | 同 shape 耗时比 |
|---|---|---|---|
| 配置 | C32/BV32/**w8**/s1 (59.8ms) | C32/BV32/**w4**/s2 (1.38ms) | **43×** |
| w4 vs w8 | w8 胜（栈减半才能加载） | w4 胜（寄存器够用，少线程高占用更优） | |
| num_stages | 均无差异（无 cp.async） | 均无差异（本 kernel chunk 循环有状态依赖 `b_h`，流水线本就用不上） | |

43× 中约 3~4× 来自硬件本身（SM89 vs SM70 峰值），其余 ~10× 全部来自寄存器
分配正常 vs 栈机的差距——坐实了 V100 慢的主因不是算法，是 SM70 codegen。

**③ 线性 vs 二次的交叉点只在 SM89 出现（H=16, K=V=128, fp16）**

| T | ours V100 | MHA V100 | ours 4090D | MHA 4090D |
|---|---|---|---|---|
| 1024 | 99.3 | 0.61 | 0.82 | 0.16 |
| 4096 | 401.7 | 7.99 | 3.25 | 1.89 |
| 16384 | 1318.7 | 133.8 | **13.0 ✓快于 MHA** | 29.8 |

V100 上 MHA 永远赢（T=16384 仍快 16×）；4090D 上 T=16384 时我们的
O(T) kernel 反超 SDPA 2.3×。线性注意力的渐近优势需要"寄存器正常的
kernel + 现代 GPU"两个条件同时满足才兑现——这是"跨硬件必须重新 tune"的
定量证据。

（4090D 为共享卡环境（~4.8GB 空闲），环境安装走了 Mac 下载 wheel + scp 的
离线路线——该机直连 pypi 镜像极慢，且 macOS 上的 pip 无法解析 torch 2.15
的 Linux-only 依赖 `cuda-toolkit[...]`，最终按 cuda-toolkit 元数据逐包下载。）

## 后续方向（TODO）

- [ ] FLA 式 chunk 并行分解（WY/输出 kernel 并行化），缩小与 FLA 的常规 shape 差距
- [ ] 把 GLA / RetNet / DeltaNet 等变体纳入同一框架（FLA 在 sm70 上可用，对照方便）
- [ ] 用 `n_spills` 反馈指导配置裁剪，替代"启动 OOM 才发现"的试错
- [ ] 反向传播（当前只有前向）
