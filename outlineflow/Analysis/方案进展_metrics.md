# OutlineFlow 方案进展 & Metrics 跟踪

> 单一事实源：记录每次改动的「方案 → 对应指标」。所有分数为官方 InceptionV3 (torchmetrics) + prdc(k=5)，`real-vs-real sanity ≈ (FID 0, D 1, C 1)` 已校验，held 集对齐。
> 环境：AMD MI300X + ROCm，`/root/kaiyingwu/rocm-venv`（详见 memory）。评估口径：`sample_eval.py --inception --n_eval 500`。

---

## 1. 基线（jiahua, commit `ad24843`）

无 cross-attention、无数量条件、mandatory gap-fill 解码器。1.33M params。

| group | FID ↓ | Density ↑ | Coverage ↑ |
|---|---|---|---|
| **unit_id**（~9 房/户） | 129.2 | 0.044 | 0.035 |
| **plan_id**（~37 房/层，brief 官方口径） | 190.5 | 0.052 | 0.060 |

---

## 2. 改进方案（当前工作区，未提交）

在 unit_id 上开发验证。模型 1.68M params。四组改动：

| 代号 | 改动 | 文件 | 动机 |
|---|---|---|---|
| **P0-A** | outline 编码升级：pooled 向量 → **(全局向量 + 点级特征)**；边界点坐标加 2D Fourier 编码；每个 DiT block 增加**零门控 cross-attention**（room token=Q，边界点=K/V） | `model.py` | 单一池化向量是 Density 的结构性天花板；token 需「看见」自己贴哪段边界才能落到真实流形 |
| **P0-B** | **数量条件**：`outline_cond=(log面积, log周长)` → zero-init `cond_embed` 注入 AdaLN 条件 | `model.py`, `params.py`, `cfg.py` | real 房间数 std≫gen，需用轮廓几何控制房间数 |
| **P1-C** | **Classifier-Free Guidance**：训练 `--cfg_drop 0.1`，采样 `v=v_u+s·(v_c−v_u)` | `flow.py`, `train.py`, `sample_eval.py` | 提升轮廓保真 → Density/Coverage |
| **Stage-0** | **capped gap-fill**（leftover 面积 > 15% 不并入，避免巨型 blob）+ presence 阈值硬下界 −0.5；padding slot type 填 −1 | `postprocess.py`, `cfg.py`, `params.py` | gap-fill 制造的大色块是最 off-manifold 的伪影，直接压低 Density |

训练：unit_id 全量数据，15k steps，batch 256，cfg_drop 0.1，MI300X ~6.5 min。ckpt：`outputs_unit/`。

---

## 3. 结果：改进 vs 基线（unit_id）

### 3.1 Guidance 扫描（同一 ckpt，仅改采样，免重训）

| guidance | FID ↓ | Density ↑ | Coverage ↑ | gen 房间数 |
|---|---|---|---|---|
| 基线 | 129.2 | 0.044 | 0.035 | — |
| 1.0 | 112.7 | 0.078 | 0.098 | 8.46±2.06 |
| **1.5** ⭐ | **112.2** | 0.084 | **0.124** | 8.47±2.32 |
| 2.0 | 113.8 | **0.089** | 0.104 | 8.48±2.45 |
| 2.5 | 115.1 | 0.079 | 0.092 | 8.48±2.49 |
| 3.0 | 114.7 | 0.088 | 0.100 | 8.34±2.59 |

**最佳工作点 g=1.5**（FID/Coverage 最优；Density 在 g=2.0 略高但 FID 退化）。knee 在 1.5–2.0，≥2.5 全面退化。

### 3.2 当前最佳 vs 基线（g=1.5）

| 指标 | 基线 | 改进@g1.5 | 变化 |
|---|---|---|---|
| **Density** ↑ | 0.044 | **0.084** | **+91%** |
| **Coverage** ↑ | 0.035 | **0.124** | **+254%** |
| **FID** ↓ | 129.2 | **112.2** | **−13%** |
| interior coverage | ~0.5 | **0.85** | 房间填满轮廓 |

**结论**：outline 编码改进（P0-A 的 cross-attn）让房间贴合边界（interior coverage 0.5→0.85）；数量条件让房间数基本对齐；CFG 再补 Coverage。三项目标（改进编码、提升 Density）均达成。

### 3.3 Tier 1/2 消融 + 调参结果（unit_id, 15k steps, eval g=1.5 除非注明）

| 变体 | FID ↓ | Density ↑ | Coverage ↑ | gen 房间数 | 解读 |
|---|---|---|---|---|---|
| **full（参照）** | **112.2** | 0.084 | **0.124** | 8.47 | 最优 FID & Coverage |
| E1 − cross-attn | 114.3 | 0.086 | 0.092 | 7.99 | Coverage/房间数明显掉 → **P0-A 净正**（贡献在填充/Coverage 非裸 Density） |
| E2 − 数量条件 | 116.2 | 0.087 | 0.108 | **8.97** | FID/Coverage 变差，但房间数反更近 9.39（cond 与 calib 有交互） |
| E3 − CFG 训练 (g1.0) | 116.2 | 0.080 | 0.104 | 8.39 | vs full@g1.0=112.7 → **CFG 训练净正** |
| E4 w_presence 4 | 114.5 | 0.088 | 0.110 | 8.33 | 仍 clamp，房间数没回升 → **加权救不了欠分离（需结构改动）** |
| E5 放大 9.6M | 126.0 | 0.068 | 0.098 | 8.48 | **明显变差** → 当前数据/步数下容量非瓶颈，单纯放大会退化 |
| E6 Fourier 32 | 115.9 | **0.091** | 0.114 | 8.49 | Density 最高但 FID 退化（频带↑ 拿 Density 换 FID） |

**Tier 1/2 关键结论：**
1. **现有 full 配方 @g1.5 仍是综合最优**（FID 112.2、Coverage 0.124 均第一）；无单变量改动能在主指标上超越它。
2. **cross-attn / 数量条件 / CFG 三者皆净正**（去掉任一 FID 都变差）→ 消融验证每块都值得保留。
3. **w_presence 加权救不了 presence 欠分离**（E4 仍 clamp、房间数没回升）→ 问题是**结构性**的，必须上 §7-C（presence 独立头 / 集合匹配），而非调 loss 权重。
4. **盲目放大模型会退化**（E5 9.6M FID 126）→ 容量不是当前瓶颈；提升要靠生成范式（§7）与结构（§7-C），不是堆参数。
5. freqs32 用 FID 换 Density；如最终只看 Density 可选，但综合不划算。

> 注：§6 的 E7（30k steps）未纳入本批脚本，单独按需跑（验证是否欠拟合）。

---

## 4. 已知问题 / 待优化

1. **presence 欠分离**（最主要）：校准想把阈值压到 −0.99 才凑够数量，被 floor −0.5 钳住；gen 房间数 8.47 < real 9.39，且方差偏小（±2.3 vs ±2.87）。presence 通道对「真实槽 vs padding 槽」区分不够干净。
2. **仅在 unit_id 验证**，尚未迁移 plan_id（官方口径，~37 房，更难）。
3. **FID 仍偏高**（112），离真实分布还有距离，可能受模型容量 / 训练步数 / 解码器（轴对齐矩形）限制。

---

## 5. 实验日志（按时间）

| 日期 | 实验 | group | steps | 关键结果 |
|---|---|---|---|---|
| 2026-06-27 | 全量改进模型 | unit_id | 15k | g1.0 FID112.7/D0.078/C0.098；g1.5 FID112.2/D0.084/C0.124 |
| 2026-06-28 | guidance 扫描 | unit_id | — | knee@1.5–2.0；g2.0 D 峰值 0.089 |
| 2026-06-28 | Tier1/2 消融+调参 E1-E6 | unit_id | 15k | full 仍最优；3 块改动皆净正；w_pres4/放大/freqs32 均未超越；presence 欠分离需结构改动（见 §3.3） |

---

## 6. 优化实验设计（待执行）

原则：**单变量**改动便于归因；unit_id 上快迭代（~7 min/run），赢家再迁 plan_id。统一评估口径 §0，固定 g=1.5（除非该变体改变了 CFG 语义）。已加 CLI 开关：`--w_presence / --d_model / --n_layers / --n_cond / --fourier_freqs / --no_cross_attn`，每个变体一条命令。

### Tier 1 — 归因消融（验证每块改动的价值，各 ~7 min）

| # | 变体 | 命令关键 flag | 验证假设 |
|---|---|---|---|
| E1 | − cross-attn | `--no_cross_attn` | P0-A 对填充/Density 的真实贡献 |
| E2 | − 数量条件 | `--n_cond 0` | P0-B 对房间数对齐的作用 |
| E3 | − CFG 训练 | `--cfg_drop 0`（eval g=1.0） | CFG-trained field 是否是收益来源 |

### Tier 2 — 快速调参（同架构，各 ~7 min）

| # | 变体 | flag | 假设 |
|---|---|---|---|
| E4 | presence 加权 | `--w_presence 4` | 修复欠分离 → 房间数 8.5→9.4、↑Coverage |
| E5 | 放大容量 | `--d_model 256 --n_layers 6`（9.6M） | 更大容量 → ↓FID |
| E6 | 更多 Fourier 频带 | `--fourier_freqs 32` | 更细边界编码 → ↑Density |
| E7 | 更长训练 | `--steps 30000` | 排除欠拟合 |

### Tier 3 — 不同模型 / 结构性改动（更大改动，分别评估）

1. **DETR 式可学习 query**：给 N 个 slot 加可学习 query embedding（打破纯 set 对称），配合二分匹配/匈牙利损失做集合预测 → 直接改善 count 方差与 presence 分离。改 `model.py`(embed_tok) + `flow.py`(loss)。**针对已知问题 #1 最对症**。
2. **两段式（count→layout）**：先用轻量 head 从 outline 预测房间数分布并采样 K，再只对 K 个 token 跑 flow → 根治房间数欠分散，Coverage 上限更高。
3. **presence 独立分类头 + focal/类别平衡**：把 presence 从回归通道拆成独立二分类（focal loss），消除「欠分离 → 阈值钳制」。低成本、对症 #1。
4. **边界编码器升级**：PointNet-lite → 1D 卷积 / 小 Transformer over 边界点（带顺序），或图注意力，捕捉边界的局部连续几何 → ↑Density。
5. **解码器**：轴对齐矩形 → 允许有限旋转 / 简单多边形（snap 到边界角点），减少与真实多边形布局的分布差 → ↓FID。
6. **采样器**：rectified-flow 增加采样步数 / Heun，或重流（reflow）一次拉直轨迹 → ↓FID。

### 评估与验收
- 每个变体出一行 FID/D/C，并核对 `[diag] rooms/plan real ...` 一致（held 对齐）。
- 写回本文件 §5 实验日志与 §3 对照表。
- Tier 1/2 跑完确定最佳配方 → 迁 plan_id（steps 20k–30k）对标 190.5/0.052/0.060。

> 驱动脚本：`scratchpad/run_matrix.sh`（Tier 1+2 串行，各写独立 `exp_*` 目录）。

---

## 7. 第二轮：生成范式实验（method-level，待 Tier 1/2 后执行）

**前提**：先用 Tier 1/2 选出的**最优架构配方**作为固定底座（架构/数据/步数不变），只换**训练目标 + 采样器**，单变量归因「生成方法」本身值多少。当前底座 = rectified flow（线性 OT 路径，x_t=(1−t)x0+t·x1，target v=x1−x0，x0 噪声 x1 数据）。

### A. 低成本（仅改 `flow.py`，不动架构，每个 ~7 min 或免训练）

| # | 方法 | 改动 | 假设 / 针对指标 |
|---|---|---|---|
| M1 | **非均匀时间步采样**（SD3/logit-normal） | `fm_loss` 里 t 从 logit-normal 采样而非 U(0,1)，强化中段时间步 | rectified-flow transformer 实证能提 FID，~15 行，零额外成本 |
| M2 | **采样器升级** | `sample()` 用 Heun 全程 / 增步 / DPM-solver-like | 免重训，直接 ↓FID（轨迹积分更准） |
| M3 | **概率路径切换** | 线性 OT → **cosine / VP（diffusion）路径**：x_t=α_t·x1+σ_t·x0，target 改为对应路径的条件速度 | 路径形状影响易学性与 FID；对照 OT vs diffusion path |

### B. 不同生成范式（中等成本，改 `flow.py`+`train.py`，各 ~10–15 min）

| # | 方法 | 要点 | 假设 / 针对指标 |
|---|---|---|---|
| M4 | **EDM（Karras 2022）** | 预处理去噪器 D=c_skip·x+c_out·F(c_in·x; c_noise)；σ~lognormal(P_mean −1.2, P_std 1.2)；loss=λ(σ)‖D−x1‖²；采样 Heun + σ 调度(ρ=7) + 可选 churn | **FID 最强候选**；预处理对宽动态范围最稳 |
| M5 | **Stochastic Interpolant / SDE 采样** | 在 FM 上加 score 项，推理用随机（Langevin 校正）采样而非确定 ODE | 随机采样改善 mode 覆盖 → **专攻 Coverage**（本任务核心指标之一） |
| M6 | **Reflow（重流蒸馏）** | 用已训 RF 生成 (x0,x1) 配对，再训一遍拉直轨迹，支持 1–2 步采样 | 轨迹更直 → 少步且常更干净；↓FID + 提速 |

### C. 结构性范式（高成本，对症 §4 主瓶颈 presence 欠分离）

- **集合扩散 + DETR query + 匈牙利匹配损失**：把房间数/presence 当集合预测，二分匹配监督，根治房间数欠分散。
- **两段式**：先扩散/回归出房间数与类别直方图，再条件生成几何 → count 与 Coverage 上限更高。
- **离散-连续混合扩散**：type 用离散扩散（D3PM/多项式）、几何用连续 flow，匹配「类别+坐标」的混合本质。

### 最优配方（目标）
**最优 = (Tier1/2 赢家架构) × (M-series 最佳生成方法) × (C 类 presence/count 结构)。** 三轴各自单变量定标后再组合，最后迁 plan_id 对标官方口径。

### 验收
每个 method 变体同口径出 FID/D/C，写回 §3/§5；与 rectified-flow 底座逐行对比，重点看 **FID（M1/M2/M3/M4）与 Coverage（M5）**。

### 7.x M-series 结果（unit_id, 底座=full配方, eval g=1.5）

| 方法 | FID ↓ | Density ↑ | Coverage ↑ | 结论 |
|---|---|---|---|---|
| 底座 RF（100/heun5） | 112.2 | 0.084 | **0.124** | 参照 |
| **M2** 100/full-Heun | **111.9** | 0.080 | 0.118 | FID ±1 噪声内、C 反降 |
| **M2** 200/full-Heun | 112.9 | 0.087 | 0.114 | 同上 |
| M1 logit-normal t | 119.7 | 0.080 | 0.098 | **更差**（见下） |
| M4 EDM | 113.9 | **0.089** | 0.106 | D/房间数↑、interior cov 0.87 最佳，但 FID 没破 floor、C↓ |

**M2 结论（重要负结果）**：采样器升级（full-Heun / 增步）对 FID 无实质改善、Coverage 反降 → **ODE 离散化不是瓶颈**，默认采样器已近收敛。~112 的 FID floor 是**模型/分布**层面的限制，提升必须来自**训练目标（M1/M4）**或**结构（§7-C）**，而非采样。

**M1 结论（负结果）**：logit-normal 时间步（强化中段 t）全面变差（FID 112→120）。本任务的 presence 通道靠近 t→1（数据端）才稳定，logit-normal 反而欠采样两端 → 伤害。**保持 uniform-t RF**。

**M4/EDM 结论**：EDM 提升 Density(0.089) 与 interior coverage(0.87)、房间数略回升(8.63)，但 **FID 没破 floor（113.9）**、Coverage 反降。三个 method 实验（M2/M1/M4）一致表明：**生成范式不是瓶颈**。每次运行都打印 `under-separates presence` → 瓶颈是 **presence/count 结构**。

### 7.y 决定性结论 & 转向 §7-C

三类杠杆已测：① 架构超参（Tier1/2，full 最优、放大退化）；② 采样器（M2 无效）；③ 生成目标（M1 负、M4 EDM 平 FID）。**全部卡在 ~112 FID + presence 欠分离**。→ 唯一未碰、且被反复指向的是 **presence/count 的建模结构**。下一步实现 **C1（per-outline 计数匹配解码，免重训，先在现有 ckpt 验证）**：用 outline 面积预测每个户型的房间数 K，解码改为**按 presence 排名取 top-K**（而非单一全局阈值），直接修复「房间数 8.5<9.4 + 阈值 clamp + 方差」。若 C1 有效再上 presence 独立头/集合匹配重训。

