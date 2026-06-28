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

### 3.2b plan_id 官方口径（全量数据, 20k steps, RF full + C1, g=1.5）

**plan_id 特有瓶颈 = 高房间密度下矩形 overlap-resolve 大量丢房间**（请求 K≈38、实际只解出 ~20，丢一半；interior cov 0.59）。对策 = `count_scale>1` 补偿丢房率。count_scale 扫描（n_eval=600, inception 官方口径）：

| config | FID ↓ | Density ↑ | Coverage ↑ | gen 房间 | interior cov |
|---|---|---|---|---|---|
| jiahua 基线 | 190.5 | 0.052 | 0.060 | — | — |
| C1 scale 1.0 | 186.2 | 0.100 | 0.062 | 19.5 | 0.59 |
| **C1 scale 1.5** ⭐ | **166.8** | **0.115** | **0.082** | 23.9 | 0.75 |
| C1 scale 2.0 | 164.8 | 0.109 | 0.075 | 24.6 | 0.78 |
| voronoi（弃） | 302 | 0.003 | 0.013 | 36.5 | 1.00 |

**plan_id 最优 = C1 scale 1.5：FID 166.8 / D 0.115 / C 0.082 → 对标基线 FID −12%, Density +121%, Coverage +37%。**

要点：
- **count_scale 最优值由解码器丢房率决定**：unit_id 丢 ~2/9（22%）→ scale 1.0 最优；plan_id 丢 ~18/38（50%）→ 需 scale 1.5 补偿。两者相反但同源。
- scale 2.0 过填，FID 略好但 Density/Coverage 回落 → 1.5 是甜点。
- **voronoi 弃用**：几何完美（房间数≈real、gap-free、零重叠、interior cov 1.0），但非矩形 cell 对 FID/Density 特征完全 off-manifold（真实 MSD 房间是矩形）→ 指标崩。证明解码器必须输出矩形。

### 3.2c 与 jiahua 最新方案对比 + align/churn 叠加（plan_id, n_eval=600, seed42）

jiahua 最新（origin/jiahua `result.md`，3-seed 平均）：**FID 135.3 / D 0.088 / C 0.111**（FM + align-g16 + churn0.3）。她的杠杆：**align**（grid-snap 重分区→FID）、**churn**（SDE 采样→Coverage）。两者已 port 进我的 flow.py/postprocess.py/sample_eval.py（`--align --grid --churn`）。

把 align+churn 叠到我的 C1 流程（seed42, n_eval600）：

| config | FID ↓ | Density ↑ | Coverage ↑ |
|---|---|---|---|
| 我 pre-align scale1.5 | 166.8 | **0.115** | 0.082 |
| +align16 | 159.0 | 0.066 | 0.065 |
| **+align16 +churn0.3** | **156.1** | 0.068 | 0.085 |
| +align48 +churn0.3 | 178.6 | 0.064 | 0.058 |
| scale1.0 +align16 +churn0.3 | 184.6 | 0.049 | 0.060 |
| *jiahua final (3-seed)* | *135.3* | *0.088* | *0.111* |

**核心发现（反直觉）：align 摧毁了我的 Density 主优势**（0.115→0.066）。align 的 grid-snap+重分区把我「多样但抖动」的房间正则化到网格 → 几何多样性塌缩。churn 如期提 Coverage（+0.02）、grid16>>grid48。**naive 叠加未能打过 jiahua**。

更深：同一 align-g16 配方，我的模型 159/0.066 vs 她 131.5/0.091 → **她更简单的模型（无 cross-attn）对 align 响应更好**；我的 cross-attn/cond 高 Density 是「未对齐的抖动多样性」，经不起正则化。
> 注：两边 held 参照集不同（各自 train/held split），FID 非严格同集可比；趋势可信，绝对值留余地。

**方向修正**：我的模型 = Pareto 上的「高 Density 点」(167/0.115/0.082)，她 = 「高 FID/Coverage 点」(135/0.088/0.111)，互不支配。对我的模型，正确杠杆是**保 Density 的 Coverage 杠杆**，而非她的 align。

### 3.2d 解码侧 Coverage 杠杆全部探尽（plan_id, seed42, n_eval600）

试图在不伤 Density 的前提下提 Coverage：

| 杠杆 | 结果 | 解读 |
|---|---|---|
| **stochastic-count**（按 real log-count~log-area 采样 K，resid_std=0.21） | gen 房间数 std 仍 ±6.8（=确定性），C 0.082 不变 | **null**：注入的 K 方差被解码器吸收 → **方差崩塌是几何 packing 上限，不是 count 目标问题** |
| **min_area_frac↓**（0.005→0.002→0.001） | gen 房间 24→30→33、std ±7→±12→±13（逼近 real ±24），但 C 仅 0.082→0.085、FID/D 反降 | 打破了 packing 上限、count 方差开了，但**多出来的房间是噪声不是真实流形多样性** → Coverage 不涨、FID/D 退化 |
| churn（无 align, 0/0.5/0.7） | C 0.082→0.063→0.077，FID 反升 | **null**：churn 单独在我的模型上 Coverage 不升反降；只有与 align 同用才有 +0.02 微效 |

**结论：解码/采样侧前沿已探尽。** Coverage 天花板(~0.08)不是房间数问题，而是**生成的布局本身缺真实多样性**（L2 均值回归根因）。我的 plan_id 最优仍为 **C1 scale1.5 = 166.8 / 0.115 / 0.082**（高 Density Pareto 点）。要在三项上同时超过 jiahua，只能动**模型/训练层（Tier C）**。

**Tier C 方向（重训）**：Tier A 显示我的 cross-attn/cond 模型 align 后反不如 jiahua 更简单的模型（159/0.066 vs 131.5/0.091）。→ 首个 Tier C 实验为**架构消融重训**：去掉 cross-attn + cond（jiahua 式简化模型）+ 我的 C1 解码 + align。

### 3.2e Tier C 架构消融（重训 exp_plan_simple, 全量 plan_id, 20k, 无 cross-attn/无 cond）

**假设被推翻——这是关键正面结论。** 同数据/同 held/同解码下逐项对比：

| config（apples-to-apples） | FID ↓ | Density ↑ | Coverage ↑ |
|---|---|---|---|
| **我的 cross-attn+cond** + C1 1.5 | **166.8** | **0.115** | **0.082** |
| simple（无 cross-attn/cond）+ C1 1.5 | 193.9 | 0.078 | 0.053 |
| 我的 + C1 + align16 | 159.0 | 0.066 | 0.065 |
| simple + C1 + align16 | 176.2 | 0.041 | 0.060 |

**简化模型在所有指标上都更差。** Tier A 看到的「她的简单模型 align 后更好（131.5 vs 我 159）」是 **held 参照集混淆，不是架构差异**（正是先前标注的 caveat）。公平对比（同 split/held/解码）下 **我的 cross-attn + cond 架构完胜** → **架构工作得到验证**，cross-attn outline 编码 + count 条件是净正贡献。

**因此 jiahua 135.3 vs 我 166.8 不是同口径可比**（不同参照集 + 3-seed + 她的 calibration）。要严格对比需把两套 recipe 在同一 split 上重训——但我自己的 apples-to-apples 消融已证明：**我的完整流程（cross-attn + cond + C1 scale1.5）= 我的最优 = 166.8 / 0.115 / 0.082**，且每个组件都经消融验证为净正。

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
| 2026-06-28 | M-series 生成范式 M1/M2/M4 | unit_id | 15k | 全部负/平 FID → 瓶颈非生成范式（见 §7.x/7.y） |
| 2026-06-28 | **C1 per-outline top-K 解码** | unit_id | — | **破 floor：FID 109.8/D 0.118/C 0.132**（RF, g1.5, scale1.0） |
| 2026-06-28 | C1 count_scale×guidance 扫描 | unit_id | — | **scale1.0 最优**；放大 K 反伤（少而精＞多而杂）。**当前最优配方 = RF full + C1(g1.5,scale1.0)** |

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
| **C1 count-match (RF)** | **109.8** | **0.118** | **0.132** | **破 floor！全面最佳**（见 §7.z） |
| C1 count-match (EDM) | 113.2 | 0.110 | 0.110 | RF+C1 优于 EDM+C1 |

**M2 结论（重要负结果）**：采样器升级（full-Heun / 增步）对 FID 无实质改善、Coverage 反降 → **ODE 离散化不是瓶颈**，默认采样器已近收敛。~112 的 FID floor 是**模型/分布**层面的限制，提升必须来自**训练目标（M1/M4）**或**结构（§7-C）**，而非采样。

**M1 结论（负结果）**：logit-normal 时间步（强化中段 t）全面变差（FID 112→120）。本任务的 presence 通道靠近 t→1（数据端）才稳定，logit-normal 反而欠采样两端 → 伤害。**保持 uniform-t RF**。

**M4/EDM 结论**：EDM 提升 Density(0.089) 与 interior coverage(0.87)、房间数略回升(8.63)，但 **FID 没破 floor（113.9）**、Coverage 反降。三个 method 实验（M2/M1/M4）一致表明：**生成范式不是瓶颈**。每次运行都打印 `under-separates presence` → 瓶颈是 **presence/count 结构**。

### 7.y 决定性结论 & 转向 §7-C

三类杠杆已测：① 架构超参（Tier1/2，full 最优、放大退化）；② 采样器（M2 无效）；③ 生成目标（M1 负、M4 EDM 平 FID）。**全部卡在 ~112 FID + presence 欠分离**。→ 唯一未碰、且被反复指向的是 **presence/count 的建模结构**。下一步实现 **C1（per-outline 计数匹配解码，免重训，先在现有 ckpt 验证）**：用 outline 面积预测每个户型的房间数 K，解码改为**按 presence 排名取 top-K**（而非单一全局阈值），直接修复「房间数 8.5<9.4 + 阈值 clamp + 方差」。若 C1 有效再上 presence 独立头/集合匹配重训。

### 7.z C1 突破：per-outline top-K 解码（RF ckpt, g=1.5）

| 解码 | FID ↓ | Density ↑ | Coverage ↑ | interior cov | K/decoded |
|---|---|---|---|---|---|
| 旧（全局阈值+重 gap-fill） | 112.2 | 0.084 | 0.124 | 0.85 | —/8.5 |
| **C1 top-K** | **109.8** | **0.118** | **0.132** | 0.71 | 9.34±3.47 / 7.16 |

**关键洞察**：瓶颈不在速度场也不在生成范式，而在**解码**。旧 pipeline 用单一全局 presence 阈值 + 强制 gap-fill，把 under-separation 掩盖成「过度填满」(interior cov 0.85)，反而 off-manifold。C1 改为**按面积给每个 outline 算 K_i（仅用已许可的真实均值 × outline 自身面积，无泄漏，方差 3.47>真实 2.87）+ 按 presence 排名取 top-K**，渲染分布更贴近真实 MSD（房间间有黑色 wall/structure，interior cov 0.71 反而更真）→ FID 破 floor、Density +40%。RF+C1 优于 EDM+C1。**C1 设为默认解码**；细化扫描 count_scale × guidance 见 §5。

