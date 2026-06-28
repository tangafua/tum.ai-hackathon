# OutlineFlow 深度诊断报告

**审计日期：** 2026-06-27 | **检查点：** `outputs/ckpt.pt`（plan_id, n_max=140, 5000步, 1.33M参数）

---

## 0. 诊断摘要

| 症状 | 指标/日志证据 | 根因（一句话） |
|---|---|---|
| presence 阈值跌破填充基线 | `[calib] presence_thresh -1.031`（低于填充 -1.0） | 模型未能区分真实房间与填充槽，所有 140 个槽位 presence 分布趋同 |
| 数量方差坍缩 | `real 36.27±20.13 → gen 31.44±6.03` | 全局池化的 128-d 条件向量无法编码每个平面图的房间数量 |
| Density=0, Coverage=0 | `FID 3682, D=0.000, C=0.000` | 生成的 token 几何量近乎噪声，gap-fill 把整个轮廓涂成 1-3 个超大房间，完全偏离真实流形 |
| Coverage=1, overlap≈0（后处理层面） | `coverage 1.000 overlap 0.0007` | 强制 gap-fill（`postprocess.py:99-107`）掩盖了布局崩溃 |
| 生成图右侧为大块绿色 + 散点矩形 | `pair_000.png` 视觉 | gap-fill 将残余面积归入最近房间（多为 Balcony=绿色），导致大色块 |
| 槽位排列增强从未生效 | `train.py:85` + `synth_data.py:100` | `build_tensors` 在训练前一次性调用 `build_x1`，每个 plan 全程使用固定槽位映射 |
| 填充槽 type 通道为 0 而非 -1 | `params.py:153,164` | `np.zeros` 初始化后只对真实槽写 ±1，填充槽 type 目标不一致 |
| 数据严重不足 | `[data]` 日志默认 `msd_limit=2000` | 实际训练平面图仅约 1600 个，CSV 有 5372k 行 |

---

## 1. 数据流水线审计

### 1.1 CSV → token tensor 完整路径

```
msd_data.load_msd_samples()           # msd_data.py:104
  └─ _build_sample()                   # msd_data.py:83
       ├─ _as_polygon(wkt)             # WKT 解析，MultiPolygon 取最大块（隐患见 1.2）
       ├─ build_outline(room_polys)    # buffer(+0.3).union.buffer(-0.3)（正确）
       └─ params.mrr_params(poly)      # → (cx,cy,w,h,theta)（正确）

train.py:84  stats = compute_stats(train_samples, CFG)   # 仅训练集，正确
train.py:85  X, OUT = build_tensors(train_samples, stats, CFG, rng)
  └─ synth_data.build_tensors()        # synth_data.py:94
       └─ params.build_x1(rooms, outline, stats, cfg, rng)  # 逐 plan 调用一次 ← BUG
```

### 1.2 已确认的数据流 Bug

**BUG-1：WKT 多边形截断（`msd_data.py:58-59`）**

```python
# msd_data.py:58-59
if geom.geom_type == "MultiPolygon":
    geom = max(geom.geoms, key=lambda g: g.area)   # ← 只保留最大块
```

对于 L 形或有内院的房间，MRR 后的多边形可能被 shapely 解析为 MultiPolygon，`_as_polygon` 静默丢弃小块。后果：某些房间的几何面积被低估。影响较小但存在。

**BUG-2：填充槽 type 通道为 0，不是 -1（`params.py:153,164`）**

```python
# params.py:153
x1 = np.zeros((cfg.n_max, D), dtype=np.float32)   # type channels[7:20] = 0
x1[:, 0] = -1.0                                    # presence = -1
# ...
# params.py:164-165: 只对真实槽写 ±1
x1[slot, 7:7 + cfg.k] = -1.0
x1[slot, 7 + int(t) % cfg.k] = 1.0
```

真实槽的 type 目标是 `{-1,+1}`，填充槽的 type 目标是全零 `{0}`。在整流流的训练路径 `xt = (1-t)*x0 + t*x1` 中，t 接近 1 时 xt 接近 x1，这个目标不一致会给 type 通道引入错误梯度。

**修复（一行）：**

```python
# params.py:153 改为：
x1 = np.full((cfg.n_max, D), -1.0, dtype=np.float32)
```

**BUG-3：槽位排列在训练前固定（`synth_data.py:100` + `train.py:85`）**

```python
# train.py:85 — 训练前只调用一次
X, OUT = synth_data.build_tensors(train_samples, stats, CFG, rng)

# synth_data.py:100 — 每个 plan 调用一次 build_x1，rng 仅推进一次
X[i] = params.build_x1(s["rooms"], s["outline"], stats, cfg, rng)
```

`build_x1` 内部 `rng.permutation(n_max)[:n]` 只在构建阶段调用一次。5000 步训练中，plan_i 的房间永远在同一批槽位上。排列不变性增强完全失效。详细修复见第 6 节。

### 1.3 Stats 泄漏检查

无泄漏。`train.py:84` 在 held/train 分割后只对 `train_samples` 计算 stats，eval 时从 checkpoint 加载同一 stats（`sample_eval.py:34`）。

### 1.4 轮廓编码一致性

`sample_outline_points`（`params.py:82`）在训练（`synth_data.py:101`）和 eval（`sample_eval.py:74`）中调用方式完全一致，`P=cfg.p_outline=128`。法向量方向判断（`params.py:116-119`）用 shapely `contains` 做"指向内→翻转"，逻辑正确。

---

## 2. 模型架构审计

### 2.1 信息流 ASCII 图

```
outline_pts [B,128,4]
      │
      ▼
OutlineEncoder (PointNet-lite)          model.py:32-47
  shared MLP: 4→128→128→128
      │
  max-pool ⊕ mean-pool over 128 pts    ← 空间细节在此丢失 ①
      │
  proj: 256→128                        outline_vec [B,128]
      │
      ▼
c = SiLU(t_embed [B,128] + outline_vec [B,128])    model.py:99
      │
      ▼  (同一个 c 广播到所有 token)
AdaLNZeroBlock × 4                     model.py:88,100-101
  self-attn(room tokens → room tokens)
  AdaLN from c (no per-token spatial info)  ← 条件维度有效为 128-d 全局标量 ②
      │
      ▼
norm_out + ada_out(c) → head            model.py:102-104
      │
      ▼
v [B,N,D]  速度场预测
```

**① 空间信息丢失点：** `model.py:46` 的 `max-pool` + `mean-pool` 将 128 个边界点压缩为 1 个向量。房间 token 无法"看到"边界点的具体位置，无从知道自己应该靠近哪条墙。这是 Density=0 的结构性天花板。

**② 每 token 有效条件维度：** 128-d，且对所有 token 相同。模型无从区分"这个 token 应该在左上角"和"这个 token 应该在右下角"。

### 2.2 各组件详细分析

**OutlineEncoder（`model.py:32-47`）**
- 计算什么：PointNet-lite 提取每个边界点的局部特征，然后全局池化。
- 丢失什么：池化后的向量保留了"整体形状/大小"，但丢弃了"边界的哪部分在哪里"。模型只知道"这个公寓大致是这种形状"，不知道"东北角有一个凹进去的角"。

**AdaLNZeroBlock（`model.py:54-75`）**
- 计算什么：自注意力让 room token 互相交流，AdaLN 用全局条件 `c` 调制归一化。
- 问题：所有 token 共享同一个 `c`，自注意力是 token 间交流的唯一机制，但没有与边界点的交叉注意力，token 无法锚定到边界区域。

**时间嵌入（`model.py:84-85`）**
- `GaussianFourierProjection` + 2-layer MLP，scale=30.0，设计合理。

**头部零初始化（`model.py:93-95`）**
- `ada_out` 和 `head` 权重/偏置零初始化，训练起点为恒等映射，稳定性好。

### 2.3 参数量与任务复杂度的匹配

```
1.33M 参数 / ~1600 训练平面图 / 5000 步
每步见到 batch_size=128 个 plan，相当于 ~400 个有效 epoch
plan_id 平均 37 个房间，MultiPolygon 轮廓，n_max=140
```

对 plan_id 的复杂度而言，1.33M 参数明显不足（欠拟合），不是过拟合。切换到 unit_id（~9 个房间，n_max~25）则匹配良好。

---

## 3. 训练循环审计

### 3.1 整流流约定检查（`flow.py`）

```python
# flow.py:36-37
xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1   # 噪声 t=0，数据 t=1
v_target = x1 - x0                                           # 常数速度场
```

约定：噪声在 t=0，数据在 t=1。采样器（`flow.py:62-70`）从 t=0 积分到 t=1。**约定一致，无 off-by-one。**

Heun 修正（`flow.py:65-68`）在最后 5 步使用，t2 正确 clamped 到 1.0。**无误。**

### 3.2 损失权重与类别频率匹配（`flow.py:14-29`）

```python
# flow.py:22-28
cw[1:7] = cfg.w_geometry  # = 1.0
cw[7:]  = cfg.w_type      # = 0.5
W[..., 0] = cfg.w_presence  # = 2.0，所有槽
slot_w = present + (1 - present) * cfg.w_pad_slot  # 填充槽权重 0.3
```

- presence 通道（w=2.0）压制填充正确。
- `w_pad_slot=0.3` 对填充槽的几何/type 降权合理。
- **潜在问题**：由于 BUG-2（填充 type=0），即使用 w_pad_slot=0.3 降权，仍在用错误目标（0 而非 -1）训练 type 通道。

### 3.3 BUG-3 精确定位：槽位排列时机

```python
# train.py:85 ← 此处调用一次，之后 X 静态不变
X, OUT = synth_data.build_tensors(train_samples, stats, CFG, rng)

# synth_data.py:94-102
def build_tensors(samples, stats, cfg, rng=None):
    for i, s in enumerate(samples):
        X[i] = params.build_x1(s["rooms"], ..., rng)  # 每 plan 一次
    return X, OUT
```

### 3.4 EMA 衰减（`flow.py:74-100`）

decay=0.999，5000 步后初始权重残余 0.67%，EMA 权重以最后几百步为主。合理，无问题。

### 3.5 梯度裁剪

`train.py:109`：`clip_grad_norm_(model.parameters(), 1.0)`。合理但无日志记录梯度范数——建议每 100 步打印以监控训练稳定性。

---

## 4. 采样与后处理审计

### 4.1 单样本从 t=0 到最终多边形的完整追踪

```
t=0: x ~ N(0,I)  [1, 140, 20]          flow.py:60

Euler 步 × 95 步 + Heun 修正 × 5 步     flow.py:62-70
模型预测速度 v ← 由于条件信息不足，v 近乎随机偏移
结果: x[t=1] ← 几何量接近训练数据均值，方差很小
      presence 通道: 所有槽位约在 -1 附近（模型未学会区分）

↓
decode_x1(x1_np, outline, stats, cfg)   params.py:169-196
→ 140 个 dict，presence 分布在 [-1, ~0.5]，已排序

↓
calibration: binary search thresh=-1.031  sample_eval.py:99-107
→ thresh=-1.031 < -1.0，所有 140 个槽位通过 presence > -1.031
→ present = 全部 140 个 dict

↓
layout_from_tokens: clip 每个矩形到 outline  postprocess.py:69-76
→ ~36 个矩形（噪声几何）与 outline 有交叠，其余变 empty
→ 这 ~36 个矩形尺寸随机，几何混乱

↓
greedy overlap-resolve              postprocess.py:81-91
→ 高-presence/大面积的矩形先占地，后续矩形被切除
→ 最终保留 ~31 个互不重叠的零碎矩形

↓
sliver drop (area < 0.005 × outline.area)  postprocess.py:93-95

↓
gap-fill: leftover 合并入最近房间   postprocess.py:99-107
→ leftover = outline - union(rooms) 可能是大面积
→ 被 gap-fill 合并到最近的 1-3 个房间
→ 视觉结果: 少数房间变成不规则巨型多边形（绿色 Balcony 等）
```

### 4.2 presence 阈值校准路径病（`sample_eval.py:91-107`）

校准的二分搜索范围：

```python
lo = float(min(x[:, 0].min() for x in raw))  # 可能远低于 -1
hi = float(max(x[:, 0].max() for x in raw))  # 可能略高于 +1
```

当模型质量差时，`lo` 下探到 -1.05 乃至更低，搜索就把 thresh 推到 -1.031，包进所有槽位。**这是症状，不是原因**。根治需要让模型学会正确的 presence 分布（真实槽≈+1，填充槽≈-1）。

临时缓解（不训练也能用）：

```python
# sample_eval.py:107
thresh = max(0.5 * (lo + hi), -0.5)   # 添加下界 -0.5，避免纳入真正填充槽
```

### 4.3 gap-fill 的副作用

`postprocess.py:106`：`comp` 的面积不限，任何大小的 leftover 都会被无差别 gap-fill。当 leftover 占 outline 面积 60%+ 时，gap-fill 后的几何图形与真实平面图完全不同。

### 4.4 eval/train 信息泄漏检查

- **Stats**：仅训练集，正确（见 1.3）。
- **Held 集**：`train.py:78-79` 分割发生在 `rng.shuffle` 之后，固定 seed=42，可复现。无泄漏。
- **轮廓**：eval 时 `sample_eval.py:74` 从 held 集取轮廓，模型训练时只见过 train 集轮廓。无泄漏。

---

## 5. 指标审计

### 5.1 phi-proxy 的本质与局限

`phi()`（`metrics.py:21-60`）提取的是一个 **30 维**手工特征向量：

```
[1]     房间数量 n
[13]    房间类型归一化频率直方图
[8]     面积归一化直方图（8 个 bin）
[2+2+2] 宽/高/长宽比的均值+标准差
[1]     未覆盖背景比例 bg_frac
[1]     相邻房间对数 contacts
```

### 5.2 FID=0, D=1, C=1 的视觉含义

- **FID=0**：生成平面图的 phi 特征分布与真实集完全相同。视觉上：每个生成图有 36±20 个正确类型的房间，均匀铺满轮廓。
- **Density=1**：每个生成样本都落在至少一个真实样本的 k-NN 球内（k=5）。视觉上：生成图"看起来像"一个真实平面图。
- **Coverage=1**：每个真实样本的 k-NN 球内至少有一个生成样本。视觉上：生成图能覆盖真实数据的多样性。

---

## 6. 优先级修复清单

按对 **Density** 预期影响排序。所有修改在 unit_id、n=500 eval、2h 单 GPU 内可验证。

| 优先级 | 文件:行号 | 改动 | 预期指标变化 |
|---|---|---|---|
| **P0-A** | `model.py:87-101` | 在每个 AdaLNZeroBlock 后添加交叉注意力层 | Density: 0→0.1~0.3 |
| **P0-B** | `train.py:85,99-111` | 将 `build_x1` 移入训练循环，每 batch 重排 | 训练损失稳定，Density 提升 |
| **P1-A** | `params.py:153` | `np.zeros` → `np.full(..., -1.0)` | 消除 type 噪声 |
| **P1-B** | `sample_eval.py:107` | 添加 `thresh = max(thresh, -0.5)` | 临时修复阈值病（无需训练） |
| **P1-C** | `train.py:46` | 去掉 `--msd_limit 2000`，steps 增至 15000+ | FID 下降，Density 提升 |
| **P1-D** | `train.py:42-43` | 切换 `--group unit_id` 训练 | 用简单任务验证修复 |
| **P2-A** | `model.py:79-104` | 添加房间数量条件 | Count 方差恢复 |
| **P2-B** | `flow.py:32-40` | CFG 随机 drop outline，采样时 guidance | Density 提升 |
| **P3-A** | `synth_data.py:66-90` | 随机旋转/翻转平面图 | Coverage 提升 |

---

## 7. 推荐验证顺序

**假设条件：** 单 GPU，2h/次运行，unit_id 数据。

### 步骤 1：立即验证（无需重训，~10 分钟）

应用 **P1-B** 临时修复，重新运行 `python sample_eval.py --n_eval 120`

**成功条件：**
```
[calib] presence_thresh > -0.5
[diag] gen std > 2.0
```

### 步骤 2：基线建立（P1-A + P0-B + 全量数据，~1.5h）

```bash
python train.py --data_csv mds_V2_5.372k.csv --group unit_id --steps 5000 --msd_limit 99999
python sample_eval.py --n_eval 500
```

**成功条件：**
```
Density > 0.05
Coverage > 0.1
```

### 步骤 3：架构修复（P0-A，~2h）

添加 cross-attn，重训 unit_id

**成功条件：**
```
Density > 0.15  (相对步骤 2 提升 ≥ 3×)
```

### 步骤 4：迁移 plan_id（全量，~2h）

```bash
python train.py --data_csv mds_V2_5.372k.csv --group plan_id --steps 20000 --msd_limit 99999
```

---

## 关键监控指标

| 日志行 | 正常范围 | 异常处理 |
|---|---|---|
| `presence_thresh` | -0.5 ~ 0.8 | 若 <-0.5 → 检查 w_presence 和 BUG-2 修复 |
| `gen std / real std` | >0.4 | 若 <0.2 → count 条件崩溃，考虑 P2-A |
| `coverage` | >0.3 | 若 <0.1 → 多样性不足，增加训练数据 |
| training loss | 单调下降到 <0.5 | 若停滞 >1.0 → 检查 lr、数据管道 |
