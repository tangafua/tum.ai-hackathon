# OutlineFlow 实验交接 — baseline(jiahua) vs improved 对照

> 给接手的 Claude Code:请用 **plan 模式**先制定计划再执行。本文件自包含,你无需其它上下文。
> 工作目录:`/root/kaiyingwu/outlineflow/`。环境:**AMD GPU + ROCm**,虚拟环境在 `/root/kaiyingwu/rocm-venv/`(执行前先 `source /root/kaiyingwu/rocm-venv/bin/activate`)。

## 1. 项目一句话

Flow-matching 模型,**只给公寓轮廓(outline)**,生成室内房间多边形。用 FID↓ / Density↑ / Coverage↑ 对照真实瑞士户型(MSD)打分。

## 2. 这次要做的事

把 **jiahua 的基线** 和 **我的改进版** 做一次**公平对照**,产出一张 FID/Density/Coverage 对比表。

- **基线 = git HEAD**(分支 `jiahua`,commit `ad24843`):无 cross-attention、无 outline 数量条件、mandatory gap-fill 解码器。
- **改进 = 当前工作区未提交改动**(`git status` 里的 M 文件):
  - `model.py`:加了 cross-attention(room token → 边界点)
  - `params.py`:加了 `outline_cond`(用轮廓面积/周长条件化房间数)+ 修复 padding slot 编码
  - `postprocess.py`:capped gap-fill(跳过过大补丁,防"绿斑")
  - `cfg.py/flow.py/train.py/synth_data.py`:上述功能的配套管线 + CFG guidance 支持

## 3. 已验证的事实(无需重复检查,但执行前可快速确认)

- 我**没有领先 `origin/jiahua` 任何 commit**(`git log origin/jiahua..jiahua` 为空)→ 改进全在未提交工作区。`git stash` 即可在"基线"和"改进"间切换。
- **评估管线在基线和改进间完全一致**:`metrics.py`、`render.py`、`msd_data.py` 都**不在** M 列表 → 两边打分代码相同 → FID 可比。
- **切分可复现**:基线和工作区的数据加载都是 `load_msd_samples` → `rng.shuffle(all_s)` → `[:n_held]`;同 `seed/msd_limit/group/n_held` 下两边 `held.pkl` 必然逐字节相同。

## 4. 已敲定的决策(不要改)

1. **CSV 统一用 kagglehub 全量**:
   `/root/.cache/kagglehub/datasets/caspervanengelenburg/modified-swiss-dwellings/versions/6/mds_V2_5.372k.csv`
   (md5 `c9b71f58...`,1,086,847 行。这是真全量,和 EDA 报告对得上。**别用别处的同名 CSV,它们是不同文件**。)
2. **decoder 算"改进的一部分"**:基线用它自己的 mandatory gap-fill,改进用 capped。两边各用各的代码版本即可(随 git stash 自动切换),无需额外处理。

## 5. 冻结的共享参数(两次运行必须一致)

| 项 | 值 |
|---|---|
| group | `plan_id` |
| msd_limit | `6000` |
| n_held | `1000` |
| seed | `42` |
| n_eval | `600` |
| 评估设备 | **CPU**(`--device cpu`):本机 metrics.py 用 torchmetrics 矩阵 sqrt,在 ROCm GPU 上可能踩 LAPACK 问题。训练用 GPU,**打分用 CPU** 最稳。 |

## 6. 怎么跑

已有脚本 `run_ablation.sh` 封装了全流程(stash 基线 → 训练+评估 → pop 改进 → 训练+评估 → 出对比表,并带 `trap` 保证中断也能恢复工作区)。

**第一步(必做):小规模冒烟测试**,确认流程不报错——把脚本里 `LIMIT=6000` 改 `500`、`NEVAL=100` 跑一遍:
```bash
cd /root/kaiyingwu/outlineflow && bash run_ablation.sh
```

**第二步:确认无误后用正式参数(LIMIT=6000 / NEVAL=600)跑全量。** 会训练两次,每次数小时。

## 7. 验收标准(跑完必须核对)

1. 脚本结尾打印的对比表有 baseline / improved 两行的 FID / Density / Coverage。
2. **两个 `eval.log` 里的这两行必须完全相同**(证明真实评估集对齐,结果才有效):
   - `[diag] rooms/plan  real X±Y`
   - `[sanity] real-vs-real ... FID≈0 / D≈1 / C≈1`
   若对不上,说明 held 集没对齐,结果作废,需排查 CSV / seed / split。
3. 改进版的 FID 应低于基线、Density/Coverage 应高于基线(否则改进无效,需如实报告)。

## 8. 注意事项

- **不要 commit、不要 push**(除非我明确要求);脚本只产出 `outputs_baseline/`、`outputs_improved/` 两个新目录。
- 训练前清理 `__pycache__`(脚本已含),避免 stash 切换代码后用到旧字节码。
- 全程忽略 `/shared-docker/aymen-work/` 及任何 aymen 相关文件,与本对照无关。
- 如训练在 ROCm 上 OOM 或报错,先减小 `--batch_size` 或 `--steps` 再试,并把报错原样反馈,不要静默改协议。
