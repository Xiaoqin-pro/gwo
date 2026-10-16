# RASE-GWO — 资源感知稀疏评价灰狼优化

面向**昂贵黑箱优化**的 GWO 改进：把「候选位置生成」和「真实函数评价」解耦，
让 GWO 在**固定的真实评价预算**下把评价名额花在最有信息量的候选狼上。

研究问题（不是"再造一个搜索算子"）：

> 当真实评价昂贵、位置更新便宜时，一只灰狼群应该把有限的名额给哪些狼？

## 目录

```
setup_project.m           把 src/、data/cec2017/ 加进 MATLAB 路径
src/GWO.m                 标准 GWO（全评价基线，CEC 接口）
src/RASE_GWO.m            本文方法 + 全部消融开关
src/algorithmConfig.m     算法注册表（名字 -> 入口/种群/开关）
src/nnProbe.m             一轮评价的空间冗余度统计（论文动机图用）
src/CEC2017Function.m     cec17_func 的薄封装
experiments/run_batch.m   并行实验驱动（parfor，每跑完一次立刻落盘 CSV + 进度可查）
experiments/run_screen.m  早期版本（整批跑完才写 CSV，保留做回归）
experiments/quick_check.m 串行快速试跑（开发调试用）
experiments/analyze_screen.py   统计与 REPORT.md 生成
experiments/make_tables.py      论文用 LaTeX 三线表
experiments/plot_figures.py     收敛曲线 / 排名图 / 误差比热图
experiments/merge_baselines.py  合并外部基线批次（配对种子）
data/cec2017/             cec17_func.mexw64 + input_data（D30/D50）
docs/01_ALGORITHM_SPEC.md 算法规格（唯一事实来源，改机制先改它）
PAPER_PLAN.md             论文规划与实验清单
paper/rase_gwo/           IEEEtran 论文骨架 + 自动生成的表格
results/                  每次实验一个时间戳目录
```

## 跑实验

```matlab
setup_project();
out = run_batch('formal');            % 29 函数 x 30 种子 x 7 变体（每跑完一次立刻落盘）
out = run_batch('formal', struct('resume', out.outDir));   % 中断后接着跑
```

进度直接数文件：

```powershell
(Get-ChildItem results\formal_*\runs\run_*.csv).Count     # 已完成次数
Get-Content results\formal_*\progress.log                 # 每次调用的汇总
```

统计与出图：

```powershell
python experiments\merge_baselines.py results\<batch> --external <外部基线 runs.csv>
python experiments\analyze_screen.py results\<batch> --file merged_runs.csv
python experiments\make_tables.py    results\<batch> --file merged_runs.csv --out paper\rase_gwo\tables
python experiments\plot_figures.py   results\<batch>
```

## 协议（同一批次内所有算法完全一致）

- CEC2017，D=50，搜索范围 [-100,100]，真值预算从初始化开始计。
- 每个 (函数, 种子) 只抽**一次**随机点池（池大小 = 该批次最大种群），
  每个算法取该池的**前缀**，因此配对比较共享初始点；前 50 点与外部基线批次逐位一致。
- 30 个独立种子（3001–3030）；批次内所有算法共享同一批种子。
- 统计：函数均值胜负、逐种子配对胜负、每函数双侧 Wilcoxon 秩和检验（Holm 校正）、
  平均排名（Friedman）。

## 算法开关（消融用）

| 开关 | 取值 | 作用 |
|---|---|---|
| `evalMode` | `sparse` / `dense` | 稀疏评价 vs 全评价 |
| `K` | 整数 | 每轮真实评价名额（默认 `round(rho*N)`） |
| `M`（即 `N`） | 50 / 250 / 500 / 1000 / 2000 | 虚拟候选池大小（过采样倍数） |
| `screenPolicy` | `geometry` / `random` | 几何筛选 vs 随机挑（去掉筛选机制） |
| `divCriterion` | `centroid` / `leaders` / `minority` | 探索名额给谁 |
| `useVirtual` | `true` / `false` | 领导层是否只读"已验证位置" |
| `verifiedMode` | `latest` / `best` | 已验证位置的更新语义 |
| `quotaPolicy` | `fixed` / `q` / `random` | 收敛/多样性配额：固定 / Q 学习 / 同触发随机 |
| `convShare` | 0.5 / 0.7 / 0.9 / 1.0 | 固定配额下给"收敛组"的比例 |

## 最新结果（2026-10-16 正式批次）

批次：`results/formal_20261016_045630/`（详细中文总结见其中的 `SUMMARY_zh.md`）；
CEC2017，D=50，100000 次真实评价，29 函数 × 30 种子，0 失败；
合并 8 个外部基线后共 15 算法 13050 次运行。

**本仓库内部（7 变体，等 FE 等代数）**

| 算法 | 平均排名 | 对 GWO 函数胜负 | 中位误差比 |
|---|---:|---:|---:|
| RASE-GWO (Q) | **2.62** | **25/4** | **0.874** |
| 固定 share=1.0 | 2.69 | 25/4 | 0.864 |
| 固定 share=0.9（少数派） | 2.86 | 23/6 | 0.882 |
| 随机筛选（去几何筛选） | 4.66 | 17/12 | 0.993 |
| GWO | 4.93 | – | 1.000 |
| 无虚拟/已验证解耦 | 7.00 | **0/29** | **2.862** |

**合并外部基线后（15 算法，29 函数 × 30 种子）**

| 排名 | 算法 | 平均排名 |
|---:|---|---:|
| 1 | HIDMS-PSO | 2.17 |
| 2 | HCLPSO | 2.72 |
| 3 | DMS-PSO | 2.83 |
| 4 | SAQ-CSO | 5.17 |
| 5 | SPSO-2011 | 5.66 |
| 6 | HPSO-TVAC | 8.21 |
| 7 | CLPSO | 8.24 |
| **8** | **RASE-GWO (Q)** | **8.28** |
| 11 | GWO | 11.10 |
| 15 | Original-CSO | 14.52 |

**结论（含不利结果，不掩盖）**

- 几何筛选 + 过采样 + 验证解耦三者都成立，消融干净：去掉筛选掉 2 个名次，
  去掉验证解耦后**比原始 GWO 还差 2.86 倍**。
- 相对 GWO：29 函数 25 胜 4 负，中位误差比 0.874。
- **绝对水平仍是中游**：对 HIDMS-PSO/HCLPSO/DMS-PSO 的中位误差比约 2.1–2.3 倍，
  对 SAQ-CSO 约 1.4 倍；30 种子下没有单个函数达到 Holm 校正显著。
- 运行时长约为 GWO 的 2.8 倍（6.27 s vs 2.27 s）；CEC2017 目标函数便宜，
  论文必须用"评价昂贵、位置更新便宜"的前提解释适用范围。

## 状态

- [x] 标准 GWO + RASE-GWO + CEC2017 并行实验管线
- [x] 协议核验（初始种群与外部基线批次逐位一致，可合法配对合并）
- [x] 正式主实验 29 函数 × 30 种子 + 完整消融
- [x] 增量落盘的 runner（`run_batch.m`）、表格与绘图脚本
- [ ] 补强 GWO 家族基线（已发表的强 GWO 变体）
- [ ] 半径分层的评价配额（针对 F1/F3/F4/F12/F13 等病态函数的改进方向）
- [ ] 论文正文（骨架与参考文献已在 `paper/rase_gwo/`）
