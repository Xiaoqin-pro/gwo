# CEC2017 基线来源与适配

CLPSO：Liang et al., Comprehensive Learning Particle Swarm Optimizer for Global Optimization of Multimodal Functions (2006), DOI: https://doi.org/10.1109/TEVC.2005.857610 。`CLPSO_CEC.m` 将仓库 UAV 版的综合学习、逐维 exemplar、Pc 和刷新间隔 7 接到连续函数接口。c=1.49445，惯性权重由 0.9 降至 0.4，速度限幅为搜索区间的 15%；边界截断与其他群智能基线一致。

L-SHADE：Tanabe and Fukunaga, Improving the Search Performance of SHADE Using Linear Population Size Reduction (CEC2014), DOI: https://doi.org/10.1109/CEC.2014.6900380 。参考作者修正版 L-SHADE 1.0.1 MATLAB/Octave 包：

- 作者页面：https://ryojitanabe.github.io/publication
- 源码包：https://ryojitanabe.github.io/code/LSHADE1.0.1_CEC2014_Octave-Matlab.zip

`LSHADE_CEC.m` 是接口适配实现，保留 current-to-pbest/1/bin、随机 archive 去重与裁剪、MF/MCR 加权 Lehmer 均值、CR terminal value、F 的 Cauchy 抽样、边界中点修正和线性种群缩减。正式设置：NP0=18D=540，NPmin=4，p=0.11，archive rate=1.4，memory size=5。随机初始化由入口提供，搜索使用独立 seed，所有评价包含初始化逐次计 FE；最后一代只评价剩余预算。用 randn 替代 normrnd，避免依赖统计工具箱。种群缩减一次保留排序后的较好个体，不改变保留集合。

五算法共享问题、每次运行的初始化随机点池和 FE 上限。PSO、CSO、CLPSO、DSS-RLCSO 使用池中前 50 个点；L-SHADE 使用全部 540 个点，并按算法规则缩减。不能把这个协议描述为所有算法始终使用相同种群。

SS-RLPSO 和 CMA-ES 不在正式比较中。历史实验文件中的名称和数字属于当时版本；正式包装 `DSS_RLCSO_CEC.m` 固定使用 MAQL-V1，`DSS_MAQLCSO.m` 仅保留历史兼容。


## 2026-10-12 CLPSO来源审计补注（覆盖旧身份说法）
旧CLPSO_CEC.m是项目/UAV迁移适配，不等于作者源码。作者仓库P-N-Suganthan/CODES公开2006-IEEE-TEC-CLPSO.zip（SHA256 e79250ea6a9fc18a5976e19906f58ea6b8418352aa5a173087ee28a4e29f11a9）；其中CLPSO_new_func与原论文的Pc、gap等设置不一致，需区分。CLPSO_CEC_AuthorSource.m基于这个public codeprofile，仅做CEC接口/seed/p0/hard真实FE/trace/guard适配，并与机械原source oracle逐点评价/RNG回归。paper-only边界差异单因素另由CLPSO_CEC_Audit.m记录。后续对照不能把旧clip基线或新codeprofile标为“完整论文复现”。详情docs/CLPSO_AUDIT_RESULTS_20261012.md及CLPSO_FULL_100K_PROTOCOL_20261012.md。
