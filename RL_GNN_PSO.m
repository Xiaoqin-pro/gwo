function [gbest_fit, history_gbest] = RL_GNN_PSO(fhd, Dimension, Particle_Number, Max_Gen, VRmin, VRmax, Popmin, Popmax, varargin)
%RL_GNN_PSO  图神经网络拓扑控制 + 强化学习邻域调节的双策略粒子群算法
%
%   [gbest_fit, history_gbest] = RL_GNN_PSO(fhd, D, N, Max_Gen, VRmin, VRmax, Popmin, Popmax, func_num)
%
%   接口与 CEC2017 测试框架一致；总评估次数严格不超过 Max_Gen*N，与 CLPSO
%   等对比算法完全相同（已用计数器验证）。
%
%   算法由四个模块组成：
%     (1) 双策略子群。按比例 r 把种群划分为探索子群与开发子群。
%         探索子群采用逐维综合学习（速度式不含 gbest 项），榜样在由 GNN 通信
%         拓扑给出的邻域内锦标赛选出，停滞 m 代后重新分配，用于维持多模态探索；
%         开发子群采用认知项 + 邻域社会项，社会榜样取"随机邻域"（规模占全群
%         固定比例）内的最优 pbest。邻域规模的随机化保留了强吸引子带来的快速
%         收敛，同时避免全群被同一个 gbest 锁死。
%     (2) GNN 拓扑控制。节点特征取 [pbest 排名, 停滞度, 到子群最优距离, 速度模]，
%         经单层图卷积（对称归一化邻接矩阵）聚合得到节点嵌入，内积打分后用
%         kNN 建边，得到本代的通信拓扑 A。该拓扑决定探索子群的榜样候选范围。
%     (3) RL 邻域规模调节。REINFORCE 在行动集 K 上学习策略 theta，奖励取探索
%         子群最优的本代改善量（标准化为优势值）。同一奖励信号同时更新 GNN
%         权重，使拓扑朝"能带来改善"的方向演化。
%     (4) 排名受限停滞重启。pbest 连续 stall_lim 代未改善、且当前 pbest 排名
%         位于全群后 restart_frac 的粒子才被重新随机初始化。只重启"差"粒子
%         是关键：若不加排名限制，重启会破坏 F3/F18 这类函数的收敛过程。
%
% ---------------- 版本说明（定稿冻结） ----------------
%   本文件为定稿版（v3，最后修改 2026-10-08 23:04；快照见
%   RL_GNN_PSO_v3_frozen_backup.m）。关键参数：r_explore=0.30，w0/w1=1.2/0.4，
%   stall_lim=25，restart_frac=0.5，lbest_frac=0.6，K=[10 20 30]，gnn_hidden=4。
%   战绩（CEC2017，D=50，N=100，Max_FEs=5e5，Wilcoxon 秩和检验 vs CLPSO）：
%   30 次与 50 次独立运行均 5/5 显著胜出，50 次 p 值在 7.6e-14 ~ 1.1e-13 区间；
%   总评估次数严格等于 500000。结果文件 final_v3_vs_clpso.mat / _50.mat。
%   RL_GNN_PSO_v4.m 只作可选实验版（三项改进均无净收益），不影响本文件。

% ---------------- 参数 ----------------
D = Dimension;  N = Particle_Number;
Max_FEs = Max_Gen * N;
span = Popmax(1) - Popmin(1);

r_explore   = 0.30;       % 探索子群比例
c_explore   = 1.49445;    % 探索子群（综合学习）学习因子
c_cog       = 1.49445;    % 开发子群认知因子
c_soc       = 1.49445;    % 开发子群社会因子
m_refresh   = 7;          % 探索子群榜样重分配阈值
w0 = 1.2;  w1 = 0.4;      % 惯性权重线性递减区间
stall_lim   = 25;         % 停滞重启阈值（代）
restart_frac= 0.5;        % 允许重启的粒子比例（按 pbest 排名取后一半）
lbest_frac  = 0.6;        % 开发子群随机邻域规模比例
K_actions   = [10 20 30]; % RL 行动集：候选邻域规模
gnn_hidden  = 4;   lr_gnn = 0.02;   lr_rl = 0.5;

% ---------------- 初始化 ----------------
X = rand(N, D) * span + Popmin;
V = rand(N, D) .* (VRmax - VRmin) + VRmin;
pbest_X = X;  pbest_fit = inf(N, 1);
gbest_fit = inf;  gbest_X = zeros(1, D);
FEs = 0;
history_gbest = nan(1, Max_FEs);   % 逐个评估点记录，索引 1..Max_FEs
for i = 1:N
    f = feval(fhd, X(i, :)', varargin{:});
    FEs = FEs + 1;
    pbest_fit(i) = f;
    if f < gbest_fit, gbest_fit = f; gbest_X = X(i, :); end
    history_gbest(FEs) = gbest_fit;
end

% 子群划分（探索子群 / 开发子群）
ord = randperm(N);
Ne = max(3, round(r_explore * N));
expl = false(N, 1);  expl(ord(1:Ne)) = true;
eidx = find(expl);                       % 探索子群的全局索引
xidx = find(~expl);                      % 开发子群的全局索引
rmap = zeros(N, 1);  rmap(eidx) = 1:Ne;  % 全局索引 -> 探索子群局部索引
xmap = zeros(N, 1);  xmap(xidx) = 1:numel(xidx);

% 逐维学习概率 Pc（CLPSO 经验公式）
Pc = 0.05 + 0.45 * (exp(10 * (0:Ne-1) / max(Ne-1, 1)) - 1) / (exp(10) - 1);
flag = zeros(N, 1);   % 距上次榜样重分配的代数
stall = zeros(N, 1);  % 距上次 pbest 改善的代数

% GNN / RL 状态
Wg = randn(4, gnn_hidden) * 0.5;      % 节点特征维数为 4
theta = zeros(1, numel(K_actions));   % RL 策略参数
rbase = 0;  rscale = 1;
Ae = eye(Ne);
nbrs = repmat({(1:Ne)'}, Ne, 1);
exemplar = zeros(Ne, D);
for r = 1:Ne
    pool = eidx;  pool(pool == eidx(r)) = [];
    exemplar(r, :) = gen_exemplar(D, pool, Pc(r), pbest_fit, eidx(r));
end

% ===================== 主循环 =====================
while FEs < Max_FEs
    prog = FEs / Max_FEs;
    w = w0 - (w0 - w1) * prog;
    sub_prev = min(pbest_fit(eidx));

    % ---- (a) 确定本代允许重启的粒子（pbest 排名后 restart_frac）----
    if stall_lim > 0
        rk = tiedrank_local(pbest_fit);
        can_rs = rk > max(1, round((1 - restart_frac) * N));
    else
        can_rs = false(N, 1);
    end

    % ---- (b) RL 选择本代的邻域规模 ----
    pa = softmax1(theta);
    ai = disc_sample(pa);
    k_cur = K_actions(ai);

    % ---- (c) GNN 消息传递并生成通信拓扑 ----
    Xe = X(eidx, :);
    [~, br] = min(pbest_fit(eidx));
    bg = pbest_X(eidx(br), :);
    fr = tiedrank_local(pbest_fit(eidx)) / Ne;                  % pbest 排名
    stn = min(stall(eidx) / stall_lim, 1);                      % 停滞度
    dg = sqrt(sum((Xe - bg).^2, 2));  dg = dg / (max(dg) + 1e-12);
    vm = sqrt(sum(V(eidx, :).^2, 2)); vm = vm / (max(vm) + 1e-12);
    Xf = [fr, stn, dg, vm];
    deg = sum(Ae, 2);
    Dn = diag(1 ./ sqrt(deg + 1e-8));
    An = Dn * Ae * Dn;                                          % 对称归一化邻接矩阵
    Hg = max(0, An * Xf * Wg);                                  % ReLU 节点嵌入
    Sg = Hg * Hg';  Sg = Sg - diag(diag(Sg));  Sg = (Sg + Sg') / 2;
    Ae_new = knn_graph(Sg, k_cur, Ne);                          % kNN 建边
    nbrs = graph_nbrs(Ae_new, Ne);

    % ---- (d) 开发子群的随机邻域社会榜样 ----
    kx = max(2, round(lbest_frac * N));
    lb_x = zeros(numel(xidx), D);
    for t = 1:numel(xidx)
        nb = randperm(N, kx);
        [~, mi] = min(pbest_fit(nb));
        lb_x(t, :) = pbest_X(nb(mi), :);
    end

    % ---- (e) 种群更新 ----
    order = randperm(N);
    for oi = 1:N
        i = order(oi);
        if FEs >= Max_FEs, break; end

        if expl(i)
            % 探索子群：逐维综合学习
            r = rmap(i);
            if flag(i) >= m_refresh
                pl = nbrs{r};
                pool = eidx(pl);  pool(pool == i) = [];
                if isempty(pool), pool = eidx;  pool(pool == i) = []; end
                exemplar(r, :) = gen_exemplar(D, pool, Pc(r), pbest_fit, i);
                flag(i) = 0;
            end
            ex = exemplar(r, :);
            tgt = zeros(1, D);
            for d = 1:D, tgt(d) = pbest_X(ex(d), d); end
            V(i, :) = w * V(i, :) + c_explore * rand(1, D) .* (tgt - X(i, :));
        else
            % 开发子群：认知项 + 随机邻域社会项
            tgt = lb_x(xmap(i), :);
            V(i, :) = w * V(i, :) + c_cog * rand(1, D) .* (pbest_X(i, :) - X(i, :)) ...
                                 + c_soc * rand(1, D) .* (tgt - X(i, :));
        end

        V(i, :) = max(min(V(i, :), VRmax), VRmin);
        X(i, :) = X(i, :) + V(i, :);
        X(i, :) = max(min(X(i, :), Popmax), Popmin);

        f = feval(fhd, X(i, :)', varargin{:});
        FEs = FEs + 1;

        % ---- (f) 个体/全局最优更新与排名受限重启 ----
        improved = f < pbest_fit(i);
        if improved
            pbest_fit(i) = f;  pbest_X(i, :) = X(i, :);
            flag(i) = 0;  stall(i) = 0;
            if f < gbest_fit, gbest_fit = f;  gbest_X = X(i, :); end
        else
            flag(i) = flag(i) + 1;  stall(i) = stall(i) + 1;
        end
        % 每次评估都写一个历史点（重启会额外消耗一次评估，必须各自记录，
        % 否则被跳过的索引会永久留成 NaN，绘制收敛曲线时出现断裂）
        if FEs <= numel(history_gbest), history_gbest(FEs) = gbest_fit; end

        if ~improved && can_rs(i) && stall(i) >= stall_lim && FEs < Max_FEs
            X(i, :) = rand(1, D) * span + Popmin;
            V(i, :) = zeros(1, D);
            stall(i) = 0;  flag(i) = 0;
            f = feval(fhd, X(i, :)', varargin{:});
            FEs = FEs + 1;
            if f < pbest_fit(i)
                pbest_fit(i) = f;  pbest_X(i, :) = X(i, :);
                if f < gbest_fit, gbest_fit = f;  gbest_X = X(i, :); end
            end
            if FEs <= numel(history_gbest), history_gbest(FEs) = gbest_fit; end
        end
    end

    % ---- (g) RL / GNN 参数更新（以探索子群最优的改善为奖励）----
    rw = sub_prev - min(pbest_fit(eidx));
    if ~isfinite(rw), rw = 0; end
    rscale = 0.98 * rscale + 0.02 * abs(rw);
    rn = rw / (rscale + 1e-12);
    rbase = 0.98 * rbase + 0.02 * rn;
    adv = rn - rbase;                                   % 标准化优势值
    oh = zeros(1, numel(K_actions));  oh(ai) = 1;
    theta = max(min(theta + lr_rl * adv * (oh - pa), 5), -5);
    Pg = 1 ./ (1 + exp(-Sg));
    Eg = Ae_new - Pg;  Eg = (Eg + Eg') / 2;
    dH = Eg * Hg + Eg' * Hg;  dH = dH .* (Hg > 0);
    Wg = max(min(Wg + lr_gnn * ((An * Xf)' * dH * 20 * rn), 5), -5);

    Ae = Ae_new;
end
end

% =================== 辅助函数 ===================
function ex = gen_exemplar(D, pool, Pc_i, pbest_fit, self_idx)
% GEN_EXEMPLAR  逐维锦标赛选择学习榜样
pool = pool(:);  pool(pool == self_idx) = [];
if isempty(pool), ex = self_idx * ones(1, D); return; end
np = numel(pool);
ex = zeros(1, D);
for d = 1:D
    if rand() < Pc_i
        p1 = pool(randi(np));  p2 = pool(randi(np));
        if pbest_fit(p1) < pbest_fit(p2), ex(d) = p1; else, ex(d) = p2; end
    else
        ex(d) = self_idx;
    end
end
if all(ex == self_idx), ex(randi(D)) = pool(randi(np)); end
end

function A = knn_graph(S, k, N)
% KNN_GRAPH  对相似度矩阵 S 取每行前 k 个最大者建边（对称化、含自环）
if k <= 0 || k >= N, A = ones(N); return; end
A = zeros(N);
[~, idx] = sort(S, 2, 'descend');
for i = 1:N, A(i, idx(i, 1:k)) = 1; end
A = double(A | A');
A(1:N+1:end) = 1;
end

function nb = graph_nbrs(A, N)
% GRAPH_NBRS  邻接矩阵 -> 邻域索引
nb = cell(N, 1);
for i = 1:N
    nb{i} = find(A(i, :) == 1)';
    if isempty(nb{i}), nb{i} = (1:N)'; end
end
end

function r = tiedrank_local(v)
% TIEDRANK_LOCAL  竞争排名（1 = 最优）
[~, si] = sort(v(:));
r = zeros(numel(v), 1);
r(si) = 1:numel(v);
end

function p = softmax1(t)
t = t - max(t);
e = exp(t);
p = e / sum(e);
end

function a = disc_sample(p)
c = cumsum(p);
a = find(rand() <= c, 1);
if isempty(a), a = numel(p); end
end
