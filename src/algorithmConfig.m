function cfg=algorithmConfig(name)
%ALGORITHMCONFIG  Algorithm registry of the RASE-GWO study.
%   cfg.f      algorithm entry point, CEC harness signature
%   cfg.N      population size used by that algorithm
%   cfg.state  extra fields merged into `state` (ablation switches)
%
% Naming: Over<M>-<convergence share>   oversampled virtual population of M
%         wolves with a fixed per-round evaluation quota (K, set in state).
switch name
    % ---- baselines -------------------------------------------------------
    case 'GWO',          cfg=struct('f',@GWO,     'N',50,'state',struct());
    case 'GWO-N10',      cfg=struct('f',@GWO,     'N',10,'state',struct());
    case 'GWO-W',        cfg=struct('f',@GWO,'N',50,'state',struct('randMode','wolf'));
    case 'DenseSync',    cfg=struct('f',@RASE_GWO,'N',50,'state',struct('evalMode','dense'));
    % ---- oversampling family: M virtual wolves, K true evaluations ------
    case 'Over100-90',   cfg=rwo(100,0.9);
    case 'Over250-55',   cfg=rwo(250,0.5);
    case 'Over250-70',   cfg=rwo(250,0.7);
    case 'Over250-90',   cfg=rwo(250,0.9);
    case 'Over250-95',   cfg=rwo(250,0.95);
    case 'Over250-100',  cfg=rwo(250,1.0);
    case 'Over500-95',   cfg=rwo(500,0.95);
    case 'Over1000-90',  cfg=rwo(1000,0.9);
    case 'Over1000-95',  cfg=rwo(1000,0.95);
    case 'Over1000-100', cfg=rwo(1000,1.0);
    case 'Over1000-90r', cfg=rwo(1000,0.9,struct('screenPolicy','random'));
    case 'Over1000-90q', cfg=rwo(1000,0.9,struct('quotaPolicy','q'));
    case 'Over2000-90',  cfg=rwo(2000,0.9);
    case 'Over1000-90w', cfg=rwo(1000,0.9,struct('randMode','wolf'));
    case 'Over2000-90w', cfg=rwo(2000,0.9,struct('randMode','wolf'));
    case 'Over1500-90',  cfg=rwo(1500,0.9);
    case 'Over1000-90q2',cfg=rwo(1000,0.9,struct('quotaPolicy','q','convShares',[0.5 0.9 1.0], ...
        'qEpsStart',0.10,'qEpsEnd',0.02,'windowFE',2000,'qInit',true));
    case 'Over1000-90g', cfg=rwo(1000,0.9,struct('driftGate',true));
    case 'Over1000-90ld',cfg=rwo(1000,0.9,struct('divCriterion','leaders'));
    case 'Over1000-90min',cfg=rwo(1000,0.9,struct('divCriterion','minority'));
    case 'Over1000-80min',cfg=rwo(1000,0.8,struct('divCriterion','minority'));
    case 'Over1000-70min',cfg=rwo(1000,0.7,struct('divCriterion','minority'));
    case 'Fixed55min',   cfg=rwo(1000,0.5,struct('divCriterion','minority'));
    case 'Over1000-90nv',cfg=rwo(1000,0.9,struct('useVirtual',false,'divCriterion','minority'));
    case 'Over1000-90q3',cfg=rwo(1000,0.9,struct('quotaPolicy','q','divCriterion','minority', ...
        'convShares',[0.7 0.9 1.0],'qEpsStart',0.10,'qEpsEnd',0.02,'windowFE',2000,'qInit',true));
    case 'Over500-90',   cfg=rwo(500,0.9);
    case 'Over250-90r',  cfg=rwo(250,0.9,struct('screenPolicy','random'));
    case 'Over250-90nv', cfg=rwo(250,0.9,struct('useVirtual',false,'divCriterion','minority'));
    case 'Over250-90q',  cfg=rwo(250,0.9,struct('quotaPolicy','q'));
    % ---- sparse-budget family (legacy protocol: 10 evals per round) -----
    case 'Sparse50-55',  cfg=rsp(0.5);
    case 'Sparse50-90',  cfg=rsp(0.9);
    case 'Sparse50-Q',   cfg=rsp(0.9,struct('quotaPolicy','q'));
    otherwise
        error('algorithmConfig:unknown','Unknown algorithm: %s',name);
end
end

function cfg=rwo(M,share,extra)
% oversampled variant: M virtual wolves, K fixed evaluation slots per round
st=struct('K',50,'quotaPolicy','fixed','convShare',share);
if nargin>2
    fn=fieldnames(extra);
    for k=1:numel(fn), st.(fn{k})=extra.(fn{k}); end
end
cfg=struct('f',@RASE_GWO,'N',M,'state',st);
end

function cfg=rsp(share,extra)
% sparse-budget variant: N=50 wolves, K=round(rho*N)=10 evaluations per round
st=struct('rho',0.2,'quotaPolicy','fixed','convShare',share);
if nargin>1
    fn=fieldnames(extra);
    for k=1:numel(fn), st.(fn{k})=extra.(fn{k}); end
end
cfg=struct('f',@RASE_GWO,'N',50,'state',st);
end
