function Result=quick_check(funcs,seeds,maxFE,algorithms,D,budgetOverride)
%QUICK_CHECK  Serial development runner (no parfor) for fast iteration.
%   Result=quick_check([1 6 9],3001:3002,100000,{'GWO','Fixed50'},50)
%
% budgetOverride (optional) is a name->budget mapping used by the constrained
% protocol, e.g. struct('Fixed50',20000).  Algorithms missing from it receive
% maxFE.  The initial point pool is drawn once per (function, seed) with the
% size of the largest population involved, so every algorithm consumes a
% PREFIX of the same pool and paired comparisons share their starting points.
if nargin<5||isempty(D),D=50;end
if nargin<6,budgetOverride=struct();end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'),fullfile(root,'data','cec2017'));
cecDir=fullfile(root,'data','cec2017'); cd(cecDir);
Npool=50;
for ai=1:numel(algorithms)
    Npool=max(Npool,algorithmConfig(algorithms{ai}).N);
end
rows={};
for fid=funcs
    for seed=seeds
        initSeed=261013000+fid*10000+seed;
        rng(initSeed,'twister'); pool=-100+200*rand(50,D);
        if Npool>50, pool=[pool; -100+200*rand(Npool-50,D)]; end
        for ai=1:numel(algorithms)
            name=algorithms{ai};
            cfg=algorithmConfig(name);
            if isfield(budgetOverride,name), B=budgetOverride.(name); else, B=maxFE; end
            st=struct('initialPopulation',pool(1:cfg.N,:), ...
                'lowerBound',-100*ones(1,D),'upperBound',100*ones(1,D), ...
                'maxEvaluations',B);
            sfn=fieldnames(cfg.state);
            for q=1:numel(sfn), st.(sfn{q})=cfg.state.(sfn{q}); end
            t0=tic;
            [Bs,Ts,ii]=cfg.f(@(x)CEC2017Function(x,fid),st,[],cfg.N,seed+1000000);
            el=toc(t0);
            fprintf('F%-2d s%d %-12s err=%12.5g rounds=%6d FE=%7d %5.2fs\n', ...
                fid,seed,name,Bs.Cost-100*fid,ii.Rounds,ii.Evaluations,el);
            rows(end+1,:)={fid,seed,name,Bs.Cost-100*fid,ii.Rounds,ii.Evaluations,el}; %#ok<AGROW>
        end
    end
end
Result=cell2table(rows,'VariableNames',{'FunctionID','Seed','Algorithm', ...
    'FinalError','Rounds','Evaluations','Seconds'});
end
