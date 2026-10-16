function out=run_screen(mode,opt)
%RUN_SCREEN  Experiment driver of the RASE-GWO study.
%
%   run_screen('smoke')   wiring/correctness check only (tiny budget).
%   run_screen('over')    oversampling family screen (same FE, same rounds).
%   run_screen('sparse')  sparse-budget family screen (20% of the FE).
%   run_screen('study')   formal run: 29 functions, 30 seeds, 100k FE.
%   run_screen(...,opt)   opt overrides any protocol field:
%       funcs, seeds, maxFE, D, algorithms, tag, workers, tracePoints
%
% Protocol (identical for every compared algorithm inside one call):
%   * one shared random point pool per (function, seed), drawn with the size
%     of the largest population in the batch; every algorithm consumes a
%     PREFIX of that pool, so paired comparisons share their starting points;
%   * the true-evaluation budget counts the initialization;
%   * results are written to results/<tag>_<timestamp>/ as runs.csv + runs.mat.
%
% Every run is independent, so the job list is executed with parfor across the
% local pool (default 4 workers).

if nargin<2, opt=struct(); end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'),fullfile(root,'data','cec2017'));
cecDir=fullfile(root,'data','cec2017');

OVER={'Over100-90','Over250-55','Over250-70','Over250-90','Over500-90','Over250-90r'};
switch mode
    case 'smoke'
        d=struct('funcs',[1 6],'seeds',3001:3002,'maxFE',3000,'D',50, ...
            'algorithms',{{'GWO','Over250-90'}},'tag','smoke','workers',4);
    case 'over'
        algs=[{'GWO','GWO-N10','DenseSync'},OVER];
        d=struct('funcs',[1 3:13],'seeds',3001:3010,'maxFE',100000,'D',50, ...
            'algorithms',{algs},'tag','over','workers',4);
    case 'sparse'
        d=struct('funcs',[1 3:13],'seeds',3001:3010,'maxFE',100000,'D',50, ...
            'algorithms',{{'GWO','Sparse50-55','Sparse50-90','Sparse50-Q'}}, ...
            'tag','sparse','workers',4,'sparseBudget',20000);
    case 'study'
        d=struct('funcs',[1 3:30],'seeds',3001:3030,'maxFE',100000,'D',50, ...
            'algorithms',{{'GWO','Over250-70','Over250-90','Over250-90r','Over250-90q'}}, ...
            'tag','study','workers',4);
    otherwise
        error('run_screen:mode','Unknown mode: %s',mode);
end
fn=fieldnames(opt);
for k=1:numel(fn), d.(fn{k})=opt.(fn{k}); end

funcs=d.funcs(:)'; seeds=d.seeds(:)'; algorithms=d.algorithms;
nF=numel(funcs); nS=numel(seeds); nA=numel(algorithms);
nRuns=nF*nS*nA;
D=d.D; maxFE=d.maxFE; tracePoints=300;
if isfield(d,'tracePoints'), tracePoints=d.tracePoints; end
sparseBudget=maxFE;
if isfield(d,'sparseBudget'), sparseBudget=d.sparseBudget; end

% population pool large enough for the widest algorithm in the batch
Npool=50;
for ai=1:nA, Npool=max(Npool,algorithmConfig(algorithms{ai}).N); end

fprintf('run_screen(%s): %d functions x %d seeds x %d algorithms = %d runs\n', ...
    mode,nF,nS,nA,nRuns);
fprintf('  D=%d  maxFE=%d  pool=%d  workers=%d\n',D,maxFE,Npool,d.workers);

% ---- flatten the job list (parfor needs a 1-D index) ----
jobFunc=zeros(nRuns,1); jobSeed=zeros(nRuns,1); jobAlg=zeros(nRuns,1);
k=0;
for ai=1:nA
    for si=1:nS
        for fi=1:nF
            k=k+1;
            jobFunc(k)=funcs(fi); jobSeed(k)=seeds(si); jobAlg(k)=ai;
        end
    end
end

traceFE=unique(round(logspace(1,log10(maxFE),tracePoints)));
traceFE=traceFE(traceFE>=1 & traceFE<=maxFE);

FinalCost=nan(nRuns,1); FinalError=nan(nRuns,1); Evaluations=nan(nRuns,1);
Rounds=nan(nRuns,1); Seconds=nan(nRuns,1); Status=cell(nRuns,1);
Message=cell(nRuns,1); ConvShareMean=nan(nRuns,1); EvalsPerRound=nan(nRuns,1);
DistinctWolves=nan(nRuns,1);
ShareEarly=nan(nRuns,1); ShareMid=nan(nRuns,1); ShareLate=nan(nRuns,1);
saveTraces=false;
if isfield(d,'saveTraces'), saveTraces=d.saveTraces; end
Traces=nan(numel(traceFE),nRuns);   % always allocated: parfor slicing

poolObj=gcp('nocreate');
if isempty(poolObj), poolObj=parpool(d.workers); end
pctRunOnAll(sprintf('addpath(''%s'',''%s'');',fullfile(root,'src'),fullfile(root,'data','cec2017')));

tic;
parfor k=1:nRuns
    fid=jobFunc(k); seed=jobSeed(k); name=algorithms{jobAlg(k)};
    cd(cecDir);
    cfg=algorithmConfig(name);
    if any(strncmp(name,{'Over','Sparse','Dense'},4)), B=sparseBudget; else, B=maxFE; end
    initSeed=261013000+fid*10000+seed;
    rng(initSeed,'twister');
    pool=-100+200*rand(50,D);   % shared 50-point prefix == 9.29 baseline protocol
    if Npool>50, pool=[pool; -100+200*rand(Npool-50,D)]; end
    st=struct('initialPopulation',pool(1:cfg.N,:), ...
        'lowerBound',-100*ones(1,D),'upperBound',100*ones(1,D), ...
        'maxEvaluations',B);
    sfn=fieldnames(cfg.state);
    for q=1:numel(sfn), st.(sfn{q})=cfg.state.(sfn{q}); end
    t0=tic;
    try
        [Best,T,info]=cfg.f(@(x)CEC2017Function(x,fid),st,[],cfg.N,seed+1000000);
        el=toc(t0);
        FinalCost(k)=Best.Cost;
        FinalError(k)=Best.Cost-100*fid;
        Evaluations(k)=info.Evaluations;
        Rounds(k)=info.Rounds;
        Seconds(k)=el;
        Status{k}='ok';
        Message{k}='';
        EvalsPerRound(k)=info.MeanEvalsPerRound;
        if isfield(info,'DistinctEvaluatedWolves'), DistinctWolves(k)=info.DistinctEvaluatedWolves; end
        if isfield(info,'QuotaTrace') && ~isempty(info.QuotaTrace)
            qt=info.QuotaTrace;
            third=[0 1/3 2/3 1]*B;
            ShareEarly(k)=mean(qt(qt(:,1)<=third(2),5));
            ShareMid(k)  =mean(qt(qt(:,1)>third(2)&qt(:,1)<=third(3),5));
            ShareLate(k) =mean(qt(qt(:,1)>third(3),5));
        end
        if saveTraces
            sub=nan(numel(traceFE),1);
            m=min(numel(traceFE),numel(T));
            sub(1:m)=T(traceFE(1:m));
            Traces(:,k)=sub;
        else
            Traces(:,k)=nan(numel(traceFE),1);
        end
        if isfield(info,'QuotaTrace') && ~isempty(info.QuotaTrace)
            ConvShareMean(k)=mean(info.QuotaTrace(:,5));
        end
    catch ME
        el=toc(t0);
        Seconds(k)=el;
        Status{k}='error';
        Message{k}=ME.message;
    end
end
elapsed=toc;

Result=table(jobFunc,jobSeed,string(algorithms(jobAlg))',FinalCost,FinalError, ...
    Evaluations,Rounds,Seconds,ConvShareMean,EvalsPerRound,DistinctWolves, ...
    ShareEarly,ShareMid,ShareLate,Status,Message, ...
    'VariableNames',{'FunctionID','Seed','Algorithm','FinalCost','FinalError', ...
    'Evaluations','Rounds','Seconds','MeanConvShare','EvalsPerRound','DistinctWolves', ...
    'ShareEarly','ShareMid','ShareLate','Status','Message'});

stamp=datestr(now,'yyyymmdd_HHMMSS');
outDir=fullfile(root,'results',sprintf('%s_%s',d.tag,stamp));
if ~exist(outDir,'dir'), mkdir(outDir); end
writetable(Result,fullfile(outDir,'runs.csv'));
Protocol=d; Protocol.mode=mode; Protocol.traceFE=traceFE; Protocol.Npool=Npool;
save(fullfile(outDir,'protocol.mat'),'Protocol');
if saveTraces
    save(fullfile(outDir,'traces.mat'),'Traces','traceFE','-v7.3');
end
fprintf('done in %.1f min\n',elapsed/60);
nErr=sum(Result.Status~="ok");
if nErr>0, fprintf('WARNING: %d runs failed.\n',nErr); end
fprintf('results: %s\n',outDir);
out=struct('Result',Result,'traceFE',traceFE,'outDir',outDir);
end
