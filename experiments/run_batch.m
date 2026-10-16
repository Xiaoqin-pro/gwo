function out=run_batch(mode,opt)
%RUN_BATCH  Incremental experiment driver: every finished run is written to disk
%           immediately, so progress can be watched and a crashed batch keeps
%           everything that was already computed.
%
%   run_batch('smoke')            wiring check (tiny budget)
%   run_batch('over')             development screen on F1,F3..F13
%   run_batch('formal')           main study, 29 functions x 30 seeds
%   run_batch(...,opt)            opt overrides: funcs, seeds, maxFE, D,
%                                 algorithms, tag, workers, saveTraces,
%                                 probeEvery, resume
%
% Layout of a batch directory:
%   runs/run_00001.csv ...     one file per finished run (written by the worker)
%   runs.csv                   concatenation of the above (rebuilt every call)
%   progress.log               "done <n>/<total>" lines, rewritten as runs land
%   protocol.mat, traces.mat
%
% Resuming: pass opt.resume = '<batch directory>' to skip run indices that
% already have a run_*.csv file and continue an interrupted batch.
%
% Protocol (identical for every algorithm inside one call):
%   * one shared random point pool per (function, seed); algorithm A consumes the
%     first N_A rows, so paired comparisons share their starting points;
%   * the true-evaluation budget includes initialization;
%   * two modes: 'equalFE' gives every algorithm maxFE; 'sparseBudget' gives the
%     RASE family `sparseBudget` evaluations instead.

if nargin<2, opt=struct(); end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'),fullfile(root,'data','cec2017'));
cecDir=fullfile(root,'data','cec2017');

OVER={'Over1000-90q3','Fixed55min','Over1000-90min','Over1000-100', ...
      'Over1000-90r','Over1000-90nv'};
switch mode
    case 'smoke'
        d=struct('funcs',[1 6],'seeds',3001:3002,'maxFE',3000,'D',50, ...
            'algorithms',{{'GWO','Over1000-90q3'}},'tag','smoke','workers',4);
    case 'over'
        algs=[{'GWO','GWO-N10','DenseSync'}, ...
              {'Over100-90','Over250-90','Over500-90','Over1000-90','Over2000-90'}];
        d=struct('funcs',[1 3:13],'seeds',3001:3010,'maxFE',100000,'D',50, ...
            'algorithms',{algs},'tag','over','workers',4);
    case 'formal'
        d=struct('funcs',[1 3:30],'seeds',3001:3030,'maxFE',100000,'D',50, ...
            'algorithms',{OVER},'tag','formal','workers',4,'saveTraces',true);
    case 'external'
        % GWO family on the 29-function grid, ready to merge with the 9.29 batch
        d=struct('funcs',[1 3:30],'seeds',3001:3030,'maxFE',100000,'D',50, ...
            'algorithms',{{'GWO','GWO-N10','DenseSync'}},'tag','external', ...
            'workers',4,'saveTraces',true);
    otherwise
        error('run_batch:mode','Unknown mode: %s',mode);
end
fn=fieldnames(opt);
for k=1:numel(fn), d.(fn{k})=opt.(fn{k}); end

funcs=d.funcs(:)'; seeds=d.seeds(:)'; algorithms=d.algorithms;
nF=numel(funcs); nS=numel(seeds); nA=numel(algorithms);
nRuns=nF*nS*nA;
D=d.D; maxFE=d.maxFE;
saveTraces=false; if isfield(d,'saveTraces'), saveTraces=logical(d.saveTraces); end
tracePoints=300; if isfield(d,'tracePoints'), tracePoints=d.tracePoints; end

% population pool wide enough for the largest algorithm in the batch
Npool=50;
for ai=1:nA, Npool=max(Npool,algorithmConfig(algorithms{ai}).N); end

% ---- batch directory ---------------------------------------------------
if isfield(d,'resume') && ~isempty(d.resume)
    outDir=char(d.resume);
    fprintf('RESUMING batch %s\n',outDir);
else
    stamp=datestr(now,'yyyymmdd_HHMMSS');
    outDir=fullfile(root,'results',sprintf('%s_%s',d.tag,stamp));
    mkdir(outDir);
end
runDir=fullfile(outDir,'runs');
if ~exist(runDir,'dir'), mkdir(runDir); end
logFile=fullfile(outDir,'progress.log');

traceFE=unique(round(logspace(1,log10(maxFE),tracePoints)));
traceFE=traceFE(traceFE>=1 & traceFE<=maxFE);
nTrace=numel(traceFE);

% ---- flatten the job list ---------------------------------------------
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

done=false(nRuns,1);
for k=1:nRuns
    if exist(fullfile(runDir,sprintf('run_%05d.csv',k)),'file'), done(k)=true; end
end
todo=find(~done);
fprintf('run_batch(%s): %d functions x %d seeds x %d algorithms = %d runs\n', ...
    mode,nF,nS,nA,nRuns);
fprintf('  D=%d  maxFE=%d  pool=%d  workers=%d  already done=%d\n', ...
    D,maxFE,Npool,d.workers,sum(done));
fprintf('  batch: %s\n',outDir);

poolObj=gcp('nocreate');
if isempty(poolObj), poolObj=parpool(d.workers); end
pctRunOnAll(sprintf('addpath(''%s'',''%s'');',fullfile(root,'src'),fullfile(root,'data','cec2017')));

% ---- run -------------------------------------------------------------------
tic;
parfor jj=1:numel(todo)
    k=todo(jj);
    fid=jobFunc(k); seed=jobSeed(k); name=algorithms{jobAlg(k)};
    cd(cecDir);
    cfg=algorithmConfig(name);
    B=maxFE;
    if isfield(d,'sparseBudget') && any(strncmp(name,{'Over','Sparse','Dense','Fixed'},4))
        B=d.sparseBudget;
    end
    initSeed=261013000+fid*10000+seed;
    rng(initSeed,'twister');
    pool=-100+200*rand(50,D);        % shared 50-point prefix (9.29 protocol)
    if Npool>50, pool=[pool; -100+200*rand(Npool-50,D)]; end
    st=struct('initialPopulation',pool(1:cfg.N,:), ...
        'lowerBound',-100*ones(1,D),'upperBound',100*ones(1,D), ...
        'maxEvaluations',B);
    sfn=fieldnames(cfg.state);
    for q=1:numel(sfn), st.(sfn{q})=cfg.state.(sfn{q}); end
    t0=tic;
    row=table(fid,seed,string(name),NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN, ...
        NaN,NaN,NaN,"pending","",'VariableNames', ...
        {'FunctionID','Seed','Algorithm','FinalCost','FinalError','Evaluations', ...
         'Rounds','Seconds','MeanConvShare','EvalsPerRound','DistinctWolves', ...
         'ShareEarly','ShareMid','ShareLate','RedFrac','Status','Message'});
    try
        [Best,T,info]=cfg.f(@(x)CEC2017Function(x,fid),st,[],cfg.N,seed+1000000);
        el=toc(t0);
        row.FinalCost=Best.Cost;
        row.FinalError=Best.Cost-100*fid;
        row.Evaluations=info.Evaluations;
        row.Rounds=info.Rounds;
        row.Seconds=el;
        row.EvalsPerRound=info.MeanEvalsPerRound;
        if isfield(info,'DistinctEvaluatedWolves'), row.DistinctWolves=info.DistinctEvaluatedWolves; end
        if isfield(info,'RedFrac') && ~isempty(info.RedFrac), row.RedFrac=mean(info.RedFrac); end
        if isfield(info,'QuotaTrace') && ~isempty(info.QuotaTrace)
            qt=info.QuotaTrace;
            t3=[0 1/3 2/3 1]*B;
            row.MeanConvShare=mean(qt(:,5));
            row.ShareEarly=mean(qt(qt(:,1)<=t3(2),5));
            row.ShareMid=mean(qt(qt(:,1)>t3(2)&qt(:,1)<=t3(3),5));
            row.ShareLate=mean(qt(qt(:,1)>t3(3),5));
        end
        row.Status="ok";
        % per-run trajectory, stored beside the row so it survives a crash
        tr=nan(nTrace,1);
        m=min(nTrace,numel(T));
        tr(1:m)=T(traceFE(1:m));
        % NOTE: save() is illegal inside parfor (workspace transparency);
        % writematrix is not, so trajectories are stored as CSV.
        writematrix(tr,fullfile(runDir,sprintf('trace_%05d.csv',k)));
    catch ME
        row.Seconds=toc(t0);
        row.Status="error";
        row.Message=string(ME.identifier)+": "+string(ME.message);
    end
    writetable(row,fullfile(runDir,sprintf('run_%05d.csv',k)));
end
elapsed=toc;

% ---- rebuild the aggregate view ---------------------------------------
files=dir(fullfile(runDir,'run_*.csv'));
Parts=cell(numel(files),1);
for i=1:numel(files)
    Parts{i}=readtable(fullfile(runDir,files(i).name),'TextType','string');
end
Result=vertcat(Parts{:});
[~,ord]=sortrows([Result.FunctionID Result.Seed],[1 2]);
Result=Result(ord,:);
writetable(Result,fullfile(outDir,'runs.csv'));

if saveTraces
    Traces=nan(numel(traceFE),nRuns);
    for k=1:nRuns
        f=fullfile(runDir,sprintf('trace_%05d.csv',k));
        if exist(f,'file'), Traces(:,k)=readmatrix(f); end
    end
    save(fullfile(outDir,'traces.mat'),'Traces','traceFE','-v7');
end

fid_=fopen(logFile,'a');
fprintf(fid_,'done %d/%d  (this call: %d new runs, %.1f min)\n', ...
    numel(files),nRuns,numel(todo),elapsed/60);
fclose(fid_);

Protocol=d; Protocol.mode=mode; Protocol.traceFE=traceFE; Protocol.Npool=Npool;
save(fullfile(outDir,'protocol.mat'),'Protocol');
nErr=sum(Result.Status=="error");
fprintf('batch done: %d runs on disk (%d errors) in %.1f min\n', ...
    height(Result),nErr,elapsed/60);
fprintf('results: %s\n',outDir);
out=struct('Result',Result,'outDir',outDir,'traceFE',traceFE);
end
