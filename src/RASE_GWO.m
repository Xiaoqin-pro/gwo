function [Best,T,info]=RASE_GWO(f,state,~,N,seed)
%RASE_GWO  Resource-Aware Sparse-Evaluation Grey Wolf Optimizer.
%
%   [Best,T,info]=RASE_GWO(f,state,[],N,seed)
%
% Research question addressed by this file:
%   "When a GWO can generate many candidate positions but only a few of them
%    can be verified by an expensive true objective function, which wolves
%    deserve the evaluation?"
%
% The search equations are the published ones.  What changes is the DECOUPLING
% of candidate generation from true evaluation, and the allocation of the
% scarce true-evaluation budget.
%
%   (1) Dual-state wolves.  Every wolf carries a virtual (search) position
%       Y_i that moves every round and costs nothing, plus a verified position
%       Xv_i that is the most recent position actually confirmed by f.  Only
%       verified positions may compete for the alpha/beta/delta hierarchy, so
%       unevaluated movement can never contaminate the leadership.
%   (2) Sparse evaluation.  Each round moves all N wolves but spends only
%       K = round(rho*N) true evaluations (default rho = 0.2, N = 50, K = 10).
%       At an unchanged true-FE budget this buys ~N/K times more generations.
%   (3) Dual-potential geometric screening.  The K slots are split between the
%       Kc wolves closest to the leader centre XL = (Xa+Xb+Xd)/3 (convergence
%       potential) and the Kd wolves farthest from the virtual population
%       centroid (diversity potential).  No surrogate model is built.
%   (4) Q-learning quota scheduler (optional).  A 6-state / 3-action agent
%       reallocates Kc:Kd between 5:5, 7:3 and 9:1 on a reward that is the
%       scale-invariant gbest improvement per decision window.  It schedules
%       the EVALUATION BUDGET, not the search operator.
%
% Ablation switches (all overridable through fields of the same name on
% `state`; see raseDefaults for values):
%   evalMode      'sparse' | 'dense'        dense = full-evaluation control
%   screenPolicy  'geometry' | 'random'     random = screening removed
%   useVirtual    true | false              false = leadership reads the
%                                           unverified search positions
%   verifiedMode  'latest' | 'best'         'best' = archive-style update
%   quotaPolicy   'fixed' | 'q' | 'random'  'random' = same-gate random quota
%   convShare     scalar for 'fixed'        ablation grid 0.5 / 0.7 / 0.9 / 1.0
%
% Required state fields: initialPopulation, lowerBound, upperBound,
% maxEvaluations.  Initial population generation belongs to the caller.
%
% Outputs: Best.Cost/Best.Vector, T (budget-by-1 best-so-far per true FE),
% info.Evaluations / Rounds / EvalsPerRound / QuotaTrace / RewardTrace /
% QTable / QVisits / MeanEvalsPerRound / WallClock / Variant.
%
% Randomness: rng(seed,'twister') drives the search stream; the screening and
% the quota policy draw from their own RandStream objects so that switching an
% ablation switch never perturbs the search trajectory of the other variants.

p=raseDefaults();
fn=fieldnames(p);
for k=1:numel(fn)
    if isfield(state,fn{k}), p.(fn{k})=state.(fn{k}); end
end
[Best,T,info]=raseFramework(f,state,p,N,seed);
info.Variant=raseVariantName(p);
end

% ===================== default parameter set =====================
function p=raseDefaults()
p.evalMode='sparse';        % 'sparse' | 'dense'
p.rho=0.2;                  % evaluation ratio: K = round(rho*N)
p.screenPolicy='geometry';  % 'geometry' | 'random'
p.divCriterion='centroid';  % 'centroid' | 'leaders' | 'minority'
p.driftGate=false;          % only evaluate wolves that moved since their last evaluation
p.driftGateFrac=0.10;      % gate threshold, as a fraction of the swarm spread
p.randMode='dim';          % 'dim' = per-wolf per-dim coefficients (classic GWO)
                            % 'wolf' = one scalar coefficient pair per wolf,
                            %          broadcast over dimensions (cheap)
p.useVirtual=true;          % dual-state leadership
p.verifiedMode='latest';    % 'latest' | 'best'
p.quotaPolicy='fixed';      % 'fixed' | 'q' | 'random'
p.convShare=0.5;            % used by 'fixed'
p.convShares=[0.5 0.7 0.9]; % action set of the Q scheduler
p.windowFE=1000;            % Q decision window, in true evaluations
p.stagThreshold=1000;       % stagnant after this many true FE without gbest gain
p.qAlpha=0.1; p.qGamma=0.9; p.qEpsStart=0.3; p.qEpsEnd=0.05;
p.qInit=false;                % optimistic initialisation towards exploitation-heavy shares
p.qInitScale=1;
p.K=[];                     % absolute per-round quota; empty = round(rho*N)
p.fillToQuota=true;         % top up to exactly K evals per round
p.screenRngOffset=4000001;
p.quotaRngOffset=3000001;
end

function name=raseVariantName(p)
switch p.quotaPolicy
    case 'q',      name=sprintf('RASE-GWO(rho=%.2f,Q)',p.rho);
    case 'random', name=sprintf('RASE-GWO(rho=%.2f,randquota)',p.rho);
    otherwise,     name=sprintf('RASE-GWO(rho=%.2f,fixed%.2f)',p.rho,p.convShare);
end
if strcmp(p.evalMode,'dense'), name='Dense-Sync-GWO'; end
if strcmp(p.screenPolicy,'random') && ~strcmp(p.evalMode,'dense')
    name=sprintf('RandomSparse-GWO(rho=%.2f)',p.rho);
end
end

% ===================== framework =====================
function [Best,T,info]=raseFramework(f,state,p,N,seed)
clockStart=tic;
pop=state.initialPopulation;
[N0,D]=size(pop);
assert(N0==N,'Population size mismatch: state holds %d rows, N=%d.',N0,N);
lb=state.lowerBound; ub=state.upperBound; budget=state.maxEvaluations;
assert(numel(lb)==D && numel(ub)==D,'Bound vectors must match D=%d.',D);
assert(budget>=N && budget==floor(budget),'Budget must be an integer >= N.');
assert(all(isfinite(pop),'all') && all(pop>=lb & pop<=ub,'all'), ...
    'Initial population must be finite and inside the bounds.');
rng(seed,'twister');

isDense=strcmp(p.evalMode,'dense');
isQ=strcmp(p.quotaPolicy,'q');
isRandomQuota=strcmp(p.quotaPolicy,'random');
if isDense
    p.screenPolicy='geometry'; p.quotaPolicy='fixed';
    isQ=false; isRandomQuota=false;
end
if isQ, assert(strcmp(p.quotaPolicy,'q')); end
if isDense
    K=N;
elseif ~isempty(p.K)
    K=max(1,min(N,round(p.K)));
else
    K=max(1,min(N,round(p.rho*N)));
end
assert(numel(p.convShares)>=1 && all(p.convShares>=0 & p.convShares<=1));

Y=pop;                 % virtual  (search) positions
evalCount=zeros(N,1);   % diagnostic: how many times each wolf was evaluated
lastEvalY=Y;            % virtual position at the moment of the last evaluation
Xv=pop;                % verified (real) positions
fv=inf(N,1);
T=nan(budget,1);
fe=0; gbest=inf; gbestX=pop(1,:); lastImproveFE=N;
for i=1:min(K,N)
    v=f(Y(i,:));
    fe=fe+1;
    fv(i)=v; Xv(i,:)=Y(i,:); evalCount(i)=evalCount(i)+1; lastEvalY(i,:)=Y(i,:);
    if v<gbest, gbest=v; gbestX=Y(i,:); lastImproveFE=fe; end
    T(fe)=gbest;
end

% dedicated streams: ablations must not perturb the search trajectory
screenStream=makeStream(seed,p.screenRngOffset);
quotaStream =makeStream(seed,p.quotaRngOffset);

% quota scheduler state
stateCount=6;
if isQ
    actionCount=numel(p.convShares);
    Q=zeros(stateCount,actionCount); QVisits=zeros(stateCount,actionCount);
    if p.qInit, Q=repmat(p.convShares(:)',stateCount,1)*p.qInitScale; end
    sIdx=raseStateIndex(fe,budget,fe-lastImproveFE,p);
    aIdx=raseEpsAction(Q,sIdx,p.qEpsStart,quotaStream);
else
    actionCount=1; Q=[]; QVisits=[]; sIdx=1; aIdx=1;
end
lastQFE=fe; fWin=gbest;
rewardTrace=zeros(0,4);      % [fe, actionIdx, reward, nextState]
quotaTrace=zeros(0,5);       % [fe, actionIdx, Kc, Kd, share]

% leadership initialisation (verified information only)
[Xa,Xb,Xd]=raseLeaders(Xv,Y,fv,p);
rounds=0; evalsPerRound=zeros(0,1);

% ===================== main loop =====================
while fe<budget
    rounds=rounds+1;

    % ---- (a) GWO movement: all N wolves, zero true evaluations ----
    aC=2-2*(fe/budget);
    if strcmp(p.randMode,'wolf')
        A1=2*aC*rand(N,1)-aC; C1=2*rand(N,1);
        A2=2*aC*rand(N,1)-aC; C2=2*rand(N,1);
        A3=2*aC*rand(N,1)-aC; C3=2*rand(N,1);
    else
        A1=2*aC*rand(N,D)-aC; C1=2*rand(N,D);
        A2=2*aC*rand(N,D)-aC; C2=2*rand(N,D);
        A3=2*aC*rand(N,D)-aC; C3=2*rand(N,D);
    end
    Y=( (Xa-A1.*abs(C1.*Xa-Y)) + (Xb-A2.*abs(C2.*Xb-Y)) + (Xd-A3.*abs(C3.*Xd-Y)) )/3;
    Y=min(max(Y,lb),ub);

    % ---- (b) quota for this round ----
    if isQ
        share=p.convShares(aIdx);
    elseif isRandomQuota
        share=p.convShares(1+floor(rand(quotaStream)*numel(p.convShares)));
    else
        share=p.convShare;
    end
    Kc=round(share*K); Kc=max(0,min(K,Kc)); Kd=K-Kc;

    % ---- (c) screening: choose the K wolves that consume true evaluations ----
    if isDense
        E=1:N;
    elseif strcmp(p.screenPolicy,'random')
        E=randperm(screenStream,N,K);
    else
        XL=(Xa+Xb+Xd)/3;
        Cent=mean(Y,1);
        dc=sqrt(sum((Y-XL).^2,2));            % convergence potential (small = good)
        switch p.divCriterion
            case 'leaders'
                dd=min([sqrt(sum((Y-Xa).^2,2)),sqrt(sum((Y-Xb).^2,2)), ...
                        sqrt(sum((Y-Xd).^2,2))],[],2);   % far from every leader
            case 'minority'
                % probe around the leader that is farthest from the consensus, so a
                % second/third basin is kept alive instead of collapsing onto alpha
                cand=[Xa;Xb;Xd];
                [~,mi]=max(sqrt(sum((cand-XL).^2,2)));
                dd=sqrt(sum((Y-cand(mi,:)).^2,2));   % small = close to the minority leader
            otherwise
                dd=sqrt(sum((Y-Cent).^2,2));      % diversity potential (large = good)
        end
        elig=true(N,1);
        if p.driftGate && nnz(evalCount)>0
            drift=sqrt(sum((Y-lastEvalY).^2,2));
            thr=p.driftGateFrac*mean(dd);
            elig=drift>thr;
        end
        [~,cOrd]=sortrows([~elig,dc]);
        if strcmp(p.divCriterion,'minority')
            [~,dOrd]=sortrows([~elig,dd]);      % closest to the minority leader
        else
            [~,dOrd]=sortrows([~elig,-dd]);
        end
        E=[cOrd(1:min(Kc,N)); dOrd(1:min(Kd,N))];
        E=unique(E,'stable')';
        if p.fillToQuota && numel(E)<K
            ci=Kc+1; di=Kd+1; takeConv=true;
            while numel(E)<K && (ci<=N || di<=N)
                if takeConv
                    while ci<=N && any(E==cOrd(ci)), ci=ci+1; end
                    if ci<=N, E(end+1)=cOrd(ci); ci=ci+1; end %#ok<AGROW>
                else
                    while di<=N && any(E==dOrd(di)), di=di+1; end
                    if di<=N, E(end+1)=dOrd(di); di=di+1; end %#ok<AGROW>
                end
                takeConv=~takeConv;
            end
        end
    end

    % ---- (d) true evaluations and verified-state update ----
    nEval=0;
    for t=1:numel(E)
        if fe>=budget, break; end
        i=E(t);
        v=f(Y(i,:));
        fe=fe+1; nEval=nEval+1; evalCount(i)=evalCount(i)+1; lastEvalY(i,:)=Y(i,:);
        if strcmp(p.verifiedMode,'best')
            if v<fv(i), Xv(i,:)=Y(i,:); fv(i)=v; end
        else
            Xv(i,:)=Y(i,:); fv(i)=v;
        end
        if v<gbest, gbest=v; gbestX=Y(i,:); lastImproveFE=fe; end
        T(fe)=gbest;
    end
    evalsPerRound(end+1,1)=nEval; %#ok<AGROW>

    % ---- (e) leadership refresh from verified information ----
    [Xa,Xb,Xd]=raseLeaders(Xv,Y,fv,p);

    % ---- (f) Q-learning quota update (window boundary) ----
    quotaTrace(end+1,:)=[fe,aIdx,Kc,Kd,share]; %#ok<AGROW>
    if isQ && (fe-lastQFE)>=p.windowFE
        if gbest<fWin
            rew=10*(fWin-gbest)/(abs(fWin)+abs(gbest)+1e-12);
        else
            rew=-1;
        end
        sNext=raseStateIndex(fe,budget,fe-lastImproveFE,p);
        Q(sIdx,aIdx)=Q(sIdx,aIdx)+p.qAlpha*(rew+p.qGamma*max(Q(sNext,:))-Q(sIdx,aIdx));
        QVisits(sIdx,aIdx)=QVisits(sIdx,aIdx)+1;
        rewardTrace(end+1,:)=[fe,aIdx,rew,sNext]; %#ok<AGROW>
        progress=fe/budget;
        epsNow=p.qEpsStart-(p.qEpsStart-p.qEpsEnd)*progress;
        aIdx=raseEpsAction(Q,sNext,epsNow,quotaStream);
        sIdx=sNext; lastQFE=fe; fWin=gbest;
    end
end

Best.Cost=gbest; Best.Vector=gbestX;
info.Evaluations=fe;
info.Rounds=rounds;
info.EvalsPerRound=evalsPerRound;
info.MeanEvalsPerRound=mean(evalsPerRound);
info.QuotaTrace=quotaTrace;
info.RewardTrace=rewardTrace;
info.QTable=Q; info.QVisits=QVisits;
info.ConvShares=p.convShares;
info.EvalRatio=p.rho;
info.ScreenPolicy=p.screenPolicy;
info.QuotaPolicy=p.quotaPolicy;
info.UseVirtual=p.useVirtual;
info.EvalCount=evalCount;
info.DistinctEvaluatedWolves=sum(evalCount>0);
info.WallClock=toc(clockStart);
info.Algorithm='RASE-GWO';
end

% ===================== helpers =====================
function [Xa,Xb,Xd]=raseLeaders(Xv,Y,fv,p)
[~,ord]=sort(fv);
m=min(3,numel(ord));
idx=ord(1:m);
if numel(idx)<3, idx=[idx; repmat(idx(end),3-numel(idx),1)]; end
if p.useVirtual
    Xa=Xv(idx(1),:); Xb=Xv(idx(2),:); Xd=Xv(idx(3),:);
else
    Xa=Y(idx(1),:);  Xb=Y(idx(2),:);  Xd=Y(idx(3),:);
end
end

function s=raseStateIndex(fe,budget,stagAge,p)
progress=fe/budget;
ph=ceil(3*progress); ph=max(1,min(3,ph));
stag=double(stagAge>p.stagThreshold);
s=(ph-1)*2+stag+1;
end

function a=raseEpsAction(Q,s,epsNow,stream)
if rand(stream)<epsNow
    a=1+floor(rand(stream)*size(Q,2));
else
    [~,a]=max(Q(s,:));
end
a=max(1,min(size(Q,2),a));
end

function st=makeStream(seed,offset)
st=RandStream('mt19937ar','Seed',rem(seed+offset,4294967295));
end

