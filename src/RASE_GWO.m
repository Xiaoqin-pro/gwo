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
p.movePolicy='classic';     % 'classic' | 'noC' (Tsai & Shi 2024 correction)
p.refinePolicy='none';      % 'none' | 'knn'  (two-stage fitness-aware screen)
p.refinePool=4;             % stage-1 superset size = refinePool*K
p.knnE=10;                  % verified elite positions used by the k-NN estimate
p.mixFrac=0;                % share of the virtual pool on the DE family
p.familyShare=1.0;          % fixed share of the evaluation quota to family 1
p.familyPolicy='fixed';     % 'fixed' | 'q'
p.familyShares=[0.9 0.7 0.5 0.3];   % Q action set for the family split
p.eliteAnchor=false;        % pin the best-ever verified position as alpha
p.microBatches=1;           % >1: spend the quota in this many sequential
                            % batches, re-screening after every batch
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
fam=ones(N,1);          % 1 = GWO movement, 2 = DE/current-to-best movement
nF2=round(p.mixFrac*N);
if nF2>0, fam(1:nF2)=2; end
famImprove=zeros(2,1);  % evaluations that improved the best verified fitness
evalCount=zeros(N,1);   % diagnostic: how many times each wolf was evaluated
lastEvalY=Y;            % virtual position at the moment of the last evaluation
Xv=pop;                % verified (real) positions
fv=inf(N,1);
T=nan(budget,1);
fe=0; gbest=inf; gbestX=pop(1,:); lastImproveFE=N;
EA=struct('Cost',inf,'Vector',zeros(1,D));   % best verified position ever seen
for i=1:min(K,N)
    v=f(Y(i,:));
    fe=fe+1;
    fv(i)=v; Xv(i,:)=Y(i,:); evalCount(i)=evalCount(i)+1; lastEvalY(i,:)=Y(i,:);
    if v<gbest, gbest=v; gbestX=Y(i,:); lastImproveFE=fe; end
    if v<EA.Cost, EA=struct('Cost',v,'Vector',Y(i,:)); end
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
    if strcmp(p.familyPolicy,'q')
        Q2=p.qInit*ones(stateCount,numel(p.familyShares));
        Q2Visits=zeros(stateCount,numel(p.familyShares));
        famIdx=raseEpsAction(Q2,sIdx,p.qEpsStart,quotaStream);
    else
        Q2=[]; Q2Visits=[]; famIdx=1;
    end
    famShare=p.familyShare;
else
    actionCount=1; Q=[]; QVisits=[]; sIdx=1; aIdx=1;
end
lastQFE=fe; fWin=gbest;
rewardTrace=zeros(0,4);      % [fe, actionIdx, reward, nextState]
quotaTrace=zeros(0,5);       % [fe, actionIdx, Kc, Kd, share]

% leadership initialisation (verified information only)
[Xa,Xb,Xd]=raseLeaders(Xv,Y,fv,p,EA);
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
    if strcmp(p.movePolicy,'noC')
        Yg=( (Xa-A1.*abs(Xa-Y)) + (Xb-A2.*abs(Xb-Y)) + (Xd-A3.*abs(Xd-Y)) )/3;
    else
        Yg=( (Xa-A1.*abs(C1.*Xa-Y)) + (Xb-A2.*abs(C2.*Xb-Y)) + (Xd-A3.*abs(C3.*Xd-Y)) )/3;
    end
    if nF2>0
        % family 2: DE/current-to-best/1, dimension-wise
        idx2=find(fam==2);
        na=randi(N,numel(idx2),1);
        na(na==idx2)=mod(na(na==idx2),N)+1;
        nb=randi(N,numel(idx2),1);
        nb(nb==idx2 | nb==na)=mod(nb(nb==idx2 | nb==na)+1,N)+1;
        Y(idx2,:)=Y(idx2,:)+rand(numel(idx2),D).*(Xa-Y(idx2,:)) ...
                              +rand(numel(idx2),D).*(Y(na,:)-Y(nb,:));
    end
    Y(fam==1,:)=Yg(fam==1,:);
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
    if strcmp(p.familyPolicy,'q'), famShare=p.familyShares(famIdx);
    else, famShare=p.familyShare; end
    famShare=max(0,min(1,famShare));

    % ---- (c,d,e) screening, true evaluation, leadership refresh ----
    % The quota may be spent in `microBatches` sequential batches.  After every
    % batch the leaders are refreshed from the newly verified positions and the
    % still unevaluated candidates are re-screened against the updated geometry,
    % so the later evaluations of a round already exploit the information of the
    % earlier ones.  microBatches=1 reproduces the plain round exactly.
    nB=1;
    if ~isDense, nB=max(1,min(p.microBatches,K)); end
    base=floor(K/nB); remK=K-base*nB;
    nEval=0; done=false(N,1); doneCount=0;
    for b=1:nB
        Kb=base+(b<=remK);
        if Kb<=0 || fe>=budget, continue; end
        kcb=round(share*Kb); kcb=max(0,min(Kb,kcb)); kdb=Kb-kcb;
        if nF2>0
            % split this batch between the two candidate generators
            Kb1=round(famShare*Kb); Kb1=max(0,min(Kb,Kb1)); Kb2=Kb-Kb1;
            E=[];
            if Kb1>0
                k1c=round(share*Kb1);
                E=[E, raseScreen(Y,Xa,Xb,Xd,k1c,Kb1-k1c,done|(fam~=1),p, ...
                     screenStream,isDense,strcmp(p.screenPolicy,'random'),fv,Xv)];
            end
            if Kb2>0
                k2c=round(share*Kb2);
                E=[E, raseScreen(Y,Xa,Xb,Xd,k2c,Kb2-k2c,done|(fam~=2),p, ...
                     screenStream,isDense,strcmp(p.screenPolicy,'random'),fv,Xv)];
            end
        else
            E=raseScreen(Y,Xa,Xb,Xd,kcb,kdb,done,p,screenStream, ...
                         isDense,strcmp(p.screenPolicy,'random'),fv,Xv);
        end
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
            if v<EA.Cost
                EA=struct('Cost',v,'Vector',Y(i,:));
                famImprove(fam(i))=famImprove(fam(i))+1;
            end
            T(fe)=gbest;
        end
        done(E)=true; doneCount=doneCount+numel(E);
        [Xa,Xb,Xd]=raseLeaders(Xv,Y,fv,p,EA);
    end
    evalsPerRound(end+1,1)=nEval; %#ok<AGROW>

    % ---- (f) Q-learning quota update (window boundary) ----
    quotaTrace(end+1,:)=[fe,aIdx,Kc,Kd,share]; %#ok<AGROW>
    if isQ && (fe-lastQFE)>=p.windowFE
        if gbest<fWin
            rew=10*(fWin-gbest)/(abs(fWin)+abs(gbest)+1e-12);
        else
            rew=-1;
        end
        sNext=raseStateIndex(fe,budget,fe-lastImproveFE,p);
        sIdx_prev_fam=sIdx;
        Q(sIdx,aIdx)=Q(sIdx,aIdx)+p.qAlpha*(rew+p.qGamma*max(Q(sNext,:))-Q(sIdx,aIdx));
        QVisits(sIdx,aIdx)=QVisits(sIdx,aIdx)+1;
        rewardTrace(end+1,:)=[fe,aIdx,rew,sNext]; %#ok<AGROW>
        progress=fe/budget;
        epsNow=p.qEpsStart-(p.qEpsStart-p.qEpsEnd)*progress;
        aIdx=raseEpsAction(Q,sNext,epsNow,quotaStream);
        if strcmp(p.familyPolicy,'q')
            Q2(sIdx_prev_fam,famIdx)=Q2(sIdx_prev_fam,famIdx)+p.qAlpha* ...
                (rew+p.qGamma*max(Q2(sNext,:))-Q2(sIdx_prev_fam,famIdx));
            Q2Visits(sIdx_prev_fam,famIdx)=Q2Visits(sIdx_prev_fam,famIdx)+1;
            famIdx=raseEpsAction(Q2,sNext,epsNow,quotaStream);
        end
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
info.RefinePolicy=p.refinePolicy;
info.FamilyImprove=famImprove;
info.MixFrac=p.mixFrac;
info.FamilyShare=famShare;
info.FamilyQ=Q2;
info.FamilyQVisits=Q2Visits;
info.EvalCount=evalCount;
info.DistinctEvaluatedWolves=sum(evalCount>0);
info.WallClock=toc(clockStart);
info.Algorithm='RASE-GWO';
end

% ===================== helpers =====================
function E=raseScreen(Y,Xa,Xb,Xd,Kc,Kd,excluded,p,stream,isDense,isRandom,fv,Xv)
%RASESCREEN  Select the candidates that consume true evaluations this batch.
%   Candidates already evaluated in the current round (`excluded`) are skipped,
%   so a micro-batched round never spends two evaluations on the same wolf.
N=size(Y,1);
if isDense
    E=1:N; return;
end
avail=find(~excluded);
if isempty(avail), E=[]; return; end
nA=numel(avail);
if isRandom
    if nA<=Kc+Kd
        E=avail(:)';
    else
        E=avail(randperm(stream,Kc+Kd));
    end
    return;
end
Ya=Y(avail,:);
XL=(Xa+Xb+Xd)/3;
Cent=mean(Y,1);
dc=sqrt(sum((Ya-XL).^2,2));              % convergence potential (small = good)
switch p.divCriterion
    case 'leaders'
        dd=min([sqrt(sum((Ya-Xa).^2,2)),sqrt(sum((Ya-Xb).^2,2)), ...
                sqrt(sum((Ya-Xd).^2,2))],[],2);          % far from every leader
    case 'minority'
        cand=[Xa;Xb;Xd];
        [~,mi]=max(sqrt(sum((cand-XL).^2,2)));
        dd=sqrt(sum((Ya-cand(mi,:)).^2,2));              % close to the minority leader
    otherwise
        dd=sqrt(sum((Ya-Cent).^2,2));                    % diversity potential (large = good)
end
elig=true(nA,1);
if p.driftGate && nnz(~excluded)>0
    drift=sqrt(sum((Ya-Y(excluded,:)).^2,2));
    elig=drift>p.driftGateFrac*mean(dd);
end
[~,cOrd]=sortrows([~elig,dc]);
if strcmp(p.divCriterion,'minority')
    [~,dOrd]=sortrows([~elig,dd]);
else
    [~,dOrd]=sortrows([~elig,-dd]);
end
Kc=min(Kc,nA); Kd=min(Kd,max(0,nA-Kc));
if strcmp(p.refinePolicy,'knn') && Kc>0 && nA>Kc
    % ---- stage 2: re-rank the geometric superset by a k-NN fitness estimate ----
    npool=min(nA,max(Kc+1,round(p.refinePool*Kc)));
    pool=avail(cOrd(1:npool));
    [~,eo]=sort(fv);
    eo=eo(1:min(p.knnE,numel(eo)));
    eo=eo(isfinite(fv(eo)));
    if numel(eo)>=2
        Pv=Xv(eo,:); Fv=fv(eo);
        d2=zeros(numel(pool),numel(eo));
        for j=1:numel(eo)
            d2(:,j)=sum((Y(pool,:)-Pv(j,:)).^2,2);
        end
        w=1./(sqrt(d2)+1e-12);
        est=sum(w.*Fv',2)./sum(w,2);
        [~,eo2]=sort(est);
        Ecvg=pool(eo2(1:Kc));
    else
        Ecvg=avail(cOrd(1:Kc));
    end
else
    Ecvg=avail(cOrd(1:Kc));
end
E=[Ecvg(:); avail(dOrd(1:Kd))];
E=unique(E,'stable')';
if p.fillToQuota && numel(E)<min(nA,Kc+Kd)
    want=min(nA,Kc+Kd);
    ci=Kc+1; di=Kd+1; takeConv=true;
    while numel(E)<want && (ci<=nA || di<=nA)
        if takeConv
            while ci<=nA && any(E==avail(cOrd(ci))), ci=ci+1; end
            if ci<=nA, E(end+1)=avail(cOrd(ci)); ci=ci+1; end %#ok<AGROW>
        else
            while di<=nA && any(E==avail(dOrd(di))), di=di+1; end
            if di<=nA, E(end+1)=avail(dOrd(di)); di=di+1; end %#ok<AGROW>
        end
        takeConv=~takeConv;
    end
end
end

function [Xa,Xb,Xd]=raseLeaders(Xv,Y,fv,p,EA)
%RASELEADERS  Elite triangle, built from verified information only.
%   Default: the three best verified positions.
%   eliteAnchor=true: the best verified position EVER seen is pinned as
%   alpha (so a later, worse re-evaluation of the same wolf cannot delete the
%   elite), while beta and delta are re-derived from the current verified
%   records.  This keeps a diverse triangle instead of collapsing all three
%   leaders onto the same historical point.
if nargin<5, EA=struct('Cost',inf,'Vector',zeros(1,size(Xv,2))); end
[~,ord]=sort(fv);
m=min(4,numel(ord));
idx=ord(1:m);
if p.useVirtual
    pos=Xv(idx,:);
else
    pos=Y(idx,:);
end
cost=fv(idx);
if p.eliteAnchor && isfinite(EA.Cost)
    d=sqrt(sum((pos-EA.Vector).^2,2));
    keep=d>1e-12;                      % drop a duplicate of the anchor
    pos=[EA.Vector; pos(keep,:)];
    cost=[EA.Cost; cost(keep)];
    [~,o2]=sort(cost);
    pos=pos(o2,:);
end
while size(pos,1)<3, pos=[pos; pos(end,:)]; end
Xa=pos(1,:); Xb=pos(2,:); Xd=pos(3,:);
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

