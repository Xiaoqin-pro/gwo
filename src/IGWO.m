function [Best,T,info]=IGWO(f,state,~,N,seed)
%IGWO  Improved Grey Wolf Optimizer (I-GWO), faithful reimplementation.
%
%   [Best,T,info]=IGWO(f,state,[],N,seed)
%
% Nadimi-Shahraki, Taghian and Mirjalili, "An improved grey wolf optimizer for
% solving engineering problems", Expert Systems with Applications 166:113917,
% 2021.  Rebuilt against the authors' released MATLAB source (File Exchange
% 81253, mirrored at VG-TechCenter/Optimizers) so that the four mechanisms that
% are easy to get wrong are reproduced exactly:
%
%   (1) PERSONAL-BEST RETENTION.  A wolf keeps its own historical best position
%       and only moves there when a new candidate strictly improves it:
%           pBestScore = min(pBestScore, candidate);  Positions = pBest
%       The naive version (accept the better of the two candidates regardless of
%       the wolf's own record) lets a wolf drift to a worse position and is not
%       I-GWO.
%   (2) PER-DIMENSION DLH NEIGHBOURS.  For dimension d the exemplar neighbour is
%       drawn independently,
%           X_DLH(t,d) = X(t,d) + r_d * ( X(n_d,d) - X(r(t),d) )
%       with n_d uniform over the radius-neighbourhood of wolf t and r(t) a
%       random wolf (the authors use a random permutation, so each wolf gets a
%       distinct partner).  Using ONE neighbour for all dimensions changes the
%       sampling distribution of the DLH candidate.
%   (3) LEADER UPDATE FROM THE RETAINED POPULATION.  alpha/beta/delta are
%       refreshed at the start of each generation from Positions/Fit, i.e. from
%       positions a wolf actually keeps -- never from a candidate that was
%       evaluated but rejected.
%   (4) MIDPOINT BOUND REPAIR.  A violating component is pulled halfway back to
%       the violated bound,
%           v < LB  ->  (old + LB)/2 ,      v > UB  ->  (old + UB)/2
%       instead of being clamped onto the boundary.
%
% Cost and budget accounting: one I-GWO generation evaluates 2N candidates, so
% under a fixed budget it performs about half as many generations as canonical
% GWO.  The counter is exact and stops at the budget; if the budget ends inside a
% generation, the candidates that were still evaluated compete against their
% wolf's personal best, which is exactly equivalent to the authors' selection
% followed by the personal-best comparison.
%
% Required state fields: initialPopulation, lowerBound, upperBound,
% maxEvaluations.  Randomness: rng(seed,'twister').

pop=state.initialPopulation;
[N0,D]=size(pop);
assert(N0==N,'Population size mismatch: state holds %d rows, N=%d.',N0,N);
lb=state.lowerBound; ub=state.upperBound; budget=state.maxEvaluations;
assert(budget>=N && budget==floor(budget),'Budget must be an integer >= N.');
assert(numel(lb)==D && numel(ub)==D,'Bound vectors must match D=%d.',D);
% ---- screening options (defaults reproduce the published I-GWO) ----
p=struct('evalPolicy','full','K',[],'screenPolicy','geometry','parentAware','off', ...
         'divCriterion','minority','convShare',0.9,'screenRngOffset',4000001);
fn=fieldnames(p);
for k=1:numel(fn)
    if isfield(state,fn{k}), p.(fn{k})=state.(fn{k}); end
end
isScreen=strcmp(p.evalPolicy,'screen');
screenStream=RandStream('mt19937ar','Seed',rem(seed+p.screenRngOffset,4294967295));
if isScreen
    Kq=p.K;
    if isempty(Kq), Kq=N; end
    Kq=max(1,min(2*N,round(Kq)));
else
    Kq=2*N;
end

rng(seed,'twister');
clockStart=tic;

X=pop;
fit=zeros(N,1);
T=nan(budget,1);
fe=0; bestEver=inf;
for i=1:N
    v=f(X(i,:));
    fe=fe+1; fit(i)=v;
    bestEver=min(bestEver,v);
    T(fe)=bestEver;
end
pBest=X; pBestScore=fit;                 % personal records (mechanism 1)

Alpha=inf; Beta=inf; Delta=inf;
Alpha_pos=zeros(1,D); Beta_pos=zeros(1,D); Delta_pos=zeros(1,D);
maxIter=max(1,ceil((budget-N)/(2*N)));
iter=0;
rounds=0;

% ===================== main loop =====================
while fe<budget
    rounds=rounds+1;
    % ---- (3) leaders refreshed from the retained population ----
    for i=1:N
        fitness=fit(i);
        if fitness<Alpha
            Alpha=fitness; Alpha_pos=X(i,:);
        end
        if fitness>Alpha && fitness<Beta
            Beta=fitness; Beta_pos=X(i,:);
        end
        if fitness>Alpha && fitness>Beta && fitness<Delta
            Delta=fitness; Delta_pos=X(i,:);
        end
    end
    if isScreen
        aC=2-2*(fe-N)/max(1,budget-N);      % schedule tied to evaluation progress
    else
        aC=2-iter*(2/maxIter);              % published I-GWO schedule
    end

    % ---- candidate X_GWO (classic GWO move, per dimension) ----
    A1=2*aC*rand(N,D)-aC; C1=2*rand(N,D);
    A2=2*aC*rand(N,D)-aC; C2=2*rand(N,D);
    A3=2*aC*rand(N,D)-aC; C3=2*rand(N,D);
    Xg=( (Alpha_pos-A1.*abs(C1.*Alpha_pos-X)) + ...
         (Beta_pos -A2.*abs(C2.*Beta_pos -X)) + ...
         (Delta_pos-A3.*abs(C3.*Delta_pos-X)) )/3;
    Xg=igwoBounds(Xg,X,lb,ub);           % (4)

    % ---- candidate X_DLH (per-dimension neighbours) ----
    radius=sqrt(sum((X-Xg).^2,2));       % radius(t,t) of pdist2(X, X_GWO)
    Dm=zeros(N,N);
    for t=1:N
        Dm(t,:)=sqrt(sum((X-X(t,:)).^2,2))';
    end
    rp=randperm(N);                      % authors' random partner permutation
    Xd=zeros(N,D);
    for t=1:N
        idx=find(Dm(t,:)<=radius(t));    % radius neighbourhood, includes t
        nSel=idx(randi(numel(idx),1,D)); % (2) one neighbour PER DIMENSION
        lin=sub2ind([N D],nSel(:)',1:D);
        Xd(t,:)=X(t,:)+rand(1,D).*(X(lin)-X(rp(t),:));
    end
    Xd=igwoBounds(Xd,X,lb,ub);           % (4)

    % ---- select which candidates consume true evaluations ----
    C=[Xg; Xd];                       % rows 1..N = X_GWO, N+1..2N = X_DLH
    if isScreen
        % Only the screened subset is evaluated.  If the budget ends inside the
        % generation, the remaining candidates are simply not evaluated; the
        % wolves keep their personal records (which is what the verified-state
        % decoupling means).
        sel=igwoScreen(C,X,Alpha_pos,Beta_pos,Delta_pos,Kq,p,screenStream,N);
        for sIdx=sel
            if fe>=budget, break; end
            i=1+mod(sIdx-1,N);
            v=f(C(sIdx,:));
            fe=fe+1; bestEver=min(bestEver,v); T(fe)=bestEver;
            if v<pBestScore(i), pBest(i,:)=C(sIdx,:); pBestScore(i)=v; end
        end
    else
        % Published I-GWO order: both candidates of a wolf back to back.  The
        % order is kept deliberately -- with a budget that is not a multiple of
        % 2N the last generation is partial, and the interleaved order evaluates
        % a different (and for I-GWO canonical) subset than a block order would.
        for i=1:N
            if fe>=budget, break; end
            vg=f(C(i,:));
            fe=fe+1; bestEver=min(bestEver,vg); T(fe)=bestEver;
            if vg<pBestScore(i), pBest(i,:)=C(i,:); pBestScore(i)=vg; end
            if fe>=budget, break; end
            vd=f(C(N+i,:));
            fe=fe+1; bestEver=min(bestEver,vd); T(fe)=bestEver;
            if vd<pBestScore(i), pBest(i,:)=C(N+i,:); pBestScore(i)=vd; end
        end
    end
    X=pBest; fit=pBestScore;             % (1)
    iter=iter+1;
end

Best.Cost=bestEver;
[~,bi]=min(pBestScore);
Best.Vector=pBest(bi,:);
info.Evaluations=fe;
info.Rounds=rounds;
info.MeanEvalsPerRound=fe/max(rounds,1);
info.WallClock=toc(clockStart);
info.Algorithm='IGWO';
info.EvalPolicy=p.evalPolicy;
info.K=Kq;
info.ScreenPolicy=p.screenPolicy;
info.ParentAware=p.parentAware;
end

% ===================== helpers =====================
function sel=igwoScreen(C,X,Xa,Xb,Xd,Kq,p,stream,N)
%IGWOSCREEN  Choose Kq of the 2N generated candidates to evaluate.
%   Convergence slots go to the candidates nearest the leader-consensus centre,
%   exploration slots to the candidates nearest the minority leader (the member
%   of the elite triangle farthest from that centre).  Purely geometric: no
%   model is trained and no fitness is estimated.
n=size(C,1);
if strcmp(p.screenPolicy,'random')
    sel=randperm(stream,n,Kq); return;
end
XL=(Xa+Xb+Xd)/3;
dc=sqrt(sum((C-XL).^2,2));
switch p.divCriterion
    case 'minority'
        cand=[Xa;Xb;Xd];
        [~,mi]=max(sqrt(sum((cand-XL).^2,2)));
        dd=sqrt(sum((C-cand(mi,:)).^2,2));
        [~,dOrd]=sortrows(dd);
    case 'leaders'
        dd=min([sqrt(sum((C-Xa).^2,2)),sqrt(sum((C-Xb).^2,2)), ...
                sqrt(sum((C-Xd).^2,2))],[],2);
        [~,dOrd]=sortrows(-dd);
    otherwise
        Cent=mean(X,1);
        dd=sqrt(sum((C-Cent).^2,2));
        [~,dOrd]=sortrows(-dd);
end
[~,cOrd]=sortrows(dc);
Kc=round(p.convShare*Kq); Kc=max(0,min(Kq,Kc)); Kd=Kq-Kc;
sel=[cOrd(1:min(Kc,n)); dOrd(1:min(Kd,n))];
sel=unique(sel,'stable')';
if ~strcmp(p.parentAware,'off') && strcmp(p.screenPolicy,'geometry')
    % PAEA: coverage-first pass over the union of both families.
    % Candidate s belongs to parent 1+mod(s-1,N).
    par=@(s) 1+mod(s-1,N);
    covered=false(N,1);
    pick=[];
    for c=[cOrd; dOrd]'
        if numel(pick)>=Kq, break; end
        if strcmp(p.parentAware,'hard') && any(par(pick)==par(c)), continue; end
        if ~covered(par(c))
            pick(end+1)=c; covered(par(c))=true; %#ok<AGROW>
        end
    end
    for c=[cOrd; dOrd]'
        if numel(pick)>=Kq, break; end
        if strcmp(p.parentAware,'hard') && any(par(pick)==par(c)), continue; end
        if ~any(pick==c), pick(end+1)=c; end %#ok<AGROW>
    end
    sel=pick;
end
ci=Kc+1; di=Kd+1; takeConv=true;
while numel(sel)<min(Kq,n) && (ci<=n || di<=n)
    if takeConv
        while ci<=n && any(sel==cOrd(ci)), ci=ci+1; end
        if ci<=n, sel(end+1)=cOrd(ci); ci=ci+1; end %#ok<AGROW>
    else
        while di<=n && any(sel==dOrd(di)), di=di+1; end
        if di<=n, sel(end+1)=dOrd(di); di=di+1; end %#ok<AGROW>
    end
    takeConv=~takeConv;
end
end

function V=igwoBounds(V,old,lb,ub)
%IGWOBOUNDS  Authors' midpoint bound repair (mechanism 4).
LB=repmat(lb,size(V,1),1);
UB=repmat(ub,size(V,1),1);
p=V<LB; V(p)=(old(p)+LB(p))/2;
p=V>UB; V(p)=(old(p)+UB(p))/2;
end
