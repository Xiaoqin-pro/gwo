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
    aC=2-iter*(2/maxIter);

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

    % ---- evaluate both candidates; a wolf keeps only strict improvements ----
    for i=1:N
        if fe>=budget, break; end
        vg=f(Xg(i,:));
        fe=fe+1; bestEver=min(bestEver,vg); T(fe)=bestEver;
        if vg<pBestScore(i), pBest(i,:)=Xg(i,:); pBestScore(i)=vg; end
        if fe>=budget, break; end
        vd=f(Xd(i,:));
        fe=fe+1; bestEver=min(bestEver,vd); T(fe)=bestEver;
        if vd<pBestScore(i), pBest(i,:)=Xd(i,:); pBestScore(i)=vd; end
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
end

% ===================== helpers =====================
function V=igwoBounds(V,old,lb,ub)
%IGWOBOUNDS  Authors' midpoint bound repair (mechanism 4).
LB=repmat(lb,size(V,1),1);
UB=repmat(ub,size(V,1),1);
p=V<LB; V(p)=(old(p)+LB(p))/2;
p=V>UB; V(p)=(old(p)+UB(p))/2;
end
