function [Best,T,info]=IGWO(f,state,~,N,seed)
%IGWO  Improved Grey Wolf Optimizer (I-GWO).
%
%   [Best,T,info]=IGWO(f,state,[],N,seed)
%
% Reconstruction of the I-GWO of Nadimi-Shahraki, Taghian and Mirjalili,
% "An improved grey wolf optimizer for solving engineering problems",
% Expert Systems with Applications 166:113917, 2021.
%
% I-GWO adds Dimension Learning-Based Hunting (DLH) to the canonical GWO:
% for every wolf two candidates are generated per generation.
%   (1) the standard GWO candidate X_GWO,i(t+1) built from alpha/beta/delta;
%   (2) the DLH candidate, built dimension by dimension from the wolves inside
%       the neighbourhood radius R_i(t) = ||X_i(t) - X_GWO,i(t+1)||:
%           N_i(t) = { X_j(t) : ||X_i(t)-X_j(t)|| <= R_i(t), j ~= i }
%           X_DLH,i,d(t+1) = X_i,d(t) + rand*( X_n,d(t) - X_r,d(t) )
%       with X_n a random neighbour and X_r a random wolf of the population.
% The better of the two candidates becomes X_i(t+1).
%
% COST NOTE (matters for a fair reading of the results): selecting the better of
% two candidates requires evaluating both, so one I-GWO generation consumes
% 2N true evaluations -- i.e. under a fixed budget it performs about half as
% many generations as canonical GWO.  This is the published algorithm; the
% evaluation count is tracked exactly and never exceeds the budget.
%
% If the neighbourhood of a wolf holds fewer than two members (which happens
% when R_i is small), the two nearest other wolves are used instead, so the DLH
% candidate is always well defined.
%
% Required state fields: initialPopulation, lowerBound, upperBound,
% maxEvaluations.  Randomness: rng(seed,'twister').

pop=state.initialPopulation;
[N0,D]=size(pop);
assert(N0==N,'Population size mismatch: state holds %d rows, N=%d.',N0,N);
lb=state.lowerBound; ub=state.upperBound; budget=state.maxEvaluations;
assert(budget>=N && budget==floor(budget),'Budget must be an integer >= N.');
rng(seed,'twister');
clockStart=tic;

X=pop;
fit=zeros(N,1);
T=nan(budget,1);
Alpha=struct('Cost',inf,'Vector',zeros(1,D));
Beta=Alpha; Delta=Alpha;
fe=0;
for i=1:N
    v=f(X(i,:));
    fe=fe+1; fit(i)=v;
    [Alpha,Beta,Delta]=insertLeader(Alpha,Beta,Delta,X(i,:),v);
    T(fe)=Alpha.Cost;
end
rounds=0;

% ===================== main loop =====================
while fe<budget
    rounds=rounds+1;
    aC=2-2*(fe/budget);
    Xg=zeros(N,D); Xd=zeros(N,D);
    for i=1:N
        % ---- (1) standard GWO candidate ----
        r1=rand(1,D); r2=rand(1,D);
        A1=2*aC*r1-aC; C1=2*r2;
        X1=Alpha.Vector-A1.*abs(C1.*Alpha.Vector-X(i,:));
        r1=rand(1,D); r2=rand(1,D);
        A2=2*aC*r1-aC; C2=2*r2;
        X2=Beta.Vector -A2.*abs(C2.*Beta.Vector -X(i,:));
        r1=rand(1,D); r2=rand(1,D);
        A3=2*aC*r1-aC; C3=2*r2;
        X3=Delta.Vector-A3.*abs(C3.*Delta.Vector-X(i,:));
        Xg(i,:)=min(max((X1+X2+X3)/3,lb),ub);
        % ---- (2) DLH candidate ----
        Ri=norm(X(i,:)-Xg(i,:));
        dist=sqrt(sum((X-X(i,:)).^2,2));
        dist(i)=inf;
        nb=find(dist<=Ri);
        if numel(nb)<2
            [~,ord]=sort(dist);
            nb=ord(1:2);
        end
        nSel=nb(randi(numel(nb)));
        rCand=randi(N);
        while rCand==i || rCand==nSel
            rCand=randi(N);
        end
        Xd(i,:)=min(max(X(i,:)+rand(1,D).*(X(nSel,:)-X(rCand,:)),lb),ub);
    end
    % ---- evaluate both candidates and keep the better ----
    for i=1:N
        if fe>=budget, break; end
        vg=f(Xg(i,:)); fe=fe+1;
        [Alpha,Beta,Delta]=insertLeader(Alpha,Beta,Delta,Xg(i,:),vg);
        T(fe)=Alpha.Cost;
        if fe>=budget, break; end
        vd=f(Xd(i,:)); fe=fe+1;
        [Alpha,Beta,Delta]=insertLeader(Alpha,Beta,Delta,Xd(i,:),vd);
        T(fe)=Alpha.Cost;
        if vg<=vd
            X(i,:)=Xg(i,:); fit(i)=vg;
        else
            X(i,:)=Xd(i,:); fit(i)=vd;
        end
    end
end

Best.Cost=Alpha.Cost; Best.Vector=Alpha.Vector;
info.Evaluations=fe;
info.Rounds=rounds;
info.MeanEvalsPerRound=fe/max(rounds,1);
info.WallClock=toc(clockStart);
info.Algorithm='IGWO';
end

% ===================== helpers =====================
function [A,B,Dl]=insertLeader(A,B,Dl,x,v)
if v<A.Cost
    Dl=B; B=A; A=struct('Cost',v,'Vector',x);
elseif v<B.Cost
    Dl=B; B=struct('Cost',v,'Vector',x);
elseif v<Dl.Cost
    Dl=struct('Cost',v,'Vector',x);
end
end
