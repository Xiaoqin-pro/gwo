function [medNN,redFrac,minNN]=nnProbe(P,tauNorm)
%NNPROBE  Spatial redundancy statistics of the positions evaluated in one round.
%
%   [medNN,redFrac,minNN]=nnProbe(P,tauNorm)
%
%   P        n-by-D matrix of positions that consumed true evaluations in the
%            round (n = population for GWO, n = evaluation quota for RASE-GWO)
%   tauNorm  absolute distance below which two evaluations are considered to be
%            the same probe (use e.g. 1e-3*norm(ub-lb))
%
%   medNN    median nearest-neighbour distance inside the round's evaluation set
%   redFrac  fraction of evaluations whose nearest neighbour is closer than tauNorm
%   minNN    smallest nearest-neighbour distance
%
% The statistic answers the question that motivates the whole framework: how much
% spatial diversity does one round of true evaluations actually carry?  Canonical
% GWO evaluates its entire population, which contracts toward the leader triangle,
% so its evaluations become mutually redundant as the run proceeds.
%
% O(n^2 D), n is small (tens), so the probe is negligible next to one objective
% evaluation.  It draws no random numbers and therefore never perturbs a run.

n=size(P,1);
if n<2
    medNN=NaN; redFrac=NaN; minNN=NaN; return;
end
sq=sum(P.^2,2);
Dm=sq+sq'-2*(P*P');
Dm(Dm<0)=0;
Dm=sqrt(Dm);
Dm(1:n+1:end)=inf;
nn=min(Dm,[],2);
medNN=median(nn);
minNN=min(nn);
if nargin>1 && ~isempty(tauNorm)
    redFrac=mean(nn<tauNorm);
else
    redFrac=NaN;
end
end
