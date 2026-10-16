# -*- coding: utf-8 -*-
"""Add evaluation screening to IGWO.m: RASE-IGWO = I-GWO generator + our allocator.

Motivation.  The 30-seed balanced table (results/merged_30seed) shows that the
faithful I-GWO ranks 1.66 while every RASE variant sits at 3.6 or worse, and the
pre-registered gate says the round-2 allocation work does not clear G1.  The
reviewers' diagnosis -- the bottleneck is search capability -- is therefore
confirmed, and the natural move is to keep I-GWO's generators and put OUR
contribution (which candidates deserve an expensive evaluation) on top.

What changes: I-GWO evaluates both candidates of every wolf, i.e. 2N true
evaluations per generation.  With evalPolicy='screen' the same 2N candidates are
generated, the geometric dual-potential screening picks K of them, and only
those K consume true evaluations.  At an unchanged budget that buys 2N/K times
more generations.  Movement, personal-best retention, leader update, the
per-dimension DLH neighbourhood and the midpoint bound repair are untouched.

    state.evalPolicy  'full' (default, exact I-GWO) | 'screen'
    state.K           per-round quota under 'screen' (default N)
    state.screenPolicy 'geometry' | 'random'
    state.divCriterion 'minority' | 'leaders' | 'centroid'
    state.convShare   convergence share of the quota (default 0.9)

Under 'screen' the convergence factor follows the evaluation progress,
a = 2 - 2*(FE-N)/(budget-N), so the exploration/exploitation schedule stays
comparable to the full-evaluation run instead of being stretched.

DO NOT RUN while an experiment using src/IGWO.m is in flight.
"""
import io, sys
P = r"D:\111\Desktop\噜噜\狼群\src\IGWO.m"
s = io.open(P, encoding="utf-8").read()
if "evalPolicy" in s:
    print("already patched"); sys.exit(0)

# ---- defaults + state override -----------------------------------------
s = s.replace("rng(seed,'twister');\nclockStart=tic;",
"""% ---- screening options (defaults reproduce the published I-GWO) ----
p=struct('evalPolicy','full','K',[],'screenPolicy','geometry', ...
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
clockStart=tic;""")

# ---- convergence schedule ----------------------------------------------
s = s.replace("    aC=2-iter*(2/maxIter);",
"""    if isScreen
        aC=2-2*(fe-N)/max(1,budget-N);      % schedule tied to evaluation progress
    else
        aC=2-iter*(2/maxIter);              % published I-GWO schedule
    end""")

# ---- replace the evaluation block --------------------------------------
old = """    % ---- evaluate both candidates; a wolf keeps only strict improvements ----
    for i=1:N
        if fe>=budget, break; end
        vg=f(Xg(i,:));
        fe=fe+1; bestEver=min(bestEver,vg); T(fe)=bestEver;
        if vg<pBestScore(i), pBest(i,:)=Xg(i,:); pBestScore(i)=vg; end
        if fe>=budget, break; end
        vd=f(Xd(i,:));
        fe=fe+1; bestEver=min(bestEver,vd); T(fe)=bestEver;
        if vd<pBestScore(i), pBest(i,:)=Xd(i,:); pBestScore(i)=vd; end
    end"""
new = """    % ---- select which candidates consume true evaluations ----
    C=[Xg; Xd];                       % rows 1..N = X_GWO, N+1..2N = X_DLH
    if isScreen
        sel=igwoScreen(C,X,Alpha_pos,Beta_pos,Delta_pos,Kq,p,screenStream);
    else
        sel=1:2*N;                    % published I-GWO: evaluate both candidates
    end
    % ---- evaluate the selected candidates; a wolf keeps strict improvements --
    for sIdx=sel
        if fe>=budget, break; end
        i=1+mod(sIdx-1,N);
        v=f(C(sIdx,:));
        fe=fe+1; bestEver=min(bestEver,v); T(fe)=bestEver;
        if v<pBestScore(i), pBest(i,:)=C(sIdx,:); pBestScore(i)=v; end
    end"""
assert old in s, "evaluation block not found"
s = s.replace(old, new)

s = s.replace("info.Algorithm='IGWO';",
              "info.Algorithm='IGWO';\ninfo.EvalPolicy=p.evalPolicy;\ninfo.K=Kq;\ninfo.ScreenPolicy=p.screenPolicy;")

# ---- screening helper ---------------------------------------------------
helper = '''function sel=igwoScreen(C,X,Xa,Xb,Xd,Kq,p,stream)
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

'''
s = s.replace("function V=igwoBounds(V,old,lb,ub)", helper + "function V=igwoBounds(V,old,lb,ub)", 1)
io.open(P, "w", encoding="utf-8").write(s)
print("IGWO.m extended with evaluation screening;", len(s.splitlines()), "lines")
