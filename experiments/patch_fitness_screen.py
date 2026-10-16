# -*- coding: utf-8 -*-
"""Add a fitness-aware refinement stage to the screening of RASE_GWO.m.

Why: the current screening is entirely fitness-blind -- it ranks candidates by
their geometric distance to the leader consensus (and to the minority leader).
The low-budget sweep showed that every purely geometric setting lands in the same
narrow 0.82-0.90 error-ratio band, i.e. the criterion cannot separate "close to
the leaders" from "actually promising".

Mechanism (two-stage screening, still no model training, no matrix inversion):
  stage 1  geometry keeps a SUPERSET: the best `refinePool*K` candidates by the
           existing score;
  stage 2  those candidates are re-ranked by a k-nearest-neighbour fitness
           estimate taken from the verified elite archive (the `knnE` best
           verified positions, whose true fitness is already known), using an
           inverse-distance weighting:
               f_hat(y) = sum_j w_j f_j / sum_j w_j,  w_j = 1/(||y-x_j||+eps)
           The K candidates with the best estimate are evaluated.

Cost per round: refinePool*K*knnE*D multiply-adds, e.g. 4*50*10*50 = 1e5 -- one
third of the cost of moving the virtual pool, and ~1/500 of one expensive true
evaluation if the objective is a simulation.

Defaults reproduce the current behaviour (refinePolicy = 'none').

DO NOT RUN while an experiment using src/RASE_GWO.m is in flight.
"""
import io, sys
P = r"D:\111\Desktop\噜噜\狼群\src\RASE_GWO.m"
s = io.open(P, encoding="utf-8").read()
if "p.refinePolicy" in s:
    print("already patched"); sys.exit(0)

s = s.replace("p.mixFrac=0;                % share of the virtual pool on the DE family",
              "p.refinePolicy='none';      % 'none' | 'knn'  (two-stage fitness-aware screen)\n"
              "p.refinePool=4;             % stage-1 superset size = refinePool*K\n"
              "p.knnE=10;                  % verified elite positions used by the k-NN estimate\n"
              "p.mixFrac=0;                % share of the virtual pool on the DE family")

old = ("    case 'minority'\n"
       "        cand=[Xa;Xb;Xd];\n"
       "        [~,mi]=max(sqrt(sum((cand-XL).^2,2)));\n"
       "        dd=sqrt(sum((Ya-cand(mi,:)).^2,2));              % close to the minority leader\n")
assert old in s

old_pick = ("Kc=min(Kc,nA); Kd=min(Kd,max(0,nA-Kc));\n"
            "E=[avail(cOrd(1:Kc)); avail(dOrd(1:Kd))];")
new_pick = ("Kc=min(Kc,nA); Kd=min(Kd,max(0,nA-Kc));\n"
            "if strcmp(p.refinePolicy,'knn') && Kc>0 && nA>Kc\n"
            "    % ---- stage 2: re-rank the geometric superset by a k-NN fitness estimate ----\n"
            "    npool=min(nA,max(Kc+1,round(p.refinePool*Kc)));\n"
            "    pool=avail(cOrd(1:npool));\n"
            "    [~,eo]=sort(fv);\n"
            "    eo=eo(1:min(p.knnE,numel(eo)));\n"
            "    eo=eo(isfinite(fv(eo)));\n"
            "    if numel(eo)>=2\n"
            "        Pv=Xv(eo,:); Fv=fv(eo);\n"
            "        d2=zeros(numel(pool),numel(eo));\n"
            "        for j=1:numel(eo)\n"
            "            d2(:,j)=sum((Y(pool,:)-Pv(j,:)).^2,2);\n"
            "        end\n"
            "        w=1./(sqrt(d2)+1e-12);\n"
            "        est=sum(w.*Fv',2)./sum(w,2);\n"
            "        [~,eo2]=sort(est);\n"
            "        Ecvg=pool(eo2(1:Kc));\n"
            "    else\n"
            "        Ecvg=avail(cOrd(1:Kc));\n"
            "    end\n"
            "else\n"
            "    Ecvg=avail(cOrd(1:Kc));\n"
            "end\n"
            "E=[Ecvg(:); avail(dOrd(1:Kd))];")
assert old_pick in s, "selection block not found"
s = s.replace(old_pick, new_pick)

s = s.replace("info.FamilyImprove=famImprove;", "info.RefinePolicy=p.refinePolicy;\ninfo.FamilyImprove=famImprove;")
io.open(P, "w", encoding="utf-8").write(s)
print("two-stage fitness-aware screening added;", len(s.splitlines()), "lines")
