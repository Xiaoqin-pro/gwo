# -*- coding: utf-8 -*-
"""Add a candidate-generation portfolio to RASE_GWO.m.

Motivation (measured, see results/perf1_20261016_062150/REPORT.md): the three
mechanisms tested in the first performance round peak on DIFFERENT functions --
the elite anchor wins F5/F8/F10/F12/F13, micro-batched re-screening wins F6/F9,
I-GWO's dimension-wise DLH wins F3/F11.  A single fixed screening rule therefore
cannot be optimal, which is exactly what the low-budget sweep showed: all quota
settings sit in a narrow 0.82-0.90 band.

Mechanism: keep a portfolio of candidate GENERATORS inside the virtual pool and
let the evaluation quota flow to the generator that is currently productive.

  family 1  standard GWO movement (three-leader average)
  family 2  DE/current-to-best/1 movement, dimension-wise
                Y_i <- Y_i + r1 .* (X_alpha - Y_i) + r2 .* (Y_a - Y_b)

The per-round quota is split between the families, and the split is scheduled by
a second Q-table (state = budget phase x stagnation, action = family-1 share)
on the same reward as the convergence/diversity scheduler.  The agent therefore
learns "which candidate generator deserves the expensive evaluations", not just
"which side of one generator".

Defaults reproduce the current behaviour exactly (mixFrac = 0).

DO NOT RUN while an experiment using src/RASE_GWO.m is in flight.
"""
import io, sys
P = r"D:\111\Desktop\噜噜\狼群\src\RASE_GWO.m"
s = io.open(P, encoding="utf-8").read()
if "p.mixFrac" in s:
    print("already patched"); sys.exit(0)

# ---- 1. defaults ---------------------------------------------------------
s = s.replace("p.movePolicy='classic';     % 'classic' | 'noC' (Tsai & Shi 2024 correction)",
              "p.movePolicy='classic';     % 'classic' | 'noC' (Tsai & Shi 2024 correction)\n"
              "p.mixFrac=0;                % share of the virtual pool on the DE family\n"
              "p.familyShare=1.0;          % fixed share of the evaluation quota to family 1\n"
              "p.familyPolicy='fixed';     % 'fixed' | 'q'\n"
              "p.familyShares=[0.9 0.7 0.5 0.3];   % Q action set for the family split")

# ---- 2. family bookkeeping ----------------------------------------------
s = s.replace(
"evalCount=zeros(N,1);   % diagnostic: how many times each wolf was evaluated",
"fam=ones(N,1);          % 1 = GWO movement, 2 = DE/current-to-best movement\n"
"nF2=round(p.mixFrac*N);\n"
"if nF2>0, fam(1:nF2)=2; end\n"
"famImprove=zeros(2,1);  % evaluations that improved the best verified fitness\n"
"evalCount=zeros(N,1);   % diagnostic: how many times each wolf was evaluated")

# ---- 3. movement ---------------------------------------------------------
old_move = ("    if strcmp(p.movePolicy,'noC')\n"
            "        Y=( (Xa-A1.*abs(Xa-Y)) + (Xb-A2.*abs(Xb-Y)) + (Xd-A3.*abs(Xd-Y)) )/3;\n"
            "    else\n"
            "        Y=( (Xa-A1.*abs(C1.*Xa-Y)) + (Xb-A2.*abs(C2.*Xb-Y)) + (Xd-A3.*abs(C3.*Xd-Y)) )/3;\n"
            "    end\n"
            "    Y=min(max(Y,lb),ub);")
new_move = ("    if strcmp(p.movePolicy,'noC')\n"
            "        Yg=( (Xa-A1.*abs(Xa-Y)) + (Xb-A2.*abs(Xb-Y)) + (Xd-A3.*abs(Xd-Y)) )/3;\n"
            "    else\n"
            "        Yg=( (Xa-A1.*abs(C1.*Xa-Y)) + (Xb-A2.*abs(C2.*Xb-Y)) + (Xd-A3.*abs(C3.*Xd-Y)) )/3;\n"
            "    end\n"
            "    if nF2>0\n"
            "        % family 2: DE/current-to-best/1, dimension-wise\n"
            "        idx2=find(fam==2);\n"
            "        na=randi(N,numel(idx2),1);\n"
            "        na(na==idx2)=mod(na(na==idx2),N)+1;\n"
            "        nb=randi(N,numel(idx2),1);\n"
            "        nb(nb==idx2 | nb==na)=mod(nb(nb==idx2 | nb==na)+1,N)+1;\n"
            "        Y(idx2,:)=Y(idx2,:)+rand(numel(idx2),D).*(Xa-Y(idx2,:)) ...\n"
            "                              +rand(numel(idx2),D).*(Y(na,:)-Y(nb,:));\n"
            "    end\n"
            "    Y(fam==1,:)=Yg(fam==1,:);\n"
            "    Y=min(max(Y,lb),ub);")
assert old_move in s, "movement block not found"
s = s.replace(old_move, new_move)

# ---- 4. quota: allocate between families --------------------------------
old_q = ("    Kc=round(share*K); Kc=max(0,min(K,Kc)); Kd=K-Kc;\n"
         "\n"
         "    % ---- (c,d,e) screening, true evaluation, leadership refresh ----")
new_q = ("    Kc=round(share*K); Kc=max(0,min(K,Kc)); Kd=K-Kc;\n"
         "    if strcmp(p.familyPolicy,'q'), famShare=p.familyShares(famIdx);\n"
         "    else, famShare=p.familyShare; end\n"
         "    famShare=max(0,min(1,famShare));\n"
         "\n"
         "    % ---- (c,d,e) screening, true evaluation, leadership refresh ----")
assert old_q in s, "quota block not found"
s = s.replace(old_q, new_q)

old_scr = ("        E=raseScreen(Y,Xa,Xb,Xd,kcb,kdb,done,p,screenStream, ...\n"
           "                     isDense,strcmp(p.screenPolicy,'random'));")
new_scr = ("        if nF2>0\n"
           "            % split this batch between the two candidate generators\n"
           "            Kb1=round(famShare*Kb); Kb1=max(0,min(Kb,Kb1)); Kb2=Kb-Kb1;\n"
           "            E=[];\n"
           "            if Kb1>0\n"
           "                k1c=round(share*Kb1);\n"
           "                E=[E, raseScreen(Y,Xa,Xb,Xd,k1c,Kb1-k1c,done|(fam~=1),p, ...\n"
           "                     screenStream,isDense,strcmp(p.screenPolicy,'random'))];\n"
           "            end\n"
           "            if Kb2>0\n"
           "                k2c=round(share*Kb2);\n"
           "                E=[E, raseScreen(Y,Xa,Xb,Xd,k2c,Kb2-k2c,done|(fam~=2),p, ...\n"
           "                     screenStream,isDense,strcmp(p.screenPolicy,'random'))];\n"
           "            end\n"
           "        else\n"
           "            E=raseScreen(Y,Xa,Xb,Xd,kcb,kdb,done,p,screenStream, ...\n"
           "                         isDense,strcmp(p.screenPolicy,'random'));\n"
           "        end")
assert old_scr in s, "screening call not found"
s = s.replace(old_scr, new_scr)

# ---- 5. per-family improvement accounting -------------------------------
s = s.replace("            if v<EA.Cost, EA=struct('Cost',v,'Vector',Y(i,:)); end",
              "            if v<EA.Cost\n"
              "                EA=struct('Cost',v,'Vector',Y(i,:));\n"
              "                famImprove(fam(i))=famImprove(fam(i))+1;\n"
              "            end")

# ---- 6. second Q-table for the family split -----------------------------
old_qinit = ("    sIdx=raseStateIndex(fe,budget,fe-lastImproveFE,p);\n"
             "    aIdx=raseEpsAction(Q,sIdx,p.qEpsStart,quotaStream);")
new_qinit = ("    sIdx=raseStateIndex(fe,budget,fe-lastImproveFE,p);\n"
             "    aIdx=raseEpsAction(Q,sIdx,p.qEpsStart,quotaStream);\n"
             "    if strcmp(p.familyPolicy,'q')\n"
             "        Q2=p.qInit*ones(stateCount,numel(p.familyShares));\n"
             "        Q2Visits=zeros(stateCount,numel(p.familyShares));\n"
             "        famIdx=raseEpsAction(Q2,sIdx,p.qEpsStart,quotaStream);\n"
             "    else\n"
             "        Q2=[]; Q2Visits=[]; famIdx=1;\n"
             "    end\n"
             "    famShare=p.familyShare;")
assert old_qinit in s, "Q init block not found"
s = s.replace(old_qinit, new_qinit)

old_qup = ("        aIdx=raseEpsAction(Q,sNext,epsNow,quotaStream);\n"
           "        sIdx=sNext; lastQFE=fe; fWin=gbest;")
new_qup = ("        aIdx=raseEpsAction(Q,sNext,epsNow,quotaStream);\n"
           "        if strcmp(p.familyPolicy,'q')\n"
           "            Q2(sIdx_prev_fam,famIdx)=Q2(sIdx_prev_fam,famIdx)+p.qAlpha* ...\n"
           "                (rew+p.qGamma*max(Q2(sNext,:))-Q2(sIdx_prev_fam,famIdx));\n"
           "            Q2Visits(sIdx_prev_fam,famIdx)=Q2Visits(sIdx_prev_fam,famIdx)+1;\n"
           "            famIdx=raseEpsAction(Q2,sNext,epsNow,quotaStream);\n"
           "        end\n"
           "        sIdx=sNext; lastQFE=fe; fWin=gbest;")
assert old_qup in s, "Q update block not found"
s = s.replace(old_qup, new_qup)

# remember the state the family action was taken in
s = s.replace("        sNext=raseStateIndex(fe,budget,fe-lastImproveFE,p);",
              "        sNext=raseStateIndex(fe,budget,fe-lastImproveFE,p);\n"
              "        sIdx_prev_fam=sIdx;")

s = s.replace("info.FamilyImprove=famImprove;", "")  # not present yet
s = s.replace("info.EvalCount=evalCount;",
              "info.FamilyImprove=famImprove;\ninfo.MixFrac=p.mixFrac;\n"
              "info.FamilyShare=famShare;\ninfo.FamilyQ=Q2;\ninfo.FamilyQVisits=Q2Visits;\n"
              "info.EvalCount=evalCount;")

io.open(P, "w", encoding="utf-8").write(s)
print("generation-family portfolio added;", len(s.splitlines()), "lines")
