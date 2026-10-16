# -*- coding: utf-8 -*-
"""Add movePolicy = 'classic' | 'noC' to RASE_GWO.m.

'noC' applies the correction of Tsai & Shi, "Potential corrections to grey wolf
optimizer", Applied Soft Computing 161:111776, 2024: drop the C randomisation of
the encircling distance so that

    D_alpha = |X_alpha - Y|,      Y1 = X_alpha - A1 .* D_alpha

instead of D_alpha = |C1 .* X_alpha - Y|.  With the classic form a wolf that
already sits exactly on a leader is still displaced by |(C-1) X_alpha|, which is
large whenever the (shifted) leader is far from the coordinate origin -- exactly
the situation on the CEC2017 shifted/rotated functions.  The correction removes
that artefact.  This is a PUBLISHED correction and must be cited as such; it is
not a contribution of this work.

Default 'classic' reproduces the current behaviour bit for bit.

DO NOT RUN while an experiment using src/RASE_GWO.m is in flight.
"""
import io, sys
P = r"D:\111\Desktop\噜噜\狼群\src\RASE_GWO.m"
s = io.open(P, encoding="utf-8").read()
if "p.movePolicy" in s:
    print("already patched"); sys.exit(0)

s = s.replace("p.mixFrac=0;                % share of the virtual pool on the crossover family",
              "p.movePolicy='classic';     % 'classic' | 'noC' (Tsai & Shi 2024 correction)\n"
              "p.mixFrac=0;                % share of the virtual pool on the crossover family")

old = "    Y=( (Xa-A1.*abs(C1.*Xa-Y)) + (Xb-A2.*abs(C2.*Xb-Y)) + (Xd-A3.*abs(C3.*Xd-Y)) )/3;"
new = ("    if strcmp(p.movePolicy,'noC')\n"
       "        Y=( (Xa-A1.*abs(Xa-Y)) + (Xb-A2.*abs(Xb-Y)) + (Xd-A3.*abs(Xd-Y)) )/3;\n"
       "    else\n"
       "        Y=( (Xa-A1.*abs(C1.*Xa-Y)) + (Xb-A2.*abs(C2.*Xb-Y)) + (Xd-A3.*abs(C3.*Xd-Y)) )/3;\n"
       "    end")
assert old in s, "movement line not found"
s = s.replace(old, new)
s = s.replace("info.FamilyImprove=famImprove;", "info.MovePolicy=p.movePolicy;\ninfo.FamilyImprove=famImprove;")
io.open(P, "w", encoding="utf-8").write(s)
print("movePolicy added")
