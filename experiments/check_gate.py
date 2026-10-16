#!/usr/bin/env python
"""Evaluate the pre-registered round-2 decision gates.

    python experiments/check_gate.py results/<merged>/runs.csv

The gates are written down in docs/03_ROUND2_GATE.md BEFORE the perf30 batch was
run; this script only measures them, it does not choose them.

  G1  Elite90q3 vs Over1000-90q3 : >=1 Holm-significant function AND >=18/29
                                   function-mean wins
  G2  Mix50-70q vs Elite90q3     : better average rank AND >=18/29 wins
  G3  both vs GWO                : >=20/29 function-mean wins
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FROZEN, ELITE, MIX = "Over1000-90q3", "Elite90q3", "Mix50-70q"


def holm(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (n - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def compare(df, a, b, funcs):
    """function-mean wins/losses of a over b, per-seed paired median ratio and
    the Holm-corrected significant counts."""
    m = df.groupby(["FunctionID", "Algorithm"]).FinalError.mean().unstack()
    ma, mb = m[a].reindex(funcs), m[b].reindex(funcs)
    wins = int((ma < mb).sum())
    losses = int((ma > mb).sum())
    ps = []
    for f in funcs:
        x = df[(df.FunctionID == f) & (df.Algorithm == a)].FinalError.values
        y = df[(df.FunctionID == f) & (df.Algorithm == b)].FinalError.values
        if len(x) >= 5 and not np.allclose(x, y):
            ps.append(stats.mannwhitneyu(x, y, alternative="two-sided",
                                         method="asymptotic", use_continuity=True)[1])
        else:
            ps.append(1.0)
    adj = holm(ps)
    sig_b = int(((adj < 0.05) & (ma.values < mb.values)).sum())
    sig_w = int(((adj < 0.05) & (ma.values > mb.values)).sum())
    ratio = float(np.exp(np.median(np.log((ma.values + 1e-30) / (mb.values + 1e-30)))))
    rank = m.rank(axis=1).mean()
    return dict(wins=wins, losses=losses, sig_better=sig_b, sig_worse=sig_w,
                ratio=ratio, rank_a=float(rank[a]), rank_b=float(rank[b]))


def main():
    df = pd.read_csv(sys.argv[1])
    df = df[df.Status == "ok"]
    funcs = sorted(df.FunctionID.unique())
    algos = set(df.Algorithm)
    for need in (FROZEN, ELITE, MIX, "GWO"):
        if need not in algos:
            sys.exit(f"missing algorithm {need} in {sys.argv[1]}")

    print(f"{len(funcs)} functions, {df.Seed.nunique()} seeds, "
          f"budget {int(df.Evaluations.max())} FE\n")

    r1 = compare(df, ELITE, FROZEN, funcs)
    print(f"G1  {ELITE} vs {FROZEN} (frozen)")
    print(f"    function-mean W/L = {r1['wins']}/{r1['losses']}   "
          f"Holm significant better/worse = {r1['sig_better']}/{r1['sig_worse']}")
    print(f"    mean rank {r1['rank_a']:.2f} vs {r1['rank_b']:.2f}, "
          f"median error ratio {r1['ratio']:.3f}")
    g1 = (r1["sig_better"] >= 1) and (r1["wins"] >= 18)
    print(f"    G1 {'PASS' if g1 else 'FAIL'}\n")

    r2 = compare(df, MIX, ELITE, funcs)
    print(f"G2  {MIX} vs {ELITE}")
    print(f"    function-mean W/L = {r2['wins']}/{r2['losses']}   "
          f"Holm significant better/worse = {r2['sig_better']}/{r2['sig_worse']}")
    print(f"    mean rank {r2['rank_a']:.2f} vs {r2['rank_b']:.2f}, "
          f"median error ratio {r2['ratio']:.3f}")
    g2 = (r2["rank_a"] < r2["rank_b"]) and (r2["wins"] >= 18)
    print(f"    G2 {'PASS' if g2 else 'FAIL'}\n")

    for name in (ELITE, MIX):
        r3 = compare(df, name, "GWO", funcs)
        ok = r3["wins"] >= 20
        print(f"G3  {name} vs GWO: W/L {r3['wins']}/{r3['losses']}, "
              f"Holm sig better/worse {r3['sig_better']}/{r3['sig_worse']}, "
              f"ratio {r3['ratio']:.3f}  -> {'PASS' if ok else 'FAIL'}")

    print()
    if not g1:
        print("VERDICT: round-2 mechanisms do not clear G1. Keep the first-round "
              "story as the paper's backbone; do not build the per-family credit "
              "assignment on top of them.")
    elif not g2:
        print("VERDICT: elite anchor holds, family mixing does NOT. Drop the "
              "family mix and its Q-table (Occam) and continue with the elite "
              "anchor as the round-2 baseline.")
    else:
        print("VERDICT: both gates pass. Proceed to per-family credit assignment, "
              "with the three fixes and the random control required by "
              "docs/03_ROUND2_GATE.md.")


if __name__ == "__main__":
    main()
