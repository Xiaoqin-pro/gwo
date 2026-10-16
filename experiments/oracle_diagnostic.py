#!/usr/bin/env python
"""When does geometric screening pay off?  Class-level diagnostic.

    python experiments/oracle_diagnostic.py results/k30r_merged/runs.csv --out results/k30r_merged

A true oracle ("which K candidates would have been best") would require the true
fitness of all 2N candidates, i.e. full evaluation, so it is not an experiment we
can run.  What we can do with the archived 30-seed runs is ask the question the
external review actually posed -- WHEN does prioritising the leader-consensus
candidates help and when does it hurt -- by grouping the CEC2017 functions into
their published classes and reporting the geometric/random error ratio, the
per-seed win rate and the effect size per class.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = [("unimodal", [1, 3]),
           ("simple multimodal", [4, 5, 6, 7, 8, 9, 10]),
           ("hybrid", [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
           ("composition", [21, 22, 23, 24, 25, 26, 27, 28, 29, 30])]
GEO, RND, FULL = "RASE-IGWO-K25", "RASE-IGWO-K25-r", "IGWO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.runs)
    df = df[df.Status == "ok"]
    m = df.groupby(["FunctionID", "Algorithm"]).FinalError.mean().unstack()
    piv = df.pivot_table(index=["FunctionID", "Seed"], columns="Algorithm",
                         values="FinalError")

    rows = []
    for cname, fs in CLASSES:
        fs = [f for f in fs if f in m.index]
        ratio = np.exp(np.median(np.log((m.loc[fs, GEO] + 1e-30) /
                                        (m.loc[fs, RND] + 1e-30))))
        sub = piv.loc[fs]
        wins = int((sub[GEO] < sub[RND]).sum())
        n = len(sub)
        # effect size: matched-pairs rank-biserial from the signed ranks
        d = sub[GEO].values - sub[RND].values
        d = d[np.isfinite(d)]
        if len(d) and not np.allclose(d, 0):
            z = stats.wilcoxon(d, alternative="two-sided")
            try:
                r = z.statistic / (len(d) * (len(d) + 1) / 2)
            except Exception:
                r = np.nan
            try:
                p = z.pvalue
            except Exception:
                p = z[1]
        else:
            r, p = np.nan, 1.0
        rows.append(dict(Class=cname, Functions=len(fs),
                         MedianRatio=float(ratio), WinRate=wins / n if n else np.nan,
                         Paired_p=float(p), RankBiserial=float(r)))
    out = pd.DataFrame(rows)
    out.to_csv(dst / "oracle_class_diagnostic.csv", index=False)
    print(out.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(out.Class, out.MedianRatio,
           color=["#1f4e79" if v < 1 else "#c0392b" for v in out.MedianRatio])
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("median error ratio  geometric / random  (<1 favours geometry)")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    fig.savefig(dst / "figures" / "oracle_class_diagnostic.pdf")
    fig.savefig(dst / "figures" / "oracle_class_diagnostic.png")
    print("wrote", dst / "oracle_class_diagnostic.csv")


if __name__ == "__main__":
    main()
