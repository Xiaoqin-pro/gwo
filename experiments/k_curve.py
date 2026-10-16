#!/usr/bin/env python
"""Evaluation-density (K) curve.

    python experiments/k_curve.py results/ksweep_xxx results/igwo29_xxx \
        --out results/kcurve --ref IGWO

The sweep restricts the true-evaluation budget per generation (K candidates out
of the 2N that the I-GWO generator proposes) while keeping the total budget and
the generator unchanged.  This script assembles the curve: for every K it
reports the average rank, the function-level record against the full-evaluation
reference, the median error ratio, and the same quantities for the random
screening control where one exists.

Naming convention of the algorithm entries: a K is read from the suffix
-K<number>; the plain RASE-IGWO entry is K=50 and the reference (full evaluation)
is K=100 = 2N.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def k_of(name):
    m = re.search(r"-K(\d+)", name)
    if m:
        return int(m.group(1))
    if name == "RASE-IGWO":
        return 50
    if name in ("IGWO", "RASE-IGWO-full"):
        return 100
    return None


def is_random(name):
    return name.endswith("r") and "-K" in name


def holm(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (n - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="IGWO")
    args = ap.parse_args()

    parts = []
    for b in args.batches:
        f = Path(b) / "runs.csv"
        if f.exists():
            d = pd.read_csv(f)
            parts.append(d[d.Status == "ok"])
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["FunctionID", "Seed", "Algorithm"])

    funcs = sorted(df.FunctionID.unique())
    mean = df.groupby(["FunctionID", "Algorithm"]).FinalError.mean().unstack()
    ref = mean[args.ref]

    rows = []
    for a in mean.columns:
        if a.startswith("RASE-IGWO") and not is_random(a):
            pass
        K = k_of(a)
        if K is None and a != args.ref:
            continue
        ps = []
        for f in funcs:
            x = df[(df.FunctionID == f) & (df.Algorithm == a)].FinalError.values
            y = df[(df.FunctionID == f) & (df.Algorithm == args.ref)].FinalError.values
            if a == args.ref or len(x) < 5:
                ps.append(1.0)
            else:
                ps.append(stats.mannwhitneyu(x, y, alternative="two-sided",
                                             method="asymptotic",
                                             use_continuity=True)[1])
        adj = holm(ps)
        w = int((mean[a] < ref).sum())
        l = int((mean[a] > ref).sum())
        rows.append(dict(
            Algorithm=a, K=K, Control="random" if is_random(a) else "geometry",
            AvgRank=float(mean.rank(axis=1)[a].mean()),
            Wins=w, Losses=l,
            SigBetter=int(((adj < 0.05) & (mean[a].values < ref.values)).sum()),
            SigWorse=int(((adj < 0.05) & (mean[a].values > ref.values)).sum()),
            MedianRatio=float(np.exp(np.median(np.log(
                (mean[a].values + 1e-30) / (ref.values + 1e-30))))),
            MeanCPU=float(df[df.Algorithm == a].Seconds.mean())))
    out = pd.DataFrame(rows).sort_values(["Control", "K"])
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst / "k_curve.csv", index=False)
    print(out.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for ctrl, marker, col in (("geometry", "o", "#1f4e79"), ("random", "s", "#c0392b")):
        sub = out[out.Control == ctrl]
        if len(sub):
            ax.plot(sub.K, sub.AvgRank, marker + "-", color=col, label=ctrl + " screening")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(out.K.unique()))
    ax.set_xticklabels(sorted(out.K.unique()))
    ax.invert_xaxis()
    ax.set_xlabel("evaluations per generation K  (100 = full evaluation)")
    ax.set_ylabel("average rank (lower is better)")
    ax.legend(fontsize=8)
    fig.savefig(dst / "k_curve.pdf")
    fig.savefig(dst / "k_curve.png")
    print("wrote", dst / "k_curve.csv")


if __name__ == "__main__":
    main()
