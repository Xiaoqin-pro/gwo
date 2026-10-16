#!/usr/bin/env python
"""Budget-to-target analysis: how many evaluations does an algorithm need to
reach the quality that the reference reaches with the full budget?

    python experiments/budget_curve.py results/<batch> [--ref GWO]

Metric (proposed by the external reviews as the decision criterion for this
project):
    target_f = mean final error of the reference on function f at the full budget
    for every other algorithm and seed, find the first evaluation count at which
    its error trajectory is <= target_f
    report the median (over functions) of that count, as a fraction of the budget

Writes budget_curve.csv, budget_curve_by_function.csv and a figure.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--ref", default="GWO")
    ap.add_argument("--file", default="runs.csv")
    args = ap.parse_args()
    b = Path(args.batch)

    runs = pd.read_csv(b / args.file)
    runs = runs[runs.Status == "ok"]
    algos = list(dict.fromkeys(runs.Algorithm))

    tr = b / "traces_v7.mat"
    if not tr.exists():
        tr = b / "traces.mat"
    M = loadmat(tr)
    T = np.where(np.isfinite(M["Traces"]), M["Traces"], np.nan)
    fe = M["traceFE"].ravel()

    idx = {}
    for a in algos:
        idx[a] = runs.index[runs.Algorithm == a].to_numpy()

    # paired target: for seed s the target is what the reference itself achieves
    # on that same seed, so the comparison is not confounded by seed spread
    ref = runs[runs.Algorithm == args.ref].set_index(["FunctionID", "Seed"]).FinalError
    funcs = sorted(runs.FunctionID.unique())

    rows = []
    for f in funcs:
        for a in algos:
            if a == args.ref:
                continue
            cols = [c for c in idx[a] if runs.loc[c, "FunctionID"] == f]
            if not cols:
                continue
            need, hit = [], 0
            for c in cols:
                s_ = runs.loc[c, "Seed"]
                if (f, s_) not in ref.index:
                    continue
                tgt = ref.loc[(f, s_)]
                y = T[:, c]
                ok = np.where(y <= tgt)[0]
                if len(ok):
                    need.append(fe[ok[0]])
                    hit += 1
                else:
                    need.append(np.nan)
            need = np.array(need, float)
            rows.append(dict(FunctionID=f, Algorithm=a,
                             median_FE=float(np.nanmedian(need)) if hit else np.nan,
                             frac_of_budget=float(np.nanmedian(need) / fe[-1]) if hit else np.nan,
                             seeds_matched=hit, n=len(need)))
    df = pd.DataFrame(rows)
    df.to_csv(b / "budget_curve_by_function.csv", index=False)

    g = df.groupby("Algorithm")
    summ = pd.DataFrame({
        "median_frac": g.frac_of_budget.median(),
        "mean_frac": g.frac_of_budget.mean(),
        "functions_matched": g.seeds_matched.apply(lambda v: int((v > 0).sum())),
        "functions_compared": g.size(),
        "seeds_matched_pct": 100 * g.apply(
            lambda d: d.seeds_matched.sum() / d.n.sum(), include_groups=False),
    })
    summ["pct_functions_matched"] = 100 * summ.functions_matched / summ.functions_compared
    summ = summ.sort_values("median_frac")
    summ.to_csv(b / "budget_curve.csv")
    print(summ.round(3).to_string())
    print(f"\nreference = {args.ref}; budget = {fe[-1]:.0f} evaluations")
    print("median_frac = median over functions of the evaluation fraction needed "
          f"to reach {args.ref}-at-full-budget quality")

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.barh(range(len(summ)), summ.median_frac.values)
    ax.set_yticks(range(len(summ)), summ.index)
    ax.invert_yaxis()
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("median budget fraction needed to match the reference (per seed)")
    fig.savefig(b / "figures" / "budget_curve.pdf")
    fig.savefig(b / "figures" / "budget_curve.png")
    print((b / "figures" / "budget_curve.png"))


if __name__ == "__main__":
    main()
