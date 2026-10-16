#!/usr/bin/env python
"""Summarise a run_screen batch: ranks, pairwise tests, per-function table.

    python experiments/analyze_screen.py results/<batch_dir> [--ref GWO]

Writes REPORT.md, summary_by_algorithm.csv, per_function.csv and
pairwise_vs_ref.csv into the batch directory.  No result is dropped: functions
that favour the reference are kept and reported.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load(batch: Path, fname: str = "runs.csv") -> pd.DataFrame:
    runs = batch / fname
    if not runs.exists():
        sys.exit(f"missing {runs}")
    df = pd.read_csv(runs)
    bad = df[df["Status"] != "ok"]
    if len(bad):
        print(f"WARNING: {len(bad)} failed runs are excluded from the statistics")
        for _, r in bad.head(5).iterrows():
            print("   ", r["FunctionID"], r["Seed"], r["Algorithm"], r["Message"])
    return df[df["Status"] == "ok"].copy()


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--ref", default="GWO")
    ap.add_argument("--file", default="runs.csv")
    args = ap.parse_args()
    batch = Path(args.batch)
    df = load(batch, args.file)

    funcs = sorted(df["FunctionID"].unique())
    algos = list(dict.fromkeys(df["Algorithm"]))
    seeds = sorted(df["Seed"].unique())
    if args.ref not in algos:
        sys.exit(f"reference {args.ref} not in batch: {algos}")
    print(f"batch {batch.name}: {len(funcs)} functions x {len(seeds)} seeds "
          f"x {len(algos)} algorithms, {len(df)} valid runs")

    # ---------- mean error per (function, algorithm) ----------
    mean_err = (df.groupby(["FunctionID", "Algorithm"])["FinalError"]
                  .mean().unstack())
    mean_err = mean_err.reindex(index=funcs, columns=algos)
    mean_err.to_csv(batch / "per_function.csv")

    # ---------- average rank (Friedman blocks = function means) ----------
    ranks = mean_err.rank(axis=1, method="average")
    avg_rank = ranks.mean(axis=0)
    if len(algos) > 2:
        try:
            chi2, fp = stats.friedmanchisquare(
                *[mean_err[a].values for a in algos])
        except Exception:
            chi2, fp = np.nan, np.nan
    else:
        chi2, fp = np.nan, np.nan

    def col(name):
        if name in df.columns:
            return df.groupby("Algorithm")[name].mean().reindex(algos)
        return pd.Series(np.nan, index=algos)
    cpu = col("Seconds"); fev = col("Evaluations"); rnd = col("Rounds")

    summary = pd.DataFrame({
        "AverageRank": avg_rank,
        "MeanSeconds": cpu,
        "MeanEvaluations": fev,
        "MeanRounds": rnd,
    })
    summary.to_csv(batch / "summary_by_algorithm.csv")

    # ---------- pairwise vs reference ----------
    ref_pf = mean_err[args.ref]
    rows = []
    for a in algos:
        if a == args.ref:
            continue
        wins = losses = ties = 0
        pvals, rawtbl = [], {}
        seed_win = seed_loss = seed_tie = 0
        for f in funcs:
            xa = df[(df.FunctionID == f) & (df.Algorithm == a)].sort_values("Seed")
            xr = df[(df.FunctionID == f) & (df.Algorithm == args.ref)].sort_values("Seed")
            m = xa.merge(xr, on="Seed", suffixes=("_a", "_r"))
            ea, er = m["FinalError_a"].values, m["FinalError_r"].values
            seed_win += int(np.sum(ea < er))
            seed_loss += int(np.sum(ea > er))
            seed_tie += int(np.sum(ea == er))
            if mean_err.loc[f, a] < ref_pf[f]:
                wins += 1
            elif mean_err.loc[f, a] > ref_pf[f]:
                losses += 1
            else:
                ties += 1
            if len(ea) >= 5 and not np.allclose(ea, er):
                try:
                    _, p = stats.ranksum(ea, er)
                except Exception:
                    p = 1.0
            else:
                p = 1.0
            pvals.append(p)
            rawtbl[f] = p
        adj = holm(pvals)
        sig_better = int(np.sum((np.array(adj) < 0.05) & (mean_err[a].values < ref_pf.values)))
        sig_worse = int(np.sum((np.array(adj) < 0.05) & (mean_err[a].values > ref_pf.values)))
        rows.append({
            "Algorithm": a,
            "FunctionMean_wins": wins,
            "FunctionMean_losses": losses,
            "FunctionMean_ties": ties,
            "PairedSeed_wins": seed_win,
            "PairedSeed_losses": seed_loss,
            "PairedSeed_ties": seed_tie,
            "Holm_sig_better": sig_better,
            "Holm_sig_worse": sig_worse,
            "Holm_not_sig": len(funcs) - sig_better - sig_worse,
            "MeanErrorRatio_vs_ref": float(np.exp(np.median(
                np.log((mean_err[a].values + 1e-30) / (ref_pf.values + 1e-30))))),
        })
    pair = pd.DataFrame(rows).set_index("Algorithm")
    pair.to_csv(batch / "pairwise_vs_ref.csv")

    # ---------- markdown ----------
    def sci(v):
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.3e}"

    out = []
    out.append(f"# Screen report — {batch.name}\n")
    out.append(f"Reference for the pairwise tests: **{args.ref}**\n")
    out.append(f"- functions: {len(funcs)} ({funcs[0]}–{funcs[-1]})")
    out.append(f"- seeds: {len(seeds)} ({seeds[0]}–{seeds[-1]})")
    out.append(f"- algorithms: {len(algos)}; valid runs: {len(df)}")
    out.append(f"- Friedman chi2 = {chi2:.3f}, p = {fp:.3e}\n")

    out.append("## Average rank and cost\n")
    out.append("| Algorithm | Avg rank | Mean CPU s | Mean FE | Mean rounds |")
    out.append("|---|---:|---:|---:|---:|")
    for a in summary.sort_values("AverageRank").index:
        s = summary.loc[a]
        out.append(f"| {a} | {s.AverageRank:.3f} | {s.MeanSeconds:.2f} | "
                   f"{s.MeanEvaluations:.0f} | {s.MeanRounds:.0f} |")

    out.append(f"\n## Paired comparison against {args.ref}\n")
    out.append("| Algorithm | fn-mean W/L/T | paired-seed W/L/T | Holm sig. better/worse/ns | median error ratio |")
    out.append("|---|---:|---:|---:|---:|")
    for a in pair.sort_values("FunctionMean_wins", ascending=False).index:
        r = pair.loc[a]
        out.append(f"| {a} | {r.FunctionMean_wins}/{r.FunctionMean_losses}/{r.FunctionMean_ties} "
                   f"| {r.PairedSeed_wins}/{r.PairedSeed_losses}/{r.PairedSeed_ties} "
                   f"| {r.Holm_sig_better}/{r.Holm_sig_worse}/{r.Holm_not_sig} "
                   f"| {r.MeanErrorRatio_vs_ref:.3f} |")

    out.append("\n## Mean error per function\n")
    hdr = "| F | " + " | ".join(algos) + " |"
    out.append(hdr)
    out.append("|" + "---|" * (len(algos) + 1))
    for f in funcs:
        cells = []
        for a in algos:
            v = mean_err.loc[f, a]
            mark = ""
            if a != args.ref:
                mark = "+" if v < ref_pf[f] else ("−" if v > ref_pf[f] else "≈")
            cells.append(f"{sci(v)}{mark}")
        out.append(f"| F{f} | " + " | ".join(cells) + " |")
    out.append("\n*(+ better than the reference, − worse, ≈ tie; unmarked column = reference)*")

    (batch / "REPORT.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {batch/'REPORT.md'}")
    print(summary.sort_values("AverageRank").to_string())


if __name__ == "__main__":
    main()
