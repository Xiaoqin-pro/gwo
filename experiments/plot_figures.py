#!/usr/bin/env python
"""Publication figures for a RASE-GWO batch.

    python experiments/plot_figures.py results/<batch_dir> [--ref GWO]

Writes convergence_<F>.pdf/png for the requested functions, ranks.pdf and
error_ratio_heatmap.pdf into <batch_dir>/figures/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--ref", default="GWO")
    ap.add_argument("--funcs", nargs="*", default=None)
    ap.add_argument("--file", default="runs.csv")
    ap.add_argument("--subdir", default="")
    args = ap.parse_args()
    batch = Path(args.batch)
    figdir = batch / "figures" / args.subdir if args.subdir else batch / "figures"
    figdir.mkdir(exist_ok=True)

    runs = pd.read_csv(batch / args.file)
    runs = runs[runs.Status == "ok"]
    algos = list(dict.fromkeys(runs.Algorithm))
    funcs = sorted(runs.FunctionID.unique())
    if args.funcs:
        funcs = [int(f) for f in args.funcs]

    mean_err = runs.groupby(["FunctionID", "Algorithm"])["FinalError"].mean().unstack()
    mean_err = mean_err.reindex(index=sorted(runs.FunctionID.unique()), columns=algos)
    ranks = mean_err.rank(axis=1)
    avg = ranks.mean(axis=0).sort_values()

    # ---- average rank bar chart ----
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    cols = ["#c0392b" if a == args.ref else "#4a6fa5" for a in avg.index]
    ax.barh(range(len(avg)), avg.values, color=cols)
    ax.set_yticks(range(len(avg)), avg.index)
    ax.invert_yaxis()
    ax.set_xlabel("average rank across functions (lower is better)")
    fig.savefig(figdir / "ranks.pdf")
    fig.savefig(figdir / "ranks.png")
    plt.close(fig)

    # ---- error-ratio heatmap vs reference ----
    ratio = np.log10((mean_err.values + 1e-12) / (mean_err[args.ref].values[:, None] + 1e-12))
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(algos), 0.28 * len(mean_err) + 1.6))
    v = np.nanmax(np.abs(ratio))
    im = ax.imshow(ratio, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(algos)), algos, rotation=45, ha="right")
    ax.set_yticks(range(len(mean_err)), [f"F{f}" for f in mean_err.index])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=f"log10(mean error / {args.ref})")
    fig.savefig(figdir / "error_ratio_heatmap.pdf")
    fig.savefig(figdir / "error_ratio_heatmap.png")
    plt.close(fig)

    # ---- convergence curves ----
    use_traces = (args.file == "runs.csv")
    tf = batch / "traces_v7.mat"
    if use_traces and not tf.exists():
        tf = batch / "traces.mat"
    if use_traces and tf.exists():
        tr = loadmat(tf)
        T = tr["Traces"]
        fe = tr["traceFE"].ravel()
        T = np.where(np.isfinite(T), T, np.nan)
        for f in funcs:
            sel = runs.index[(runs.FunctionID == f)].to_numpy()
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            for a in algos:
                idx = runs.index[(runs.FunctionID == f) & (runs.Algorithm == a)].to_numpy()
                if len(idx) == 0:
                    continue
                sub = T[:, idx]
                med = np.nanmedian(sub, axis=1)
                ax.plot(fe, med, label=a, lw=1.2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("true function evaluations (incl. initialization)")
            ax.set_ylabel(f"median error, F{f}")
            ax.legend(fontsize=7)
            fig.savefig(figdir / f"convergence_F{f}.pdf")
            fig.savefig(figdir / f"convergence_F{f}.png")
            plt.close(fig)
    else:
        print("no traces.mat; convergence figures skipped")

    print(f"figures -> {figdir}")
    print(avg.to_string())


if __name__ == "__main__":
    main()
