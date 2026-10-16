#!/usr/bin/env python
"""Merge several experiment batches into one balanced result table.

    python experiments/merge_batches.py results/batchA results/batchB ... \
        --out results/merged_30seed/runs.csv

Why this is safe: every batch of this project draws its initial point pool from
initSeed = 261013000 + functionID*10000 + seed and consumes a prefix of it, so
two runs with the same (function, seed, algorithm) are meant to be *identical*.
This script therefore also acts as a reproducibility audit: whenever the same
triple appears in more than one batch it compares the recorded final errors and
reports the worst relative difference.  A non-zero difference means the algorithm
source changed in a way that affects that configuration, which must be known
before the two batches are averaged together.

Also reports the coverage grid, so a partially filled table is obvious rather
than silently averaged.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


def load_one(batch: Path) -> pd.DataFrame:
    f = batch / "runs.csv"
    if not f.exists():
        raise SystemExit(f"missing {f}")
    df = pd.read_csv(f)
    df = df[df.Status == "ok"].copy()
    df["__batch"] = batch.name
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="relative tolerance for the reproducibility audit")
    ap.add_argument("--traces", action="store_true",
                    help="also merge the per-run trajectories into traces.mat")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="batch:Algorithm entries to exclude, e.g. "
                         "perf29_...:IGWO -- used when a later batch supersedes "
                         "an implementation that was found to be wrong")
    args = ap.parse_args()

    parts = [load_one(Path(b)) for b in args.batches]
    for spec in args.drop:
        bn, an = spec.split(":")
        for p in parts:
            if len(p) and p.__batch.iloc[0] == bn:
                n = int((p.Algorithm == an).sum())
                if n:
                    p.drop(p.index[p.Algorithm == an], inplace=True)
                    print(f"dropped {n} runs: {bn}:{an}")
    for p in parts:
        print(f"{p.__batch.iloc[0] if len(p) else '?':28s} {len(p):6d} runs, "
              f"{p.Algorithm.nunique()} algorithms, {p.FunctionID.nunique()} functions, "
              f"{p.Seed.nunique()} seeds")

    # ---- reproducibility audit on overlapping triples ----------------------
    allr = pd.concat(parts, ignore_index=True)
    key = ["FunctionID", "Seed", "Algorithm"]
    dup = allr[allr.duplicated(key, keep=False)]
    if len(dup):
        g = dup.groupby(key).FinalError
        spread = g.max() - g.min()
        scale = g.max().abs().clip(lower=1e-30)
        rel = (spread / scale)
        worst = float(rel.max())
        n_over = int(len(rel))
        n_same = int((spread == 0).sum())
        print(f"\nreproducibility audit: {n_over} overlapping triples, "
              f"{n_same} bit-identical, worst relative difference {worst:.3e}")
        if worst > args.tol:
            bad = rel[rel > args.tol].sort_values(ascending=False).head(5)
            print("  WARNING: batches disagree beyond tolerance:")
            print(bad.to_string())
    else:
        print("\nreproducibility audit: no overlapping triples")

    merged = allr.drop_duplicates(subset=key, keep="first").copy()
    funcs = sorted(merged.FunctionID.unique())
    seeds = sorted(merged.Seed.unique())
    algos = list(dict.fromkeys(merged.Algorithm))
    full = len(funcs) * len(seeds)
    print(f"\nmerged grid: {len(algos)} algorithms x {len(funcs)} functions x "
          f"{len(seeds)} seeds (full grid = {full} runs per algorithm)")
    cover = merged.groupby("Algorithm").size().sort_values(ascending=False)
    for a, n in cover.items():
        flag = "" if n == full else "   <-- INCOMPLETE"
        print(f"  {a:28s} {n:6d}{flag}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.drop(columns=["__batch"]).to_csv(out, index=False)
    print(f"\nwrote {out}")

    if args.traces:
        # align each batch's trajectory columns with the merged row order
        keys = list(zip(merged.FunctionID, merged.Seed, merged.Algorithm))
        T = None
        fe = None
        for batch, part in zip([Path(b) for b in args.batches], parts):
            f = batch / "traces_v7.mat"
            if not f.exists():
                f = batch / "traces.mat"
            if not f.exists():
                print(f"  (no trajectories in {batch.name}, skipped)")
                continue
            M = loadmat(f)
            Ti = np.where(np.isfinite(M["Traces"]), M["Traces"], np.nan)
            fei = M["traceFE"].ravel()
            idx = {(r.FunctionID, r.Seed, r.Algorithm): i
                   for i, r in enumerate(part.itertuples())}
            if T is None:
                T = np.full((len(fei), len(keys)), np.nan)
                fe = fei
            elif len(fei) != len(fe):
                print(f"  (trace grid mismatch in {batch.name}, skipped)")
                continue
            hit = 0
            for j, k in enumerate(keys):
                i = idx.get(k)
                if i is not None and i < Ti.shape[1]:
                    T[:, j] = Ti[:, i]
                    hit += 1
            print(f"  trajectories from {batch.name}: {hit} columns")
        if T is not None:
            savemat(out.parent / "traces.mat", {"Traces": T, "traceFE": fe})
            print(f"wrote {out.parent / 'traces.mat'}  ({T.shape[0]} x {T.shape[1]})")


if __name__ == "__main__":
    main()
