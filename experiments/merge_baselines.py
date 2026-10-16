#!/usr/bin/env python
"""Merge an external baseline batch into a RASE-GWO batch.

    python experiments/merge_baselines.py results/<my_batch> \
        --external "D:/111/Desktop/噜噜/9.29/CEC2017_8Algorithm_30Seed/results/main_8algorithms/all_quality_runs.csv" \
        --funcs 1 3-30 --seeds 3001-3030

The external file must come from a run that used the same CEC2017 build and the
same initial-population protocol (initSeed = 261013000 + fid*10000 + seed, a
50-point pool consumed as a prefix).  That was verified pointwise for the
9.29 batch: its recorded InitialCost for F1/seed 3001 reproduces exactly.
Runs are only merged when (function, seed) pairs are shared, so the paired
tests stay legal.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_spec(tokens):
    out = []
    for t in tokens:
        t = str(t)
        if "-" in t and not t.startswith("-"):
            a, b = t.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--external", required=True)
    ap.add_argument("--out", default="merged_runs.csv")
    ap.add_argument("--funcs", nargs="+", default=["1", "3-30"])
    ap.add_argument("--seeds", nargs="+", default=["3001-3030"])
    args = ap.parse_args()

    batch = Path(args.batch)
    mine = pd.read_csv(batch / "runs.csv")
    ext = pd.read_csv(args.external)
    funcs = set(parse_spec(args.funcs))
    seeds = set(parse_spec(args.seeds))

    mine = mine.rename(columns={"Algorithm": "Algorithm"})
    mine["Source"] = "rase_gwo"
    ext = ext.rename(columns={"Condition": "Algorithm"})
    ext["Source"] = "external_9.29"

    keep = ["FunctionID", "Seed", "Algorithm", "FinalCost", "FinalError",
            "Evaluations", "Seconds", "Status", "Source"]
    for df in (mine, ext):
        for c in keep:
            if c not in df.columns:
                df[c] = pd.NA

    both = pd.concat([mine[keep], ext[keep]], ignore_index=True)
    both = both[both.FunctionID.isin(funcs) & both.Seed.isin(seeds)]
    both = both[both.Status == "ok"]

    # only keep external algorithms that actually cover the whole grid
    n_f = len(funcs)
    n_s = len(seeds)
    counts = both.groupby("Algorithm").size()
    full = counts[counts == n_f * n_s].index.tolist()
    dropped = sorted(set(counts.index) - set(full))
    if dropped:
        print("dropped (incomplete grid):")
        for d in dropped:
            print(f"   {d}: {counts[d]}/{n_f*n_s}")
    both = both[both.Algorithm.isin(full)]

    out = batch / args.out
    both.to_csv(out, index=False)
    print(f"merged {len(both)} runs, {len(full)} algorithms -> {out}")
    print(both.groupby("Algorithm").size().to_string())


if __name__ == "__main__":
    main()
