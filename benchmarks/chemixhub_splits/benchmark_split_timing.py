"""Partition-time comparison: Butina (paper) vs hypergraph vs low-rank.

Times ONLY the partitioning step on each task's unique-mixture set (the shared
mole-fraction-weighted-Morgan featurization is excluded — it is identical for all
three methods). One untimed warm-up call per engine absorbs the one-time
Mt-KaHyPar / CUDA / Nyström initialisation; each timed number is the best of 3.

Outputs:
  timing_report.csv
  figures/chemixhub_timing_bars.png     (per-task grouped bars, log y)
  figures/chemixhub_timing_scaling.png  (runtime vs #unique mixtures, log-log)

    CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    /homes/rzhu/miniforge3/envs/palm/bin/python benchmark_split_timing.py \
        --data-root <clone>/datasets
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_chemixhub_splits as M
from PALM.splitters import split, SplitSpec
from PALM.lowrank import nystrom_features, balanced_lloyd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

C_BUTINA = "#f4a36c"
C_HYPER = "#3b6fb6"
C_LOWRANK = "#5aa469"
REPEATS = 3


def best_of(fn, k=REPEATS):
    ts = []
    for _ in range(k):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def time_task(feature_data, X, sample_count_of_key, sorted_keys):
    """All timings are the COMPLETE production splitter (equal footing).

    ``lowrank_core`` is the bare Nyström-factor + one balanced assignment — the
    same primitive the OMol25 scaling plot timed. It is NOT equal footing with the
    single-shot hypergraph (which has no separable polish stage: Mt-KaHyPar's
    multilevel refinement is internal to its one call), so it is reported only as a
    reference for how much of low-rank's cost is the optional kmeans++/restart/FM
    polish rather than the core factorization.
    """
    spec = SplitSpec(splits=M.SPLITS, names=M.NAMES, seed=M.SEED, epsilon=0.10)

    def run_butina():
        clus = M.butina_clusters(X)
        cok = {sorted_keys[i]: int(clus[i]) for i in range(len(sorted_keys))}
        M.lpt_bin_pack(cok, sample_count_of_key, sorted_keys)

    def run_lowrank_core():
        B, _ = nystrom_features(X, rank=min(256, X.shape[0]), metric="tanimoto",
                                landmark="uniform", seed=M.SEED)
        balanced_lloyd(B, M.SPLITS, epsilon=0.10, n_iter=20, seed=M.SEED)

    return {
        "butina": best_of(run_butina),
        "hypergraph": best_of(lambda: split("hypergraph", feature_data, spec,
                                            metric="tanimoto", preset="quality")),
        "lowrank": best_of(lambda: split("lowrank", feature_data, spec, metric="tanimoto")),
        "lowrank_core": best_of(run_lowrank_core),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    # ---- warm up Mt-KaHyPar / CUDA / Nyström on a tiny throwaway problem ----
    warm = {str(i): np.eye(256, dtype=np.float32)[i % 256] for i in range(64)}
    ws = SplitSpec(splits=M.SPLITS, names=M.NAMES, seed=0, epsilon=0.2)
    split("hypergraph", warm, ws, metric="tanimoto", preset="quality")
    split("lowrank", warm, ws, metric="tanimoto")
    M.butina_clusters(np.vstack(list(warm.values())))
    print("warm-up done\n", flush=True)

    rows = []
    for dset, (csv_name, id_cols) in M.DATASETS.items():
        if args.only and dset not in args.only:
            continue
        ddir = os.path.join(args.data_root, dset, "processed_data")
        df = pd.read_csv(os.path.join(ddir, f"{csv_name}.csv"))
        fp_map, is_salt = M.load_compounds(os.path.join(ddir, "compounds.csv"))
        props = df["property"].unique() if "property" in df.columns else ["value"]
        for prop in props:
            sub = df[df["property"] == prop].reset_index(drop=True) if "property" in df.columns else df
            _, samples_of_key, feature_data = M.build_unique_mixtures(sub, id_cols, fp_map, is_salt)
            sorted_keys = sorted(feature_data.keys())
            X = np.vstack([feature_data[k] for k in sorted_keys])
            scount = {str(k): len(v) for k, v in samples_of_key.items()}
            t = time_task(feature_data, X, scount, sorted_keys)
            rows.append({"dataset": dset, "property": prop, "n_samples": len(sub),
                         "n_unique": len(sorted_keys), **t})
            print(f"{dset:22s} {prop:24s} n_uniq={len(sorted_keys):6d}  "
                  f"butina={t['butina']*1e3:7.1f}  hyper(full)={t['hypergraph']*1e3:7.1f}  "
                  f"lowrank(full)={t['lowrank']*1e3:7.1f}  lowrank(core)={t['lowrank_core']*1e3:6.1f} ms",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "timing_report.csv"), index=False)
    plot(df)


def plot(df):
    from make_comparison_chart import SHORT
    tasks = [(r.dataset, r.property) for r in df.itertuples()]
    labels = [SHORT[k] for k in tasks]
    x = np.arange(len(df)); w = 0.26

    fig, ax = plt.subplots(figsize=(15, 6))
    # three non-overlapping grouped bars — the full-vs-full equal-footing
    # comparison: every engine gets the identical embedding input and runs its
    # complete production pipeline end-to-end.
    ax.bar(x - w, df["butina"] * 1e3, w, label="Butina — paper chem-OOD", color=C_BUTINA)
    ax.bar(x, df["hypergraph"] * 1e3, w, label="Hypergraph (full — single Mt-KaHyPar cut)", color=C_HYPER)
    ax.bar(x + w, df["lowrank"] * 1e3, w, label="Low-rank (full — Nyström + 4×restart Lloyd + FM)", color=C_LOWRANK)
    ax.set_yscale("log")
    # headroom above the tallest bar so nothing is clipped and the legend (placed
    # above the axes) never overlaps a bar
    top = float(df[["butina", "hypergraph", "lowrank"]].to_numpy().max()) * 1e3
    ax.set_ylim(0.2, top * 2.5)
    ax.set_ylabel("partition time (ms, log scale) — lower is better", fontsize=12)
    ax.set_title("CheMixHub chem-OOD partition time — full vs full "
                 "(same embeddings in, each run end-to-end; featurization shared/excluded; best of 3, warm)",
                 fontsize=12, pad=34)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9.5, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=3, framealpha=0.95, columnspacing=1.8)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGS, "chemixhub_timing_bars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)

    # scaling: runtime vs #unique mixtures
    fig, ax = plt.subplots(figsize=(8, 6))
    o = df.sort_values("n_unique")
    for col, c, lab, style in (("butina", C_BUTINA, "Butina (full)", "o-"),
                               ("hypergraph", C_HYPER, "Hypergraph (full)", "o-"),
                               ("lowrank", C_LOWRANK, "Low-rank (full)", "o-")):
        ax.plot(o["n_unique"], o[col] * 1e3, style, color=c, label=lab, ms=5, lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("# unique mixtures (partition size)", fontsize=12)
    ax.set_ylabel("partition time (ms, log)", fontsize=12)
    ax.set_title("Partition-time scaling — full vs full (same embeddings in, each run end-to-end)", fontsize=11)
    ax.legend(fontsize=10); ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGS, "chemixhub_timing_scaling.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
