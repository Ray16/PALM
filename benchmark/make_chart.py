"""Consolidated benchmark + grouped bar chart: hypergraph vs DataSAIL vs random
vs paper DataSAIL-S1, all scored with the GPU L(pi) (validated == eval_split).

Hypergraph and random are recomputed here (deterministic after the Mt-KaHyPar
init fix); DataSAIL and paper values are from the earlier runs / addendum.
"""

import os
import random
import signal
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
import logging
logging.disable(logging.CRITICAL)

from ..hypergraph import run_hypergraph_split
from .leakage import scaled_lpi

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "DataSAIL_data", "1D", "moleculenet")

# small -> large
ORDER = ["freesolv", "esol", "clintox", "sider", "bace", "bbbp",
         "lipophilicity", "tox21", "qm8", "hiv", "muv"]
COL = {"bace": "mol"}
# DataSAIL fresh (v1.3.0); None = timed out (>600s) at this scale
DS = {"freesolv": 0.1424, "esol": 0.1668, "clintox": 0.2294, "sider": 0.2360,
      "bace": 0.2387, "bbbp": 0.2319, "lipophilicity": 0.2718, "tox21": 0.2230,
      "qm8": 0.2077, "hiv": None, "muv": None}
PAPER = {"freesolv": 0.1410, "esol": 0.1808, "clintox": 0.2303, "sider": 0.2345,
         "bace": 0.3036, "bbbp": 0.2866, "lipophilicity": 0.3027, "tox21": 0.2224,
         "qm8": 0.2918, "hiv": 0.3071, "muv": 0.3143}
# DataSAIL wall-clock (s). Known expensive runs reused; small ones re-timed; None=timeout(>600s).
DS_TIME = {"freesolv": 8.2, "esol": 16.2, "sider": 6.6, "bace": 306.0,
           "lipophilicity": 323.0, "qm8": 742.0, "hiv": None, "muv": None}
RETIME = {"clintox", "bbbp", "tox21"}     # re-time DataSAIL (prefiltered) for these


class _TO:
    def __init__(self, t): self.t = t
    def __enter__(self): signal.signal(signal.SIGALRM, self._h); signal.alarm(self.t)
    def __exit__(self, *a): signal.alarm(0)
    def _h(self, *a): raise TimeoutError()


def _datasail_time(valid):
    from datasail.sail import datasail
    t0 = time.time()
    try:
        with _TO(400):
            datasail(techniques=["C1e"], splits=[8, 2], names=["train", "test"],
                     e_type="M", e_data={s: s for s in valid}, max_sec=200)
        return round(time.time() - t0, 1)
    except Exception:
        return None


def morgan(sm):
    X = np.zeros((len(sm), 2048), dtype=np.int8)
    for i, s in enumerate(sm):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), X[i])
    return X


def compute():
    rows = []
    for ds in ORDER:
        col = COL.get(ds, "smiles")
        df = pd.read_csv(os.path.join(DATA, f"{ds}.csv")).dropna(subset=[col]).drop_duplicates(col).reset_index(drop=True)
        sm = [s for s in df[col].astype(str) if s and s != "nan"]
        n = len(sm)
        fd = {sm[i]: v for i, v in enumerate(morgan(sm))}
        t0 = time.time()
        hg = run_hypergraph_split(fd, [8, 2], ["train", "test"], k=15, preset="quality")
        hg_t = round(time.time() - t0, 2)
        hg_l, _ = scaled_lpi(sm, hg)
        random.seed(42)
        ids = list(sm); random.shuffle(ids); cut = int(0.8 * n)
        rnd = {**{i: "train" for i in ids[:cut]}, **{i: "test" for i in ids[cut:]}}
        r_l, _ = scaled_lpi(sm, rnd)
        ds_t = _datasail_time([s for s in sm if Chem.MolFromSmiles(s) is not None]) if ds in RETIME else DS_TIME[ds]
        rows.append({"dataset": ds, "n": n, "hypergraph": round(hg_l, 4), "hg_time": hg_t,
                     "datasail": DS[ds], "ds_time": ds_t, "random": round(r_l, 4), "paper_s1": PAPER[ds]})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def plot(df):
    labels = [f"{r.dataset}\n(n={r.n:,})" for r in df.itertuples()]
    x = np.arange(len(df)); w = 0.2
    fig, ax = plt.subplots(figsize=(15, 6))
    series = [("hypergraph", "#2563eb"), ("datasail", "#dc2626"),
              ("paper_s1", "#d97706"), ("random", "#94a3b8")]
    names = {"hypergraph": "Hypergraph (ours)", "datasail": "DataSAIL (fresh)",
             "random": "Random", "paper_s1": "Paper DataSAIL-S1"}
    def _fmt(t):
        if t is None or (isinstance(t, float) and pd.isna(t)):
            return None
        return f"{t:.1f}s" if t < 100 else f"{t:.0f}s"

    for j, (col, c) in enumerate(series):
        vals = [v if v is not None and not pd.isna(v) else 0 for v in df[col]]
        bars = ax.bar(x + (j - 1.5) * w, vals, w, label=names[col], color=c)
        # runtime annotations above the bars that actually run (hypergraph, DataSAIL)
        time_col = {"hypergraph": "hg_time", "datasail": "ds_time"}.get(col)
        if time_col:
            for i, b in enumerate(bars):
                t = _fmt(df[time_col].iloc[i])
                if t is not None:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, t,
                            rotation=90, ha="center", va="bottom", fontsize=7,
                            color=c, fontweight="bold")
                elif col == "datasail":
                    ax.text(x[i] + (j - 1.5) * w, 0.012, "timeout", rotation=90,
                            ha="center", va="bottom", fontsize=7, color="#dc2626", fontweight="bold")
    ax.set_ylabel("scaled L(π)  (lower = less leakage)", fontsize=12)
    ax.set_title("Train/test leakage: Hypergraph vs DataSAIL  (80/20, MoleculeNet)", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10, framealpha=0.9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "benchmark_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    df = compute()
    df.to_csv(os.path.join(HERE, "final_results.csv"), index=False)
    plot(df)
