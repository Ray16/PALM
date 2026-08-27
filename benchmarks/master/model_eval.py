"""Generalization-gap layer: train a fixed model on a split and score it.

Given a split's train/test entity positions and the dataset's features + target,
fit a RandomForest (classification -> ROC-AUC, regression -> R^2) and report the
test metric, the train metric, and their gap (train - test).

The gap is the honest question the benchmark asks: a leakage-minimizing split
should make the task *harder* — lower test metric, larger gap — than a random
split, because the held-out entities are genuinely less similar to training. The
model and featurization are held **fixed** across splitters, so the only thing
that varies is the partition; differences in the gap are attributable to the
split, not the model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Fixed, cheap, deterministic. n_jobs is CPU-side (the splitters own the GPU).
RF_KW = dict(n_estimators=100, n_jobs=8)

# Minimum labeled rows to bother fitting / to trust a score.
MIN_TRAIN, MIN_TEST = 10, 5


def _empty(reason=""):
    return {"model": "", "metric_name": "", "train_metric": "", "test_metric": "",
            "gen_gap": "", "n_train_lab": "", "n_test_lab": "", "gap_reason": reason}


def evaluate_gap(X: np.ndarray, y: np.ndarray, task_type: str,
                 train_pos: Sequence[int], test_pos: Sequence[int], seed: int = 0) -> dict:
    """Fit on ``train_pos`` rows, score on ``test_pos`` rows; return a metrics dict.

    NaN labels are masked out (kept in the split, excluded from fit/score). Returns
    a ``gap_reason`` instead of numbers when a gap cannot be computed (no target,
    too few labels, single-class train, ...), so the driver records *why*.
    """
    if task_type not in ("classification", "regression"):
        return _empty("no target")

    train_pos = np.asarray(list(train_pos), dtype=int)
    test_pos = np.asarray(list(test_pos), dtype=int)
    Xtr, ytr = X[train_pos], y[train_pos]
    Xte, yte = X[test_pos], y[test_pos]
    mtr, mte = np.isfinite(ytr), np.isfinite(yte)
    Xtr, ytr, Xte, yte = Xtr[mtr], ytr[mtr], Xte[mte], yte[mte]

    out = _empty()
    out["n_train_lab"], out["n_test_lab"] = int(len(ytr)), int(len(yte))
    if len(ytr) < MIN_TRAIN or len(yte) < MIN_TEST:
        out["gap_reason"] = f"too few labeled (train={len(ytr)}, test={len(yte)})"
        return out

    try:
        if task_type == "classification":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score
            if len(np.unique(ytr)) < 2:
                out["gap_reason"] = "train single-class"
                return out
            clf = RandomForestClassifier(random_state=seed, **RF_KW).fit(Xtr, ytr)
            out["model"] = "rf_classifier"
            if len(np.unique(yte)) >= 2:
                out["metric_name"] = "roc_auc"
                tr = roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1])
                te = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
            else:                                   # test degenerate -> AUC undefined
                out["metric_name"] = "accuracy"
                tr = accuracy_score(ytr, clf.predict(Xtr))
                te = accuracy_score(yte, clf.predict(Xte))
        else:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score
            reg = RandomForestRegressor(random_state=seed, **RF_KW).fit(Xtr, ytr)
            out["model"], out["metric_name"] = "rf_regressor", "r2"
            tr = r2_score(ytr, reg.predict(Xtr))
            te = r2_score(yte, reg.predict(Xte))
        out["train_metric"] = round(float(tr), 4)
        out["test_metric"] = round(float(te), 4)
        out["gen_gap"] = round(float(tr - te), 4)
    except Exception as exc:                        # noqa: BLE001 — record, don't crash the sweep
        out["gap_reason"] = f"{type(exc).__name__}: {exc}"
    return out
