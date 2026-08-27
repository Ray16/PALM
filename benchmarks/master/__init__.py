"""PALM master benchmark.

One unified sweep over every configured dataset (``PALM.data.sources.REGISTRY``)
plus the CheMixHub mixture suite, running every applicable splitter across
multiple seeds and recording, in a single long-format table:

- **split quality** — leakage L(pi), imbalance, realized test fraction, runtime;
- **generalization gap** — a fixed light model (RandomForest) trained on each
  split's train bucket and scored on its test bucket (ROC-AUC / R^2), plus the
  train-minus-test gap.

The results table (``benchmarks/results/master_benchmark.csv``) is the artifact
to come back to and derive insights from; ``analyze.py`` turns it into figures
and ``INSIGHTS.md``.
"""
