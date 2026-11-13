# Developing lowrank-split

The current development package is v0.2.0.dev4 at
`/homes/rzhu/data_splitting/low_rank`. Edit `lowrank/`, install with
`python -m pip install -e . --no-deps`, and run `python -m pytest tests -q`.
The public API is `from lowrank import split, SplitSpec`; support modules live
under `lowrank.splitters`, `lowrank.datasets`, `lowrank.features`, and `lowrank.eval`.

The prior ODAC25 working tree, including uncommitted release-preparation work,
now lives at `../maintenance/retired_workspaces/odac25_revision/`. Its package
copy is archival; make future development changes here. The frozen manuscript
producer remains v0.2.0.dev3 and must not be relabelled as current dev4 evidence.

The pre-existing `data/`, `paper/`, `results/`, `figures/`, `docs/`, and `archive/`
contain historical research assets. Their presence does not make old results
measurements of the current code. Old v0.1 runtime code and test suites were
replaced; no separate v0.1 checkout was retained.

Git history is retained on `main`. This standalone repository has no configured
remote. The PALM publication repository has its own remote and archived worktree.
