# Low-rank development

This directory is the canonical current lowrank-split development package.
Use this version for reviews and new work. Do not edit or import an archived
package as the current implementation. Frozen publication producers retain
their original versions and numerical evidence.

Work on main, preserve unrelated changes, and run the package tests for code
changes. Use namespaced imports (`lowrank.splitters`, `lowrank.datasets`, etc.).
CPU checks on shared hosts should pin a small CPU set and disable CUDA.

## Mandatory GitHub Push Policy
Once code modifications are completed and verified by tests:
1. Commit the changes in `low_rank` (the `post-commit` hook syncs `lowrank/`, `tests/`, and `CHANGELOG.md` into `../PALM/low_rank/`).
2. Commit the synced changes in `../PALM` and push to GitHub (`origin/main`; enforced by `PALM/.git/hooks/post-commit` and `.agents/hooks.json`).
