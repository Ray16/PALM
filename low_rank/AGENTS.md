# Low-rank development

This directory is the canonical current lowrank-split development package.
Use this version for reviews and new work. Do not edit or import an archived
package as the current implementation. Frozen publication producers retain
their original versions and numerical evidence.

Work on main, preserve unrelated changes, and run the package tests for code
changes. Use namespaced imports (`lowrank.splitters`, `lowrank.datasets`, etc.).
CPU checks on shared hosts should pin a small CPU set and disable CUDA.
