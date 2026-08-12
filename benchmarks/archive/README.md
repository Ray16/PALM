# Archived benchmark scripts

These scripts are **superseded** and kept only for provenance. They were moved
here as-is and were **not** repointed to the new `PALM.splitters` API, so their
imports still reference the deleted `PALM.hypergraph` / `PALM.lowrank_split`
modules and will not run unchanged.

| script | superseded by |
| --- | --- |
| `omol25_embed.py` | `../omol25/uma_embed.py` (UMA backbone embedding) |
| `omol25_split.py` | `../omol25/uma_split.py` + `PALM.splitters` `split("lowrank", …)` |
| `omol25_datasail_scaling.py` | `../omol25/omol25_datasail_finemode.py` |
| `replot_scaling_300dpi.py` | in-script replotting in `../omol25/omol25_scaling.py` / `uma_scaling.py` |
| `derisk_lowrank.py` | `../moleculenet/validate_lowrank.py` |
| `test_lowrank_split.py` | `PALM/splitters/tests/test_splitters.py` (correctness checks ported) |
