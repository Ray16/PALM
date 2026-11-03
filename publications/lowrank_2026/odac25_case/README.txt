ODAC25 publication evidence
==========================

This is the frozen metadata-only ODAC25 case imported on 2026-11-03. It is
separate from the primary v0.2.0.dev3 studies. Its weighted adapter uses the
included, original v0.1 factor implementation under method_source/lowrank.
Every imported artifact is covered by evidence_manifest.json. PROVENANCE.json
retains the pinned dataset URLs, per-shard selected columns and transfer ledger.
Comparison and scientific-audit directories retain controls, assignments and
interpretation limits. Eight metadata tests and three weighted-comparison tests
are included. The paper generator checks the frozen evidence before use.

The approximately 1.2-GiB expanded per-shard metadata caches are omitted. No
coordinates, forces or energy labels are required to reproduce the reported
composition evaluations. To regenerate the omitted catalogue, collect.py reads
only the same selected columns from the pinned remote shards (63.4 MB metered
production payload, with small pilot/API requests additional).

Offline replay uses a separate output directory and explicitly loads the
included numerical implementation. From the publication directory:

  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python odac25_case/replay.py --stage evaluate --output /tmp/odac25-replay

Use --resume only for intentionally reusing an existing output directory. Other
stages are collect, split, compare, audit and finalize. Split requires the
expanded catalogue; compare/evaluate use the small retained group/formula tables.
The original producer source files are unchanged; replay.py adjusts only the
method-source import path and the package-source hashing root for relocation.

USAGE.txt and comparison_original/USAGE.txt preserve the original workspace
commands as historical run documentation. Use replay.py for portable commands
from this publication bundle. The current numerical metrics use exact original
train/validation configuration counts, rather than the earlier 80/10/10 group
split. Neither the case nor its reported optimizer timing establishes a speedup
over dense methods at 2,050 group representatives.
