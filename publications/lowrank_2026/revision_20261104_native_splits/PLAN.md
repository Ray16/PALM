# Metadata-only native-split large datasets

The user extended the active manuscript task on 4 November: identify large
datasets beyond ODAC25, preferably with native splits, and compare overlap and
splitting costs without downloading hundreds of gigabytes. Preserve the completed
controlled-optimization checkpoint; extend its evidence after this study is
verified. Do not replace or relabel earlier measurements.

## Initial candidates and decision rules

Use primary publisher/maintainer documentation and inspect actual transfer sizes
before downloading. Prefer independently identified molecules with SMILES and
published assignment indices; configuration count is not independent entity
count. Initial candidates are PCQM4Mv2 (native CID split, four partitions),
MOSES (train, random test and scaffold test), and ogbg-molpcba (native scaffold
split). Molecule3D remains a candidate only if compact original identities and
native assignments can be obtained without downloading structures; it overlaps
the PubChemQC source of PCQM4Mv2 and is not automatically an independent corpus.

Hard initial transfer budget: 200 MiB total data/metadata payload, with no
single metadata object larger than 128 MiB. Read only required members of ZIP
archives using HTTP ranges where available. Fail closed if a server ignores a
range or the declared size exceeds budget. Download no coordinates, SDFs,
trajectory archives, processed graph bundles or model checkpoints. Metadata
and locally generated packed fingerprints are cached outside the public bundle.

## Required evidence

1. Pinned source/version, native split semantics and sizes; actual metadata bytes,
   hashes and a reproducible selective-download recipe.
2. Validate row identities, split disjointness and coverage. Keep native partition
   names and counts. Record any unparseable SMILES and evaluate all methods on the
   same declared retained rows; do not hide removal or merge official test sets.
3. Featurize SMILES using the primary molecular representation, without reading
   response labels for optimization. Retain packed binary fingerprints and
   source-to-retained-row identities; avoid permanent dense feature/graph caches.
4. Compare native assignments, count-matched random assignments and the current
   low-rank implementation on identical entities, with independent original-
   Tanimoto global overlap and per-held-out-fold nearest-training exposure.
   At large n use declared paired sampling/uncertainty for global overlap and
   uniformly sampled queries compared with the entire training fold.
5. Separate source-native assignment loading from split generation. Published
   native construction times are unknown unless an actual source recipe is
   rerun. Do not call file-loading time a competing optimization runtime.
   Measure low-rank split-call time and fresh-process RSS separately from
   metadata loading, fingerprint construction and independent scoring.
6. Freeze numerical settings before production; bounded CPU allocations,
   source/input manifests, ownership-aware claims and resumable complete outputs.
   Retain explicit budget failures rather than invent timings or restart silently.
7. Add verified results and a concise native-split comparison to the manuscript
   when justified. Preserve the clear four-question narrative, native-split
   limitations, existing producer versions and author-declaration placeholders.
   Build/review and synchronize GitHub main and Overleaf under the standing rule.

The precise production grid is declared after metadata/identity preflight and
bounded resource pilots, before examining production outcomes. Numerical method
changes are outside this extension unless a reproducible defect requires a fix.
