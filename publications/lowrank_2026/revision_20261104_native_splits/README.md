# Large datasets with native splits, using metadata only

This extension addresses the user's request to add large datasets beyond ODAC25
and compare separation and splitting cost without downloading structure archives.
The earlier controlled-methods manuscript remains preserved in GitHub main
commit `46268a6` and Overleaf commit `1d75416c`.

| Dataset | Original records | Native roles | Metadata source |
|---|---:|---|---|
| PCQM4Mv2 | 3,746,620 | train / valid / test-dev / test-challenge | [OGB-LSC](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) SMILES CSV and supplied indices |
| MOSES | 1,936,962 | train / test / test_scaffolds | [Maintainer repository](https://github.com/molecularsets/moses) three compressed SMILES files |
| ogbg-molpcba | 437,929 | train / valid / test | [OGB](https://ogb.stanford.edu/docs/graphprop/) molecule mapping and scaffold indices |

PCQM4Mv2's released split is **CID-based**, as documented by OGB; it is not the
intended scaffold split. Its four partitions remain separate. MOSES has a random
test and a separate scaffold test, which serve different generation-evaluation
roles. Native scaffold/CID boundaries are not automatically the same feasible
partition set or scientific task as entity-level aggregate-cut optimization.

The three collections required **90,293,593 metered dataset-payload bytes**
(90.3 MB), including ZIP catalog ranges. Small source-discovery pages/JSON and
HTTP headers are outside that payload tally. No coordinates, SDFs, trajectories,
processed graph arrays or model weights were retrieved. `transfer_ledger.json`
records each range/body request; the hard initial dataset-payload cap is 200 MiB.

`fetch_metadata.py` reads only required ZIP members via HTTP ranges. ZIP CRCs
and extracted-member SHA-256 hashes are retained. MOSES files are fetched from
an immutable upstream commit and checked against their Git-LFS SHA-256 objects.
SMILES and partition identities suffice for the chosen molecular representation.
The property/assay columns bundled with some CSVs are not loaded for optimization.

RDKit preparation retains 3,746,581 PCQM4Mv2 records (39 parsing failures), all
1,936,962 MOSES records, and 437,927 ogbg-molpcba records (two failures).
Compact original-to-retained-row maps and every parsing failure are published
in `preparation/excluded_row_maps.json`; their reconstructed array hashes are checked.
All methods use the same retained rows and native counts. Canonical identities
are audited without silently deduplicating records. Fingerprint identity is
distinct from exact chemical identity or identical geometry.

Packed radius-two, 1,024-bit fingerprints are generated locally. Their compact
files and metadata caches are ignored by Git; numerical source, manifests,
download recipes, checksums, assignments and evaluation artifacts are the public
reproducibility evidence. `preparation/*.json` records source hashes, row counts,
fingerprinting cost, native coverage and canonical-identity diagnostics.

The frozen package's ordinary SplitSpec wrapper accepts two or three roles.
`native_spec.py` is a separately hashed, validated four-role subclass that passes
PCQM4Mv2's native roles to the unchanged k-way numerical routines. Five metric
and four-role adapter regression tests pass. The first API preflight stopped
in warm-up and is preserved in `run_api_preflight_v1`/`preflight_history`.

All 30 production cases complete successfully; `verification/completion.json`
records source/assignment/metric checks, 3,072 independent RDKit pair checks,
168 independent NumPy queries against complete training folds, and 18
returned-partition fidelity audits. The main paper contains one compact native
comparison; per-role scores, intervals and complete outcomes are in the supplement.

The production grid and resource/scoring rules are declared in `PROTOCOL.md`.
Native construction times are unknown: loading an assignment file is never
reported as a competing split-generation runtime. Global overlap uses common
uniform pair samples; sampled held-out queries are each searched against the
complete training fold using exact original Tanimoto. Source/input manifests
and CPU ownership records accompany each fresh process.

Molecule3D was screened as a further candidate. Its original corpus overlaps
the PubChemQC source of PCQM4Mv2, and readily accessible mirrors bundle 3D
structures or alter parsing/version details. It is therefore not added as an
independent large-data success without a separate compact-metadata/version audit.

## Reproduce and rebuild

Rebuild the paper from saved observations without downloading datasets:

```bash
MLST_PYTHON=python bash ../paper/build.sh
```

For numerical replay, use a separate checkout/copy and the pinned
Python/RDKit/PyTorch/compiler versions. Preserve its supplied `run` and
`preparation` folders under distinct archival names before generating new
observations: preparation records include measured timings and will differ
between machines. Restore the exact metadata, rather than resolving a new
upstream revision:

```bash
python reproduce_inputs.py
CUDA_VISIBLE_DEVICES='' python prepare.py
CUDA_VISIBLE_DEVICES='' python -m pytest -q test_metrics.py test_native_spec.py
```

`reproduce_inputs.py` uses the committed MOSES object URLs and archive-member
checksums, with a separate reproduction transfer ledger. It checks existing
metadata without redownloading it. Locally generated fingerprints are packed;
float feature arrays are unpacked only for the numerical call.
The supplied paper rebuild uses the original saved observations; a fresh
numerical study needs its own manifests, audits and reviewed renderings.

Choose fresh output directories and free CPU allocations. The measured launcher
uses 16 CPU slots on lambda2 and requires at least 120 GiB available headroom
before a large case; it refuses overlapping pinned work. The measured public
call's peak RSS can be smaller than this conservative preflight headroom.
Preparation uses eight single-core workers and one distinct coordinator CPU.
All allocations, source/input identities and timeouts remain explicit. Preserve
the supplied completed run; changed sources/settings require a new manifest.

`report.py` replays every saved count, uniform query selection, mean, threshold
fraction and sampling interval before generating tables. `validate.py` adds
independent RDKit/NumPy checks, and `factor_audit.py` reconstructs sampled factor
rows from saved landmarks to quantify returned-cut fidelity. The latter is a
post-hoc diagnostic at fixed rank, not an optimality guarantee or rank selector.
