# PALM
Physics-Aware Leakage Minimizer

## Usage

### 1. Generate Features

`0_gen_features.py` generates all feature CSVs for adsorbates and adsorbents. Entries with empty `ads_symbols` are excluded, so all CSVs share the same rows (43,189 entries) keyed by `system_id`.

```bash
python 0_gen_features.py
```

Features are saved to `features/oc22/adsorbate/` and `features/oc22/adsorbent/`.

### 2. Dataset Splitting

`1_datasail_split.py` performs 2D dataset splitting using [DataSAIL](https://github.com/kalininalab/DataSAIL), treating adsorbate as the e-entity and adsorbent as the f-entity (analogous to ligand/protein in PDBBind).

**Basic usage (single embedding pair, fast):**

```bash
python 1_datasail_split.py --e-embedding features/oc22/adsorbate/physchem_features.csv --f-embedding features/oc22/adsorbent/property_features.csv --f-clusters 30 --techniques C2
```

For better accuracy, use more clusters (e.g., `--f-clusters 100`). Fewer clusters run faster but produce coarser splits.

**All embeddings concatenated (default):**

```bash
python 1_datasail_split.py
```

**Multiple embeddings per entity:**

```bash
python 1_datasail_split.py \
  --e-embedding features/oc22/adsorbate/physchem_features.csv features/oc22/adsorbate/composition_features.csv \
  --f-embedding features/oc22/adsorbent/property_features.csv features/oc22/adsorbent/stoichiometry_features.csv
```

**Full options:**

```
--e-embedding FILE [FILE ...]   Adsorbate embedding CSV(s)
--f-embedding FILE [FILE ...]   Adsorbent embedding CSV(s)
--f-clusters N                  Number of adsorbent clusters (default: 200)
--max-sec N                     Solver time limit in seconds (default: 300)
--techniques T [T ...]          Splitting techniques (default: R I1e I1f I2 C1e C1f C2)
--splits N [N ...]              Split ratios (default: 8 2)
--names NAME [NAME ...]         Split names (default: train test)
```

**Available techniques:**

| Technique | Description |
|-----------|-------------|
| `R` | Random split |
| `I1e` | Identity-cold on adsorbate |
| `I1f` | Identity-cold on adsorbent |
| `I2` | Identity-cold on both |
| `C1e` | Cluster-cold on adsorbate |
| `C1f` | Cluster-cold on adsorbent |
| `C2` | Cluster-cold on both (double-cold) |

Output CSVs are saved to `output/split_result/` with columns `system_id` and `split`, tagged by embedding names (e.g., `datasail_split_C2__e_physchem__f_property.csv`).

### 3. Summary Dashboard

`2_summary.py` generates overview figures for comparing all feature pair combinations, showing raw coverage and overlap metrics side by side. Use this to narrow down which (method, feature pair) to use.

```bash
# Compare all feature pairs across all methods
python 2_summary.py

# Compare feature pairs for a specific method
python 2_summary.py --method C1e

# Custom output directory
python 2_summary.py -o output/summary
```

**Per-method figures** (`output/summary/summary_{method}.png`):
- **(a) Coverage** — stacked train/test bar chart with coverage % annotations
- **(b) Overlap** — grouped bars for pair, adsorbate, and adsorbent overlap counts

**Cross-method figure** (`output/summary/summary_cross_method.png`):
- **(a) Coverage heatmap** — feature pairs (rows) × methods (columns), colored by coverage %
- **(b) Overlap heatmap** — same layout, showing P(air)/A(dsorbate)/B(adsorbent) overlap breakdown per cell

### 4. Deep Analysis

`3_analysis.py` performs detailed analysis on selected splits. Use this after narrowing down candidates with the summary dashboard.

```bash
# Analyze a single split
python 3_analysis.py output/split_result/datasail_split_C2__e_rdkit_descriptors__f_stoichiometry.csv

# Compare multiple selected splits
python 3_analysis.py output/split_result/datasail_split_C2__e_rdkit_descriptors__f_stoichiometry.csv \
                      output/split_result/datasail_split_C1e__e_rdkit_descriptors__f_bonding.csv

# Custom embeddings for UMAP and NN distance computation
python 3_analysis.py --e-embedding features/oc22/adsorbate/physchem_features.csv \
                     --f-embedding features/oc22/adsorbent/property_features.csv \
                     output/split_result/datasail_split_C2__e_rdkit_descriptors__f_stoichiometry.csv
```

**Per-split figures** (`output/analysis/analysis_{label}.png`):
- **(a) Adsorbent UMAP** — 2D projection colored by train/test/not-selected
- **(b) Adsorbate UMAP** — 2D projection with labeled points
- **(c) Adsorbate distribution** — per-adsorbate train vs test system counts
- **(d) Adsorbent NN distance** — histogram of test→nearest-train distances in feature space
- **(e) Adsorbate NN distance** — histogram of test→nearest-train distances
- **(f) Summary statistics** — table with all key metrics

**Comparison figure** (`output/analysis/analysis_comparison.png`, when multiple splits given):
- **(a) Entity overlap** — grouped bars comparing pair/adsorbate/adsorbent overlap
- **(b) Split size distribution** — stacked bars with coverage %
- **(c) Mean NN distances** — grouped bars for adsorbate and adsorbent separation
- **(d) NN distance distributions** — box plots for direct comparison

## Dataset
The dataset is downloaded from [is2res_total_train_val_test_lmdbs](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz)

## Generated Features

### Adsorbate Features (`features/oc22/adsorbate/`)

| File | Features | Description |
|------|----------|-------------|
| `physchem_features.csv` | 15 | Catalysis-relevant physicochemical properties (hardcoded per adsorbate): mol weight, atom counts, radical info, dipole moment, proton affinity, gas-phase BDE, electron affinity, ionization energy, polarizability |
| `rdkit_descriptors_features.csv` | 9 | RDKit molecular descriptors corrected for small radical species: MolWt, H-bond donors/acceptors, lone pairs, TPSA, MolLogP, valence/radical electrons |
| `composition_features.csv` | 10 | Element counts + composition-weighted elemental property means |
| `adsorption_features.csv` | 12 | Surface interaction descriptors (hardcoded from NIST/literature): HOMO/LUMO energies, chemical hardness/softness, electrophilicity index, formation enthalpy, entropy, characteristic vibrational frequency, bond order, lone pair/pi bond flags, surface binding mode |

### Adsorbent Features (`features/oc22/adsorbent/`)

| File | Features | Description |
|------|----------|-------------|
| `property_features.csv` | 30 | Magpie-style statistics (mean, std, min, max, range) of 6 elemental properties (atomic number, mass, electronegativity, covalent radius, electron affinity, ionization energy) weighted by composition |
| `stoichiometry_features.csv` | 8 | Num elements, total atoms, Shannon entropy, and p-norms of composition vector (L2, L3, L5, L7, L10) |
| `electronic_features.csv` | 12 | d-electron statistics (mean, std, range, max, min), valence electron stats, oxidation state stats, work function stats — d-band center proxies for catalytic activity |
| `bonding_features.csv` | 10 | Metal-oxygen bonding character: electronegativity difference, ionic radius, radius ratio, Pauling bond ionicity, Sanderson oxide basicity, metal-to-O ratio, polarizability, perovskite tolerance factor |
| `thermodynamic_features.csv` | 8 | Stability/reactivity indicators: metal melting point stats, oxide formation enthalpy, reducibility index, mass-weighted mixing entropy |
| `catalytic_features.csv` | 8 | Catalysis-specific: transition metal/noble metal/rare earth flags, TM fraction, d-band filling, metal diversity, electronegativity spread, weighted electron affinity |

## Project Structure

```
PALM/
├── 0_gen_features.py          # Feature generation
├── 1_datasail_split.py        # DataSAIL splitting
├── 2_summary.py               # Summary dashboard (coverage & overlap comparison)
├── 3_analysis.py              # Deep analysis (UMAP, NN distances, distributions)
├── data/oc22/                 # Raw dataset
├── features/oc22/             # Generated features
│   ├── adsorbate/             #   Adsorbate feature CSVs
│   └── adsorbent/             #   Adsorbent feature CSVs
└── output/
    ├── split_result/          # Split CSV outputs
    ├── summary/               # Summary dashboard figures
    └── analysis/              # Deep analysis figures
```

## Contributors
- [Ray Zhu](https://github.com/Ray16)
- [Rodrigo Ferreira](https://github.com/rpf00)

## License
The dataset is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode)

## Contribution
Contributions are welcome! Please open an issue or submit a pull request.
