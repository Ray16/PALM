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

### 3. Visualization

```bash
# Visualize all splits (grouped by method)
python visualization.py

# Visualize specific method
python visualization.py --method C1e

# Visualize specific splits
python visualization.py output/split_result/datasail_split_C1f__e_physchem__f_property.csv
```

Figures are saved to `output/figure/`.

### 4. Validation

```bash
# Validate all splits
python validation.py output/split_result/datasail_split_*.csv

# Validate a single split
python validation.py output/split_result/datasail_split_C2__e_physchem__f_property.csv
```

Validation figure is saved to `output/figure/split_validation.png`.

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
├── visualization.py           # Split visualization
├── validation.py              # Split validation
├── data/oc22/                 # Raw dataset
├── features/oc22/             # Generated features
│   ├── adsorbate/             #   Adsorbate feature CSVs
│   └── adsorbent/             #   Adsorbent feature CSVs
└── output/
    ├── split_result/          # Split CSV outputs
    └── figure/                # Visualization & validation figures
```

## Contributors
- [Ray Zhu](https://github.com/Ray16)
- [Rodrigo Ferreira](https://github.com/rpf00)

## License
The dataset is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode)

## Contribution
Contributions are welcome! Please open an issue or submit a pull request.
