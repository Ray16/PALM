# PALM
Physics-Aware Leakage Minimizer

## Usage

### Dataset Splitting

`datasail_split.py` performs 2D dataset splitting using [DataSAIL](https://github.com/kalininalab/DataSAIL), treating adsorbate as the e-entity and adsorbent as the f-entity (analogous to ligand/protein in PDBBind).

**Basic usage (single embedding pair, fast):**

```bash
python datasail_split.py --e-embedding embeddings/adsorbate/physchem_features.csv --f-embedding embeddings/adsorbent/property_features.csv --f-clusters 30 --techniques C2
```

For better accuracy, use more clusters (e.g., `--f-clusters 100`). Fewer clusters run faster but produce coarser splits.

**All embeddings concatenated (default):**

```bash
python datasail_split.py
```

**Multiple embeddings per entity:**

```bash
python datasail_split.py --e-embedding embeddings/adsorbate/physchem_features.csv embeddings/adsorbate/composition_features.csv --f-embedding embeddings/adsorbent/property_features.csv embeddings/adsorbent/stoichiometry_features.csv
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

Output CSVs are saved to `output/` with columns `system_id` and `split`, tagged by embedding names (e.g., `datasail_split_C2__e_physchem__f_property.csv`).

## Dataset
The dataset is downloaded from [is2res_total_train_val_test_lmdbs](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz)

## Generated Features

Run `python gen_features.py` to generate all feature CSVs. Entries with empty `ads_symbols` are excluded, so all CSVs share the same rows (43,189 entries) keyed by `system_id`.

### Adsorbate Features (`embeddings/adsorbate/`)

| File | Columns | Description |
|------|---------|-------------|
| `physchem_features.csv` | 15 | Catalysis-relevant physicochemical properties (hardcoded per adsorbate): `mol_weight`, `num_atoms`, `num_heavy_atoms`, `num_H`, `num_OH_bonds`, `num_OO_bonds`, `is_radical`, `unpaired_electrons`, `total_valence_electrons`, `dipole_moment`, `proton_affinity`, `gas_phase_BDE`, `electron_affinity_eV`, `ionization_energy_eV`, `polarizability` |
| `rdkit_descriptors_features.csv` | 9 | RDKit molecular descriptors corrected for small radical species: `MolWt`, `NumHBondDonors`, `NumHBondAcceptors`, `NumLonePairs`, `TPSA`, `MolLogP`, `NumHeavyAtoms`, `NumValenceElectrons`, `NumRadicalElectrons` |
| `composition_features.csv` | 10 | Element counts (`count_C`, `count_H`, `count_N`, `count_O`) + composition-weighted elemental property means (`wtd_mean_atomic_number`, `wtd_mean_atomic_mass`, `wtd_mean_electronegativity`, `wtd_mean_covalent_radius`, `wtd_mean_electron_affinity`, `wtd_mean_ionization_energy`) |

### Adsorbent Features (`embeddings/adsorbent/`)

| File | Columns | Description |
|------|---------|-------------|
| `composition_features.csv` | 54 | Per-element atom counts across 54 unique elements |
| `fraction_features.csv` | 54 | Normalized composition fractions per element |
| `property_features.csv` | 30 | Magpie-style statistics (mean, std, min, max, range) of 6 elemental properties (atomic_number, atomic_mass, electronegativity, covalent_radius, electron_affinity, ionization_energy) weighted by composition |
| `stoichiometry_features.csv` | 8 | `num_elements`, `total_atoms`, `composition_entropy` (Shannon), and p-norms of composition vector (L2, L3, L5, L7, L10) |

## Contributors
- [Ray Zhu](https://github.com/Ray16)
- [Rodrigo Ferreira](https://github.com/rpf00)

## License
The dataset is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode)

## Contribution
Contributions are welcome! Please open an issue or submit a pull request.
