# PALM
Physics-Aware Leakage Minimizer

## Usage

## Dataset
The dataset is downloaded from [is2res_total_train_val_test_lmdbs](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz)

## Generated Features

Run `python exam_metadata.py` to generate all feature CSVs. Entries with empty `ads_symbols` are excluded, so all CSVs share the same rows (43,189 entries) keyed by `system_id`.

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
