# Dataset
#### LICENSE: The ODAC25 dataset is provided under a CC-BY-4.0 license

ODAC25 represents the largest high quality DFT dataset for Direct Air Capture containing over 15,000 Metal Organic Frameworks (MOFs), including experimental, defective, synthetic, and amine functionalized MOFs, with 4 adsorbates (carbon dioxide, water, nitrogen and oxygen). For more details on the dataset, see [arxiv](https://arxiv.org/abs/2508.03162). 

Here we provide the ODAC25 training and validation splits. Dataset files are written as ase-db [`LMDBDatabase`](https://gitlab.com/ase/ase-db-backends/-/blob/main/ase_db_backends/aselmdb.py) objects. 
There are two versions of the dataset -- *Filtered* and *Full*: the *Filtered* dataset is a subset of the *Full* dataset where we removed any MOF that was deemed problematic by Jin et al 2025 
(see https://zenodo.org/records/14802658 for more information). 
In each dataset, we have 3 partitions -- (1) *mof_plus_adsorbate* includes data from full relaxations of different adsorbates on various MOFs, 
(2) *mof* includes data from re-relaxations of empty MOFs, and (3) *gcmc* includes DFT single points of configurations derived from Grand Canonical Monte Carlo (GCMC) 
simulations.

| Splits | Size | Storage | Download |
| :-----: | :---: | :--: | :------: |
| *Train* |||| 
| Train-Filtered    | 36,920,036   | 555 GB    | [odac25_filtered_train.tar.gz](https://dl.fbaipublicfiles.com/large_objects/dac/odac25/20250918/odac25_filtered_train.tar.gz)     |
| Trian-Full    | 55,926,886  | 837 GB    | [odac25_full_train.tar.gz](https://dl.fbaipublicfiles.com/large_objects/dac/odac25/20250918/odac25_full_train.tar.gz)     |
| *Validation* |||| 
| Val-Filtered    | 791,638  | 12 GB    | [odac25_filtered_val.tar.gz](https://dl.fbaipublicfiles.com/dac/odac25/20250918/odac25_filtered_val.tar.gz)     |
| Val-Full    | 1,306,372  | 19 GB    | [odac25_full_val.tar.gz](https://dl.fbaipublicfiles.com/dac/odac25/20250918/odac25_full_val.tar.gz)     |

## How to read the data

The ODAC25 datasets can be accessed with the [fairchem](https://github.com/facebookresearch/fairchem) library. The package can be [installed](https://github.com/facebookresearch/fairchem#installation) with:
```
pip install fairchem-core
```

Once installed, a dataset can be read as follows
```python
from fairchem.core.datasets import AseDBDataset

dataset_path = "/path/to/odac25_data/train/mof_plus_adsorbate"
dataset = AseDBDataset({"src": dataset_path})

# index the dataset to get a structure
atoms = dataset.get_atoms(0)
atomic_positions = atoms.positions
atomic_numbers = atoms.get_atomic_numbers()
```

Structures are stored as ASE [Atoms objects](https://wiki.fysik.dtu.dk/ase/ase/atoms.html). Each structure contains DFT total energy (eV) and force (eV/A) labels. 
Additionally, `atoms.info` contains several other properties and metadata that may be important for model development:

`atoms.info:`

```
name: 'Gen180_H2O_single' ### Internal identifier (each trajectory has a unique identifier)
mof_name: '616_486_122_relaxed' ### Internal identifier of the bare MOF
mof_type: 'synthetic' ### Type of MOF (can be pristine, defective, functionalized, or synthetic)
sid: 2058 ### Internal identifier
fid: 91 ### Frame number within the relaxation trajectory
supercell: [1, 1, 1] ### Size of MOF supercell
nco2: 0 ### Number of CO2 adsorbates
nh2o: 1 ### Number of H2O adsorbates
nn2: 0 ### Number of N2 adsorbates
no2: 0 ### Number of O2 adsorbates
nads: 1 ### Total number of adsorbates
energy: -1250.03997882 ### DFT Energy after k-point correction
energy_old: -1250.04250372 ### Old DFT energy before k-point correction
energy_mof: -1236.13153343 ### Energy of the lowest energy bare MOF (with re-relaxation corrections)
energy_mof_old: -1236.13153343 ### Energy of the bare MOF (without re-relaxations)
energy_ads_corrected: 0.32542570999999043 ### Interaction energy (after correction with bare MOF re-relaxations, as described in Sec 2 of the paper)
energy_ads: 0.32542570999999043 ### Original interaction energy before rerelaxation correction
```
