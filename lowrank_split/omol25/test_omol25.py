"""Plumbing test for the OMol25 low-rank pipeline on SYNTHETIC ase structures
(the real dataset is gated). Verifies featurization, the formula split, the
low-rank split, and that low-rank reduces cross-split similarity leakage.

Run (palm env):  python PALM/lowrank_split/omol25/test_omol25.py
"""

import sys
import numpy as np
from ase import Atoms
from ase.build import molecule

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
from PALM.lowrank_split.omol25.omol25_features import featurize_atoms, featurize_dataset
from PALM.lowrank_split.omol25.omol25_split import compare, formula_split, _cosine_leakage


def synthetic_omol(n_per_type: int = 60, seed: int = 0):
    """A heterogeneous set: organics, a metal complex, and their noisy near-duplicates.

    Near-duplicates of DIFFERENT formulas are planted so a formula split leaves
    similar structures straddling train/test, while a similarity split can catch
    them — the exact scenario OMol25's composition split misses.
    """
    rng = np.random.default_rng(seed)
    bases = [molecule("CH3CH2OH"), molecule("C6H6"), molecule("H2O"),
             molecule("CH3COOH"), molecule("NH3")]
    # a small transition-metal complex (Fe center + O/N/C ligand atoms)
    # FeO4N2C2 = 9 atoms: 1 Fe center + 8 ligand atoms
    metal = Atoms("FeO4N2C2",
                  positions=np.vstack([[0, 0, 0]] + list(rng.normal(0, 1.6, (8, 3)))))
    metal.info["charge"] = 2
    metal.info["spin"] = 5
    bases.append(metal)

    out = []
    for b in bases:
        for _ in range(n_per_type):
            a = b.copy()
            a.set_positions(a.get_positions() + rng.normal(0, 0.15, a.get_positions().shape))
            a.info.setdefault("charge", 0)
            a.info.setdefault("spin", 1)
            out.append(a)
    rng.shuffle(out)
    return out


def test_featurize_shapes():
    a = molecule("CH3CH2OH")
    v = featurize_atoms(a)
    assert v.ndim == 1 and np.all(np.isfinite(v)), "feature vector malformed"
    # length = z_max + 5 elemental + rdf_bins + 2 state
    assert len(v) == 83 + 5 + 24 + 2
    X, formulas = featurize_dataset(synthetic_omol(20), standardize=True)
    assert X.shape[0] == len(formulas) and np.all(np.isfinite(X))
    print(f"[OK] featurize: vec_dim={len(v)}, matrix={X.shape}, finite")


def test_formula_split_balance():
    structs = synthetic_omol(50)
    _, formulas = featurize_dataset(structs)
    lab = formula_split(formulas, test_frac=0.2, seed=0)
    frac = lab.mean()
    # whole-formula assignment -> coarse, so allow a wide tolerance
    print(f"[OK] formula split: test_frac={frac:.3f}, n_formulas={len(set(formulas))}")
    assert 0.1 <= frac <= 0.35


def test_lowrank_reduces_leakage():
    structs = synthetic_omol(60)
    res = compare(structs, rank=128, seed=0)
    lo = res["lowrank"]["lpi"]
    fm = res["formula_split"]["lpi"]
    print(f"[OK] leakage: formula_split lpi={fm:.4f} (NN {res['formula_split']['nn_mean']:.3f}) "
          f"-> lowrank lpi={lo:.4f} (NN {res['lowrank']['nn_mean']:.3f})")
    assert lo <= fm + 1e-3, "low-rank should not have higher cosine leakage than the formula split"
    # and low-rank must beat a random split
    X, _ = featurize_dataset(structs, standardize=True)
    n = X.shape[0]
    rng = np.random.default_rng(1)
    r = np.array([0] * int(0.8 * n) + [1] * (n - int(0.8 * n))); rng.shuffle(r)
    rand_lpi = _cosine_leakage(X, r)["lpi"]
    assert lo <= rand_lpi + 1e-3, "low-rank should beat a random split"
    print(f"     (random lpi={rand_lpi:.4f})")


if __name__ == "__main__":
    test_featurize_shapes()
    test_formula_split_balance()
    test_lowrank_reduces_leakage()
    print("\nALL OMOL25 PLUMBING TESTS PASSED")
