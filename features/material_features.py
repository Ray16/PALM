"""Formula-based material featurization (6 feature sets).

Generalized from OC22 adsorbent featurization — computes stats over ALL
elements in a formula, not just metals.
"""

import math
import numpy as np
import pandas as pd

from .elemental_data import (
    ELEM_PROPS, PROP_NAMES,
    ELEM_D_ELECTRONS, ELEM_VALENCE_ELECTRONS, ELEM_OXIDATION_STATE,
    ELEM_WORK_FUNCTION, ELEM_IONIC_RADIUS, ELEM_POLARIZABILITY,
    ELEM_MELTING_POINT, ELEM_OXIDE_FORMATION_ENTHALPY,
    TRANSITION_METALS, NOBLE_METALS, RARE_EARTHS,
)
from .utils import parse_formula, magpie_stats, p_norm

STAT_NAMES = ["mean", "std", "min", "max", "range"]


def magpie_properties(formula):
    """Magpie-style elemental property statistics over all elements."""
    comp = parse_formula(formula)
    prop_arrays = {p: [] for p in PROP_NAMES}
    for elem, cnt in comp.items():
        if elem not in ELEM_PROPS:
            continue
        for pidx, pname in enumerate(PROP_NAMES):
            prop_arrays[pname].extend([ELEM_PROPS[elem][pidx]] * cnt)

    feats = {}
    for pname in PROP_NAMES:
        stats = magpie_stats(prop_arrays[pname])
        for stat_name, val in stats.items():
            feats[f"{pname}_{stat_name}"] = val
    return feats


def stoichiometry(formula):
    """Stoichiometric descriptors: element count, total atoms, entropy, L-p norms."""
    comp = parse_formula(formula)
    counts = list(comp.values())
    total = sum(counts)
    num_elements = len(counts)
    fracs = [c / total for c in counts] if total > 0 else []
    entropy = -sum(f * math.log(f) for f in fracs if f > 0) if fracs else 0.0

    feats = {
        "num_elements": num_elements,
        "total_atoms": total,
        "composition_entropy": entropy,
    }
    for p in [2, 3, 5, 7, 10]:
        feats[f"L{p}_norm"] = p_norm(fracs, p)
    return feats


def electronic(formula):
    """Electronic features: d-band and valence electron statistics over all elements."""
    comp = parse_formula(formula)
    if not comp:
        return {k: 0.0 for k in [
            "d_electron_mean", "d_electron_std", "d_electron_range",
            "valence_electron_mean", "valence_electron_std",
            "oxidation_state_mean", "oxidation_state_std",
            "work_function_mean", "work_function_std",
            "d_electron_max", "d_electron_min", "work_function_range",
        ]}

    d_elec, valence, oxstate, wf = [], [], [], []
    for elem, cnt in comp.items():
        d_elec.extend([ELEM_D_ELECTRONS.get(elem, 0)] * cnt)
        valence.extend([ELEM_VALENCE_ELECTRONS.get(elem, 0)] * cnt)
        oxstate.extend([ELEM_OXIDATION_STATE.get(elem, 0)] * cnt)
        wf.extend([ELEM_WORK_FUNCTION.get(elem, 0.0)] * cnt)

    d_arr = np.array(d_elec, dtype=float)
    v_arr = np.array(valence, dtype=float)
    o_arr = np.array(oxstate, dtype=float)
    w_arr = np.array(wf, dtype=float)

    return {
        "d_electron_mean": float(np.mean(d_arr)),
        "d_electron_std": float(np.std(d_arr)),
        "d_electron_range": float(np.max(d_arr) - np.min(d_arr)),
        "valence_electron_mean": float(np.mean(v_arr)),
        "valence_electron_std": float(np.std(v_arr)),
        "oxidation_state_mean": float(np.mean(o_arr)),
        "oxidation_state_std": float(np.std(o_arr)),
        "work_function_mean": float(np.mean(w_arr)),
        "work_function_std": float(np.std(w_arr)),
        "d_electron_max": float(np.max(d_arr)),
        "d_electron_min": float(np.min(d_arr)),
        "work_function_range": float(np.max(w_arr) - np.min(w_arr)),
    }


def bonding(formula):
    """Bonding features: electronegativity differences, ionic radii, polarizability."""
    comp = parse_formula(formula)
    if not comp:
        return {k: 0.0 for k in [
            "avg_electronegativity_diff", "avg_ionic_radius",
            "radius_ratio", "avg_bond_ionicity", "geometric_mean_en",
            "avg_polarizability", "tolerance_factor",
            "en_diff_std", "ionic_radius_std",
        ]}

    # Compute pairwise electronegativity differences relative to mean
    en_values = []
    radii = []
    polariz = []
    ionicities = []

    elems = list(comp.keys())
    all_en = []
    for elem, cnt in comp.items():
        en = ELEM_PROPS.get(elem, [0] * 6)[2]
        if en > 0:
            all_en.extend([en] * cnt)

    mean_en = np.mean(all_en) if all_en else 0.0

    for elem, cnt in comp.items():
        en_m = ELEM_PROPS.get(elem, [0] * 6)[2]
        ir = ELEM_IONIC_RADIUS.get(elem, 0)
        pol = ELEM_POLARIZABILITY.get(elem, 0.0)

        delta_en = abs(mean_en - en_m)
        ionicity = 1.0 - math.exp(-0.25 * delta_en ** 2)

        en_values.extend([delta_en] * cnt)
        radii.extend([ir] * cnt)
        polariz.extend([pol] * cnt)
        ionicities.extend([ionicity] * cnt)

    en_arr = np.array(en_values, dtype=float)
    r_arr = np.array(radii, dtype=float)
    p_arr = np.array(polariz, dtype=float)
    ion_arr = np.array(ionicities, dtype=float)

    # Geometric mean of electronegativities
    geo_en = float(np.exp(np.mean(np.log(np.array(all_en))))) if all_en else 0.0

    # Tolerance factor for perovskite-like structures
    tolerance = 0.0
    if len(elems) >= 2:
        sorted_by_radius = sorted(elems, key=lambda e: ELEM_IONIC_RADIUS.get(e, 0), reverse=True)
        r_A = ELEM_IONIC_RADIUS.get(sorted_by_radius[0], 0)
        r_B = ELEM_IONIC_RADIUS.get(sorted_by_radius[1], 0)
        r_O = 140  # O^2- ionic radius
        if r_A > 0 and r_B > 0:
            tolerance = (r_A + r_O) / (math.sqrt(2) * (r_B + r_O))

    r_max = float(np.max(r_arr)) if len(r_arr) > 0 else 0.0
    r_min = float(np.min(r_arr)) if len(r_arr) > 0 else 0.0
    radius_ratio = r_max / r_min if r_min > 0 else 0.0

    return {
        "avg_electronegativity_diff": float(np.mean(en_arr)),
        "avg_ionic_radius": float(np.mean(r_arr)),
        "radius_ratio": radius_ratio,
        "avg_bond_ionicity": float(np.mean(ion_arr)),
        "geometric_mean_en": geo_en,
        "avg_polarizability": float(np.mean(p_arr)),
        "tolerance_factor": tolerance,
        "en_diff_std": float(np.std(en_arr)),
        "ionic_radius_std": float(np.std(r_arr)),
    }


def thermodynamic(formula):
    """Thermodynamic features: melting points, formation enthalpies, reducibility."""
    comp = parse_formula(formula)
    if not comp:
        return {k: 0.0 for k in [
            "avg_melting_point", "melting_point_std", "melting_point_range",
            "avg_oxide_formation_enthalpy", "reducibility_index",
            "entropy_mixing_mass_weighted",
            "min_oxide_formation_enthalpy", "max_oxide_formation_enthalpy",
        ]}

    mp_arr = []
    form_enth = []
    masses = []
    for elem, cnt in comp.items():
        mp_arr.extend([ELEM_MELTING_POINT.get(elem, 0)] * cnt)
        form_enth.extend([ELEM_OXIDE_FORMATION_ENTHALPY.get(elem, 0)] * cnt)
        masses.extend([ELEM_PROPS.get(elem, [0] * 6)[1]] * cnt)

    mp = np.array(mp_arr, dtype=float)
    fe = np.array(form_enth, dtype=float)
    m = np.array(masses, dtype=float)

    mass_fracs = m / m.sum() if m.sum() > 0 else np.zeros_like(m)
    entropy_mass = -float(np.sum(mass_fracs * np.log(mass_fracs + 1e-12)))
    reducibility = -float(np.mean(fe))

    return {
        "avg_melting_point": float(np.mean(mp)),
        "melting_point_std": float(np.std(mp)),
        "melting_point_range": float(np.max(mp) - np.min(mp)),
        "avg_oxide_formation_enthalpy": float(np.mean(fe)),
        "reducibility_index": reducibility,
        "entropy_mixing_mass_weighted": entropy_mass,
        "min_oxide_formation_enthalpy": float(np.min(fe)),
        "max_oxide_formation_enthalpy": float(np.max(fe)),
    }


def classification(formula):
    """Classification features: element type flags and fractions."""
    comp = parse_formula(formula)
    if not comp:
        return {k: 0.0 for k in [
            "has_transition_metal", "has_noble_metal", "has_rare_earth",
            "fraction_transition_metal", "d_band_filling", "element_diversity",
            "electronegativity_spread", "weighted_electron_affinity",
        ]}

    total_atoms = sum(comp.values())

    has_tm = int(any(e in TRANSITION_METALS for e in comp))
    has_noble = int(any(e in NOBLE_METALS for e in comp))
    has_re = int(any(e in RARE_EARTHS for e in comp))

    tm_count = sum(cnt for e, cnt in comp.items() if e in TRANSITION_METALS)
    frac_tm = tm_count / total_atoms if total_atoms > 0 else 0.0

    d_block_d = []
    for elem, cnt in comp.items():
        if elem in TRANSITION_METALS:
            d_elec = ELEM_D_ELECTRONS.get(elem, 0)
            d_block_d.extend([d_elec / 10.0] * cnt)
    d_band_filling = float(np.mean(d_block_d)) if d_block_d else 0.0

    en_all = [ELEM_PROPS.get(e, [0] * 6)[2] for e in comp if e in ELEM_PROPS]
    en_spread = max(en_all) - min(en_all) if len(en_all) > 1 else 0.0

    ea_arr = []
    for elem, cnt in comp.items():
        ea = ELEM_PROPS.get(elem, [0] * 6)[4]
        ea_arr.extend([ea] * cnt)
    wtd_ea = float(np.mean(ea_arr)) if ea_arr else 0.0

    return {
        "has_transition_metal": has_tm,
        "has_noble_metal": has_noble,
        "has_rare_earth": has_re,
        "fraction_transition_metal": frac_tm,
        "d_band_filling": d_band_filling,
        "element_diversity": len(comp),
        "electronegativity_spread": en_spread,
        "weighted_electron_affinity": wtd_ea,
    }


# Registry of all material feature sets
MATERIAL_FEATURE_SETS = {
    "magpie_properties": magpie_properties,
    "stoichiometry": stoichiometry,
    "electronic": electronic,
    "bonding": bonding,
    "thermodynamic": thermodynamic,
    "classification": classification,
}


def compute_material_features(entities, feature_sets=None):
    """Compute material features for a dict of {entity_id: formula}.

    Args:
        entities: dict mapping entity ID to formula string
        feature_sets: list of feature set names, or None for all

    Returns:
        DataFrame with entity_id as index, feature columns
    """
    if feature_sets is None:
        feature_sets = list(MATERIAL_FEATURE_SETS.keys())

    rows = {}
    for entity_id, formula in entities.items():
        feats = {}
        for fs_name in feature_sets:
            fn = MATERIAL_FEATURE_SETS[fs_name]
            feats.update(fn(formula))
        rows[entity_id] = feats

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "entity_id"
    return df
