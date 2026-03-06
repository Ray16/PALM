"""Utility functions for feature computation."""

import re
import math
import numpy as np

from .elemental_data import ELEM_PROPS, PROP_NAMES


def parse_formula(formula):
    """Parse formula like 'Ta8O20' -> {'Ta': 8, 'O': 20}."""
    pairs = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    comp = {}
    for elem, count in pairs:
        if elem == "":
            continue
        comp[elem] = comp.get(elem, 0) + (int(count) if count else 1)
    return comp


def magpie_stats(values):
    """Compute Magpie-style statistics: mean, std, min, max, range."""
    arr = np.array(values, dtype=float) if values else np.array([0.0])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
    }


def composition_weighted_mean(comp, prop_table, prop_index=None):
    """Compute composition-weighted mean of a property.

    Args:
        comp: dict of {element: count}
        prop_table: dict of {element: value} or {element: [values...]}
        prop_index: if prop_table values are lists, index into them
    """
    total = sum(comp.values())
    if total == 0:
        return 0.0
    weighted_sum = 0.0
    for elem, cnt in comp.items():
        if elem not in prop_table:
            continue
        val = prop_table[elem]
        if prop_index is not None:
            val = val[prop_index]
        weighted_sum += cnt * val
    return weighted_sum / total


def p_norm(vec, p):
    """Compute the L-p norm of a vector."""
    if not vec:
        return 0.0
    return sum(abs(x) ** p for x in vec) ** (1.0 / p)
