"""Map partition block indices to split names.

The largest block becomes the largest-ratio split (e.g. train); ties are broken
by block index for determinism. This is the ``blocks_by_size / order_by_split``
snippet that was inlined five times across the splitters.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Hashable, Sequence


def blocks_to_names(labels: Sequence[int], splits: Sequence[float],
                    names: Sequence[str]) -> Dict[int, str]:
    """``{block_index: split_name}`` by descending block size.

    NOTE: assumes the requested split ratios are distinct (e.g. [8, 2]); for
    (near-)equal ratios the size→name assignment is arbitrary among the tied
    splits, which is harmless since they are interchangeable by construction.
    """
    sizes = Counter(int(x) for x in labels)
    blocks_by_size = sorted(sizes, key=lambda b: (-sizes[b], b))
    order_by_split = sorted(range(len(splits)), key=lambda i: splits[i], reverse=True)
    return {blk: names[order_by_split[min(rank, len(order_by_split) - 1)]]
            for rank, blk in enumerate(blocks_by_size)}


def assign_split_names(ids: Sequence[Hashable], labels: Sequence[int],
                       splits: Sequence[float], names: Sequence[str]) -> Dict[Hashable, str]:
    """``{entity_id: split_name}`` from block labels aligned to ``ids``."""
    block_to_name = blocks_to_names(labels, splits, names)
    return {ids[i]: block_to_name.get(int(labels[i]), names[-1]) for i in range(len(ids))}
