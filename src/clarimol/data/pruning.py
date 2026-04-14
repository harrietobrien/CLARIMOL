"""
Data Pruning and Curriculum Ordering
"""

from __future__ import annotations
from dataclasses import dataclass
from clarimol.data.sample import Sample
import random


@dataclass
class PruningConfig:
    """Controls degree of pruning and reorder"""
    keep_n: int = 50_000
    # "middle", "top", "bottom", "random"
    subsample: str = "middle"
    # fraction trimmed from each tail before middle selection
    trim_fraction: float = 0.15
    # if True, final order is easy → hard
    sort_curriculum: bool = True


def prune_and_sort(
    samples: list[Sample],
    config: PruningConfig | None = None,
) -> list[Sample]:
    """
    Prune a list of Samples by difficulty;
    optionally reorder for curriculum learning.

    :return: a new list (to avoid mutating input list)
    """
    if config is None:
        config = PruningConfig()
    if not samples:
        return list()
    # Sort by difficulty ascending (easy first)
    ranked = sorted(samples, key=lambda s: s.difficulty)
    n = len(ranked)
    if config.subsample == "random":
        ranked_copy = list(ranked)
        random.shuffle(ranked_copy)
        selected = ranked_copy[: config.keep_n]
    elif config.subsample == "top":
        # Hardest - 'top'
        selected = ranked[-config.keep_n :]
    elif config.subsample == "bottom":
        # Easiest - 'bottom'
        selected = ranked[: config.keep_n]
    else:
        # "middle" — trim tails, then take center
        trim = int(n * config.trim_fraction)
        body = ranked[trim : n - trim] if trim > 0 else ranked
        if len(body) > config.keep_n:
            start = (len(body) - config.keep_n) // 2
            selected = body[start : start + config.keep_n]
        else:
            selected = list(body)
    # Curriculum ordering: easy → hard
    if config.sort_curriculum:
        selected.sort(key=lambda s: s.difficulty)
    return selected
