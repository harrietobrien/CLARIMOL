"""
Data Pruning and Curriculum Ordering
"""

from __future__ import annotations
from dataclasses import dataclass
from clarimol.data.sample import Sample
import random


# Valid curriculum ordering strategies
CURRICULUM_ORDERS = ("easy-hard", "hard-easy", "random", "none")


@dataclass
class PruningConfig:
    """Controls degree of pruning and reorder"""
    keep_n: int = 50_000
    # "middle", "top", "bottom", "random"
    subsample: str = "middle"
    # fraction trimmed from each tail before middle selection
    trim_fraction: float = 0.15
    # Curriculum ordering: "easy-hard", "hard-easy", "random", "none"
    curriculum_order: str = "easy-hard"
    # Legacy compat: if set, overrides curriculum_order
    sort_curriculum: bool | None = None

    def __post_init__(self):
        if self.sort_curriculum is not None:
            self.curriculum_order = "easy-hard" if self.sort_curriculum else "none"
        if self.curriculum_order not in CURRICULUM_ORDERS:
            raise ValueError(
                f"Unknown curriculum_order '{self.curriculum_order}'. "
                f"Available: {CURRICULUM_ORDERS}"
            )


def prune_and_sort(
    samples: list[Sample],
    config: PruningConfig | None = None,
    seed: int = 42,
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
    rng = random.Random(seed)
    # Sort by difficulty ascending (easy first)
    ranked = sorted(samples, key=lambda s: s.difficulty)
    n = len(ranked)
    if config.subsample == "random":
        ranked_copy = list(ranked)
        rng.shuffle(ranked_copy)
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
    # Apply curriculum ordering
    if config.curriculum_order == "easy-hard":
        selected.sort(key=lambda s: s.difficulty)
    elif config.curriculum_order == "hard-easy":
        selected.sort(key=lambda s: s.difficulty, reverse=True)
    elif config.curriculum_order == "random":
        rng.shuffle(selected)
    # "none" → preserve selection order (which is already difficulty-sorted from subsample)
    return selected
