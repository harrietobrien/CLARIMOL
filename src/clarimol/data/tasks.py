"""
Import individual tasks from their own modules with
TASK_REGISTRY dict and the get_task() convenience function.
"""

from __future__ import annotations
from clarimol.data.parsing import Parsing
from clarimol.data.functional_group import FunctionalGroup
from clarimol.data.ring_counting import RingCounting
from clarimol.data.chain_length import ChainLength
from clarimol.data.canonicalization import Canonicalization
from clarimol.data.fragment_assembly import FragmentAssembly

# Maps task names to their implementing classes
TASK_REGISTRY: dict[str, type[Parsing]] = {
    "functional_group": FunctionalGroup,
    "ring_counting": RingCounting,
    "chain_length": ChainLength,
    "canonicalization": Canonicalization,
    "fragment_assembly": FragmentAssembly,
}


def get_task(name: str) -> Parsing:
    """Instantiate a parsing task by name"""
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()