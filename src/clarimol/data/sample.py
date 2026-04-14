from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    """Sample dataclass: Training example for a SMILES parsing task."""
    smiles: str
    task: str
    question: str
    answer: str
    difficulty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)