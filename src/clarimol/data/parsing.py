from __future__ import annotations
import abc
import random
from rdkit import Chem
from clarimol.data.sample import Sample


class Parsing(abc.ABC):
    """
    Abstract base for SMILES parsing tasks

    Each subclass defines:
      - name: task identifier string
      - generate(): produce Sample(s) from a molecule
      - difficulty_key(): scalar for pruning / curriculum ordering
    """
    def __init__(self):
        self.name: str

    @abc.abstractmethod
    def generate(self, smiles: str, mol: Chem.Mol, rng: random.Random) -> list[Sample]:
        """Produce one or more Samples from a molecule. Return [] to skip"""
        ...

    @abc.abstractmethod
    def difficulty_key(self, mol: Chem.Mol, smiles: str) -> float:
        """Scalar difficulty score for data-pruning / curriculum ordering"""
        ...