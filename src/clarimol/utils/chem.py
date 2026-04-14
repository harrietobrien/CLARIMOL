"""Chemistry Utility Functions"""

from __future__ import annotations
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)


def is_valid_smiles(smiles: str) -> bool:
    """Check validity of a SMILES string"""
    return Chem.MolFromSmiles(smiles) is not None


def canonicalize(smiles: str) -> str | None:
    """Return canonical SMILES, or None if invalid"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)