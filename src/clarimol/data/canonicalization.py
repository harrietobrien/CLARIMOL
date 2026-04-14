from __future__ import annotations
import random
from rdkit import Chem
from clarimol.data.parsing import Parsing
from clarimol.data.sample import Sample


class Canonicalization(Parsing):
    """SMILES Canonicalization: Produce canonical form from randomized input."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "canonicalization"
        self.question = ("Give a canonicalized SMILES string that represents "
                         "the same molecule as the given molecule.")

    def generate(self, smiles: str, mol: Chem.Mol, rng: random.Random) -> list[Sample]:
        canonical = Chem.MolToSmiles(mol, canonical=True)
        randomized = self._randomize_smiles(mol, rng)
        if randomized is None or randomized == canonical:
            return list()
        return [Sample(smiles=randomized, task=self.name, question=self.question,
                answer=canonical, difficulty=self.difficulty_key(mol, smiles),
                metadata={"canonical": canonical, "randomized": randomized})]

    @staticmethod
    def _randomize_smiles(mol: Chem.Mol, rng: random.Random) -> str | None:
        """Generate random (non-canonical) SMILES by selecting random root atom"""
        n_atoms = mol.GetNumAtoms()
        if n_atoms < 2:
            return None
        root = rng.randint(0, n_atoms - 1)
        try:
            return Chem.MolToSmiles(mol, rootedAtAtom=root, canonical=False)
        except Exception:
            return None

    def difficulty_key(self, mol: Chem.Mol, smiles: str) -> float:
        """Longer SMILES are harder to canonicalize"""
        return float(len(smiles))