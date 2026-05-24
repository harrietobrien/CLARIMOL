from __future__ import annotations
import random
from rdkit import Chem
from clarimol.data.parsing import Parsing
from clarimol.data.sample import Sample


class TopologicalDistance(Parsing):
    """Topological Distance Query: shortest path between two atoms."""
    def __init__(self):
        super().__init__()
        self.name = "topological_distance"

    def generate(self, smiles: str, mol: Chem.Mol, rng: random.Random) -> list[Sample]:
        n_atoms = mol.GetNumHeavyAtoms()
        if n_atoms < 3:
            return []

        # Pick two distinct random atom indices
        idx1 = rng.randint(0, n_atoms - 1)
        idx2 = rng.randint(0, n_atoms - 1)
        while idx2 == idx1:
            idx2 = rng.randint(0, n_atoms - 1)

        path = Chem.rdmolops.GetShortestPath(mol, idx1, idx2)
        distance = len(path) - 1

        atom1 = mol.GetAtomWithIdx(idx1)
        atom2 = mol.GetAtomWithIdx(idx2)

        return [
            Sample(
                smiles=smiles,
                task=self.name,
                question=(
                    f"What is the shortest path distance (in bonds) between atom {idx1} "
                    f"({atom1.GetSymbol()}) and atom {idx2} ({atom2.GetSymbol()}) "
                    f"in this molecule? Answer with an integer only."
                ),
                answer=str(distance),
                difficulty=self.difficulty_key(mol, smiles),
                metadata={
                    "idx1": idx1, "idx2": idx2,
                    "distance": distance, "n_atoms": n_atoms,
                    "atom1": atom1.GetSymbol(), "atom2": atom2.GetSymbol(),
                },
            )
        ]

    def difficulty_key(self, mol: Chem.Mol, smiles: str) -> float:
        return float(mol.GetNumHeavyAtoms())
