from __future__ import annotations
import random
from rdkit import Chem
from clarimol.data.parsing import Parsing
from clarimol.data.sample import Sample


class AtomDegree(Parsing):
    """Atom Degree Sequence: produce the sorted heavy-atom connectivity sequence."""
    def __init__(self):
        super().__init__()
        self.name = "atom_degree"

    def generate(self, smiles: str, mol: Chem.Mol, rng: random.Random) -> list[Sample]:
        degrees = sorted(atom.GetDegree() for atom in mol.GetAtoms())
        if not degrees:
            return []

        answer = ",".join(str(d) for d in degrees)

        return [
            Sample(
                smiles=smiles,
                task=self.name,
                question=(
                    f"What is the sorted heavy-atom degree sequence of this molecule? "
                    f"List the number of heavy-atom neighbors for each atom, sorted in "
                    f"ascending order, separated by commas. Example: 1,1,2,2,3"
                ),
                answer=answer,
                difficulty=self.difficulty_key(mol, smiles),
                metadata={"n_atoms": mol.GetNumHeavyAtoms(), "max_degree": max(degrees)},
            )
        ]

    def difficulty_key(self, mol: Chem.Mol, smiles: str) -> float:
        return float(mol.GetNumHeavyAtoms())
