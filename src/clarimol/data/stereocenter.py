from __future__ import annotations
import random
from rdkit import Chem
from clarimol.data.parsing import Parsing
from clarimol.data.sample import Sample


class Stereocenter(Parsing):
    """Stereocenter CIP Assignment: identify chiral centers and assign R/S labels."""
    def __init__(self):
        super().__init__()
        self.name = "stereocenter"

    def generate(self, smiles: str, mol: Chem.Mol, rng: random.Random) -> list[Sample]:
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
        if not chiral_centers:
            return []

        # Format: "C2:R, C5:S"
        labels = []
        for idx, cip in chiral_centers:
            atom = mol.GetAtomWithIdx(idx)
            symbol = atom.GetSymbol()
            labels.append(f"{symbol}{idx}:{cip}")
        answer = ", ".join(labels)

        return [
            Sample(
                smiles=smiles,
                task=self.name,
                question=(
                    f"Identify all stereocenters in the molecule and assign their "
                    f"CIP (R/S) labels. Format: Symbol#Index:Label separated by commas. "
                    f"Example: C2:R, C5:S"
                ),
                answer=answer,
                difficulty=self.difficulty_key(mol, smiles),
                metadata={"n_centers": len(chiral_centers), "centers": chiral_centers},
            )
        ]

    def difficulty_key(self, mol: Chem.Mol, smiles: str) -> float:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
        return float(len(centers))
