from __future__ import annotations
import random
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from clarimol.data.parsing import Parsing
from clarimol.data.sample import Sample


class RingCounting(Parsing):
    """Ring Counting: Count the no. of rings of a size in the molecule"""
    def __init__(self):
        super().__init__()
        self.name = "ring_counting"

    def generate(self, smiles: str, mol: Chem.Mol, rng: random.Random) -> list[Sample]:
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        if not ring_sizes:
            return list()
        target_size = rng.choice(ring_sizes)
        count = ring_sizes.count(target_size)
        return [
            Sample(
                smiles=smiles,
                task=self.name,
                question=(
                    f"How many {target_size}-membered rings are in the molecule "
                    f"represented by this SMILES? Answer with an integer only."
                ),
                answer=str(count),
                difficulty=self.difficulty_key(mol, smiles),
                metadata={"ring_size": target_size, "count": count},
            )
        ]

    def difficulty_key(self, mol: Chem.Mol, smiles: str) -> float:
        """Total number of rings — more rings = harder"""
        return float(CalcNumRings(mol))