from __future__ import annotations
import random
from rdkit import Chem
from rdkit.Chem import BRICS
from clarimol.data.parsing import Parsing
from clarimol.data.sample import Sample


class FragmentAssembly(Parsing):
    """Join two molecular fragments (from BRICS decomposition) back together."""

    def __init__(self):
        super().__init__()
        self.name = "fragment_assembly"
        self.question = ("Connect the following two SMILES fragments into a unified structure "
                         "at their reactive sites. Answer with the resulting SMILES only.")

    def generate(self, smiles: str, mol: Chem.Mol, rng: random.Random) -> list[Sample]:
        frags = self._brics_fragments(mol)
        if frags is None or len(frags) < 2:
            frags = self._split_nonring_bond(mol, rng)
        if frags is None or len(frags) < 2:
            return list()
        frag1, frag2 = frags[0], frags[1]
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return [
            Sample(
                smiles=f"{frag1} . {frag2}",
                task=self.name,
                question=self.question,
                answer=canonical,
                difficulty=self.difficulty_key(mol, smiles),
                metadata={"frag1": frag1, "frag2": frag2},
            )
        ]

    @staticmethod
    def _brics_fragments(mol: Chem.Mol) -> list[str] | None:
        """BRICS decomposition into exactly two fragments with dummy atoms"""
        try:
            frags = list(BRICS.BRICSDecompose(mol, minFragmentSize=2))
            if len(frags) == 2:
                return frags
        except Exception:
            pass
        return None

    @staticmethod
    def _split_nonring_bond(mol: Chem.Mol, rng: random.Random) -> list[str] | None:
        """Split molecule on random non-ring single bond as fallback"""
        ring_bonds: set[int] = set()
        for ring in mol.GetRingInfo().BondRings():
            ring_bonds.update(ring)
        candidates = [bond.GetIdx() for bond in mol.GetBonds()
                      if bond.GetIdx() not in ring_bonds and
                      bond.GetBondTypeAsDouble() == 1.0]
        if not candidates:
            return None
        bond_idx = rng.choice(candidates)
        try:
            frags = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True)
            frag_smiles = Chem.MolToSmiles(frags).split(".")
            if len(frag_smiles) >= 2:
                return frag_smiles
        except Exception:
            pass
        return None

    def difficulty_key(self, mol: Chem.Mol, smiles: str) -> float:
        """Longer SMILES → harder to reassemble"""
        return float(len(smiles))