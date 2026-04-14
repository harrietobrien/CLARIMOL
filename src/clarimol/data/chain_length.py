from __future__ import annotations
import random
import networkx as nx
from rdkit import Chem
from clarimol.data.parsing import Parsing
from clarimol.data.sample import Sample


class ChainLength(Parsing):
    """Chain Length Measurement: Longest acyclic carbon chain."""

    def __init__(self):
        super().__init__()
        self.name = "chain_length"
        self.question = ("Report the length of the longest carbon-only chain not "
                         "contained within a ring in this molecule. Answer with an integer only.")

    def generate(self, smiles: str, mol: Chem.Mol, rng: random.Random) -> list[Sample]:
        length = self._longest_acyclic_carbon_chain(mol)
        if length < 1:
            return list()
        return [Sample(smiles=smiles, task=self.name, question=self.question, answer=str(length),
                difficulty=self.difficulty_key(mol, smiles), metadata={"chain_length": length})]

    @staticmethod
    def _longest_acyclic_carbon_chain(mol: Chem.Mol) -> int:
        """Find the longest path of carbon atoms not in a ring"""
        ring_atoms: set[int] = set()
        for ring in mol.GetRingInfo().AtomRings():
            ring_atoms.update(ring)
        # Build graph of acyclic carbons only
        g = nx.Graph()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx not in ring_atoms and atom.GetAtomicNum() == 6:
                g.add_node(idx)
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in g.nodes and b in g.nodes:
                g.add_edge(a, b)
        if len(g.nodes) == 0:
            return 0
        # Longest path in a forest = max diameter
        # (+1 to convert edges → atom count)
        longest = 0
        for comp in nx.connected_components(g):
            sub = g.subgraph(comp)
            if len(sub) == 1:
                longest = max(longest, 1)
            else:
                try:
                    diam = nx.diameter(sub)
                    longest = max(longest, diam + 1)
                except nx.NetworkXError:
                    longest = max(longest, len(sub))
        return longest

    def difficulty_key(self, mol: Chem.Mol, smiles: str) -> float:
        """Molecules w/ many branches (parentheses in SMILES) are harder"""
        return float(smiles.count("("))