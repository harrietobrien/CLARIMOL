from __future__ import annotations
import random
from importlib import resources
from pathlib import Path
import yaml
from rdkit import Chem
from clarimol.data.parsing import Parsing
from clarimol.data.sample import Sample


# Load SMARTS patterns from YAML
def _load_functional_groups() -> dict[str, str]:
    """Load functional group name → SMARTS mapping from the YAML config"""
    yaml_path = Path(__file__).parent / "functional_groups.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f)


FUNCTIONAL_GROUPS: dict[str, str] = _load_functional_groups()

# Pre-compile SMARTS patterns (lazy cache)
_FG_PATTERNS: dict[str, Chem.Mol | None] = {}


def _get_fg_pattern(name: str) -> Chem.Mol | None:
    """Get a compiled SMARTS pattern (caching on first use)"""
    if name not in _FG_PATTERNS:
        smarts = FUNCTIONAL_GROUPS.get(name)
        _FG_PATTERNS[name] = Chem.MolFromSmarts(smarts) if smarts else None
    return _FG_PATTERNS[name]


class FunctionalGroup(Parsing):
    """Functional Group Matching: Binary Substructure Detection"""
    def __init__(self):
        super().__init__()
        self.name = "functional_group"

    def generate(self, smiles: str, mol: Chem.Mol, rng: random.Random) -> list[Sample]:
        samples: list[Sample] = []
        fg_names = list(FUNCTIONAL_GROUPS.keys())
        rng.shuffle(fg_names)
        # Pick one positive and one negative example (if available)
        pos_done = neg_done = False
        for fg_name in fg_names:
            if pos_done and neg_done:
                break
            pat = _get_fg_pattern(fg_name)
            if pat is None:
                continue
            has_fg = mol.HasSubstructMatch(pat)
            if has_fg and not pos_done:
                samples.append(self._make_sample(smiles, fg_name, "Yes", mol))
                pos_done = True
            elif not has_fg and not neg_done:
                samples.append(self._make_sample(smiles, fg_name, "No", mol))
                neg_done = True
        return samples

    def _make_sample(
        self, smiles: str, fg_name: str, answer: str, mol: Chem.Mol
    ) -> Sample:
        fg_smarts = FUNCTIONAL_GROUPS[fg_name]
        human_name = fg_name.replace("_", " ")
        return Sample(
            smiles=smiles,
            task=self.name,
            question=(
                f"Does the molecule represented by this SMILES contain "
                f"the functional group {human_name} ({fg_smarts})? "
                f"Respond with 'Yes' or 'No'."
            ),
            answer=answer,
            difficulty=self.difficulty_key(mol, smiles),
            metadata={"fg_name": fg_name, "fg_smarts": fg_smarts},
        )

    def difficulty_key(self, mol: Chem.Mol, smiles: str) -> float:
        """More functional groups present = harder to reason about"""
        count = 0
        for fg_name in FUNCTIONAL_GROUPS:
            pat = _get_fg_pattern(fg_name)
            if pat and mol.HasSubstructMatch(pat):
                count += 1
        return float(count)