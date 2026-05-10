"""Unit tests for SMILES parsing task implementations."""

import random
import pytest
from rdkit import Chem

from clarimol.data.functional_group import FunctionalGroup
from clarimol.data.ring_counting import RingCounting
from clarimol.data.chain_length import ChainLength
from clarimol.data.canonicalization import Canonicalization
from clarimol.data.fragment_assembly import FragmentAssembly
from clarimol.data.tasks import get_task, TASK_REGISTRY


# Known molecules for testing
ETHANOL = "CCO"
BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
NAPHTHALENE = "c1ccc2ccccc2c1"
HEXANE = "CCCCCC"
PHENOL = "Oc1ccccc1"
ACETIC_ACID = "CC(=O)O"


@pytest.fixture
def rng():
    return random.Random(42)


class TestFunctionalGroup:
    def test_generates_samples(self, rng):
        task = FunctionalGroup()
        mol = Chem.MolFromSmiles(ETHANOL)
        samples = task.generate(ETHANOL, mol, rng)
        assert len(samples) > 0
        for s in samples:
            assert s.task == "functional_group"
            assert s.answer in ("Yes", "No")

    def test_positive_and_negative(self, rng):
        task = FunctionalGroup()
        mol = Chem.MolFromSmiles(ASPIRIN)
        samples = task.generate(ASPIRIN, mol, rng)
        answers = {s.answer for s in samples}
        # Aspirin has functional groups so should get at least one Yes
        assert "Yes" in answers

    def test_hydroxyl_in_ethanol(self, rng):
        task = FunctionalGroup()
        mol = Chem.MolFromSmiles(ETHANOL)
        samples = task.generate(ETHANOL, mol, rng)
        # Should detect hydroxyl (OH)
        yes_samples = [s for s in samples if s.answer == "Yes"]
        if yes_samples:
            assert yes_samples[0].metadata.get("fg_name") is not None

    def test_difficulty_increases_with_complexity(self, rng):
        task = FunctionalGroup()
        simple_mol = Chem.MolFromSmiles(ETHANOL)
        complex_mol = Chem.MolFromSmiles(ASPIRIN)
        d_simple = task.difficulty_key(simple_mol, ETHANOL)
        d_complex = task.difficulty_key(complex_mol, ASPIRIN)
        assert d_complex >= d_simple

    def test_metadata_contains_fg_info(self, rng):
        task = FunctionalGroup()
        mol = Chem.MolFromSmiles(ETHANOL)
        samples = task.generate(ETHANOL, mol, rng)
        for s in samples:
            assert "fg_name" in s.metadata
            assert "fg_smarts" in s.metadata


class TestRingCounting:
    def test_no_rings(self, rng):
        task = RingCounting()
        mol = Chem.MolFromSmiles(HEXANE)
        samples = task.generate(HEXANE, mol, rng)
        assert len(samples) == 0

    def test_benzene_has_one_ring(self, rng):
        task = RingCounting()
        mol = Chem.MolFromSmiles(BENZENE)
        samples = task.generate(BENZENE, mol, rng)
        assert len(samples) == 1
        assert samples[0].answer == "1"
        assert samples[0].metadata["ring_size"] == 6

    def test_naphthalene_has_two_rings(self, rng):
        task = RingCounting()
        mol = Chem.MolFromSmiles(NAPHTHALENE)
        samples = task.generate(NAPHTHALENE, mol, rng)
        assert len(samples) == 1
        count = int(samples[0].answer)
        ring_size = samples[0].metadata["ring_size"]
        # Naphthalene has 2 six-membered rings
        assert ring_size == 6
        assert count == 2

    def test_difficulty_by_ring_count(self, rng):
        task = RingCounting()
        benzene_mol = Chem.MolFromSmiles(BENZENE)
        naph_mol = Chem.MolFromSmiles(NAPHTHALENE)
        assert task.difficulty_key(naph_mol, NAPHTHALENE) > task.difficulty_key(benzene_mol, BENZENE)


class TestChainLength:
    def test_hexane_chain(self, rng):
        task = ChainLength()
        mol = Chem.MolFromSmiles(HEXANE)
        samples = task.generate(HEXANE, mol, rng)
        assert len(samples) == 1
        assert samples[0].answer == "6"

    def test_benzene_no_acyclic_chain(self, rng):
        task = ChainLength()
        mol = Chem.MolFromSmiles(BENZENE)
        samples = task.generate(BENZENE, mol, rng)
        # Benzene has no acyclic carbons
        assert len(samples) == 0

    def test_ethanol_chain(self, rng):
        task = ChainLength()
        mol = Chem.MolFromSmiles(ETHANOL)
        samples = task.generate(ETHANOL, mol, rng)
        assert len(samples) == 1
        assert samples[0].answer == "2"

    def test_branched_molecule(self, rng):
        task = ChainLength()
        isobutane = "CC(C)C"
        mol = Chem.MolFromSmiles(isobutane)
        samples = task.generate(isobutane, mol, rng)
        assert len(samples) == 1
        # Longest chain in isobutane is 3
        assert samples[0].answer == "3"


class TestCanonicalization:
    def test_generates_sample(self, rng):
        task = Canonicalization()
        mol = Chem.MolFromSmiles(ASPIRIN)
        samples = task.generate(ASPIRIN, mol, rng)
        # May be empty if randomization produces canonical form
        if samples:
            assert samples[0].task == "canonicalization"
            # Answer should be canonical SMILES
            answer_mol = Chem.MolFromSmiles(samples[0].answer)
            assert answer_mol is not None

    def test_randomized_differs_from_canonical(self, rng):
        task = Canonicalization()
        mol = Chem.MolFromSmiles(ASPIRIN)
        samples = task.generate(ASPIRIN, mol, rng)
        if samples:
            assert samples[0].smiles != samples[0].answer

    def test_answer_is_canonical(self, rng):
        task = Canonicalization()
        mol = Chem.MolFromSmiles(ASPIRIN)
        canonical = Chem.MolToSmiles(mol, canonical=True)
        samples = task.generate(ASPIRIN, mol, rng)
        if samples:
            assert samples[0].answer == canonical

    def test_small_molecule_may_skip(self, rng):
        task = Canonicalization()
        methane = "C"
        mol = Chem.MolFromSmiles(methane)
        samples = task.generate(methane, mol, rng)
        # Methane has only 1 atom, can't randomize
        assert len(samples) == 0


class TestFragmentAssembly:
    def test_generates_sample(self, rng):
        task = FragmentAssembly()
        mol = Chem.MolFromSmiles(ASPIRIN)
        samples = task.generate(ASPIRIN, mol, rng)
        # May or may not produce depending on BRICS decomposition
        for s in samples:
            assert s.task == "fragment_assembly"
            assert "." in s.smiles or " . " in s.smiles

    def test_answer_is_valid_smiles(self, rng):
        task = FragmentAssembly()
        mol = Chem.MolFromSmiles(ASPIRIN)
        samples = task.generate(ASPIRIN, mol, rng)
        for s in samples:
            answer_mol = Chem.MolFromSmiles(s.answer)
            assert answer_mol is not None

    def test_simple_molecule_uses_bond_cleavage(self, rng):
        task = FragmentAssembly()
        propane = "CCC"
        mol = Chem.MolFromSmiles(propane)
        samples = task.generate(propane, mol, rng)
        # Propane has no BRICS bonds, should use fallback
        for s in samples:
            assert len(s.metadata.get("frag1", "")) > 0


class TestTaskRegistry:
    def test_all_tasks_registered(self):
        expected = {"functional_group", "ring_counting", "chain_length",
                    "canonicalization", "fragment_assembly"}
        assert set(TASK_REGISTRY.keys()) == expected

    def test_get_task_returns_instance(self):
        for name in TASK_REGISTRY:
            task = get_task(name)
            assert task.name == name

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent_task")
