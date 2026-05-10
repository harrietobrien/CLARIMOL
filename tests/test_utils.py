"""Unit tests for utility functions."""

from clarimol.utils.chem import is_valid_smiles, canonicalize


class TestIsValidSmiles:
    def test_valid(self):
        assert is_valid_smiles("CCO") is True
        assert is_valid_smiles("c1ccccc1") is True
        assert is_valid_smiles("CC(=O)O") is True

    def test_invalid(self):
        assert is_valid_smiles("not_smiles") is False
        assert is_valid_smiles("[invalid") is False


class TestCanonicalize:
    def test_basic(self):
        assert canonicalize("OCC") == canonicalize("CCO")

    def test_already_canonical(self):
        can = canonicalize("CCO")
        assert can is not None
        assert canonicalize(can) == can

    def test_invalid_returns_none(self):
        assert canonicalize("not_a_molecule") is None
        assert canonicalize("[invalid") is None

    def test_benzene_variants(self):
        assert canonicalize("c1ccccc1") == canonicalize("C1=CC=CC=C1")
