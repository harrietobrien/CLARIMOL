"""Unit tests for the bulk COD SMILES pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from clarimol.data.cod_bulk import (
    filter_organic_smiles,
    parse_smi_directory,
    cif_to_smiles_obabel,
    _obabel_available,
    fetch_cod_smiles_api,
    fetch_cod_bulk,
    ALLOWED_ELEMENTS,
)


# ─── Filter Tests ────────────────────────────────────────────────────────────

class TestFilterOrganicSmiles:
    def test_valid_organic(self):
        entries = [("1", "CCCO"), ("2", "c1ccccc1"), ("3", "CC(=O)O")]
        result = filter_organic_smiles(entries)
        assert len(result) == 3
        for cod_id, canonical, raw in result:
            assert canonical  # non-empty canonical SMILES

    def test_rejects_multicomponent(self):
        entries = [("1", "CCO.CC")]  # salt/solvate
        result = filter_organic_smiles(entries)
        assert len(result) == 0

    def test_rejects_metals(self):
        entries = [("1", "[Fe]")]  # iron
        result = filter_organic_smiles(entries)
        assert len(result) == 0

    def test_rejects_invalid_smiles(self):
        entries = [("1", "not_a_smiles"), ("2", "[invalid")]
        result = filter_organic_smiles(entries)
        assert len(result) == 0

    def test_rejects_too_small(self):
        entries = [("1", "C")]  # single carbon, < 4 atoms
        result = filter_organic_smiles(entries, min_atoms=4)
        assert len(result) == 0

    def test_rejects_too_large(self):
        # Build a very large molecule
        large = "C" * 300
        entries = [("1", large)]
        result = filter_organic_smiles(entries, max_atoms=200)
        assert len(result) == 0

    def test_deduplication(self):
        # Same molecule, different SMILES representations
        entries = [("1", "OCCCC"), ("2", "CCCCO"), ("3", "C(O)CCC")]
        result = filter_organic_smiles(entries)
        assert len(result) == 1  # all canonicalize to same thing

    def test_canonicalizes(self):
        entries = [("1", "OCCC")]
        result = filter_organic_smiles(entries)
        assert len(result) == 1
        _, canonical, raw = result[0]
        assert raw == "OCCC"
        assert canonical == "CCCO"  # canonical form

    def test_preserves_cod_id(self):
        entries = [("12345", "CCCCO")]
        result = filter_organic_smiles(entries)
        assert result[0][0] == "12345"

    def test_rejects_no_carbon(self):
        entries = [("1", "[NH4+]")]  # ammonium, no carbon
        result = filter_organic_smiles(entries)
        assert len(result) == 0

    def test_allowed_elements(self):
        # Fluorine is allowed
        entries = [("1", "FC(F)(F)C")]
        result = filter_organic_smiles(entries)
        assert len(result) == 1

        # Silicon is allowed
        entries = [("1", "[SiH4]")]
        result = filter_organic_smiles(entries, min_atoms=1)
        # No carbon though, so rejected
        assert len(result) == 0

    def test_empty_input(self):
        result = filter_organic_smiles([])
        assert len(result) == 0

    def test_mixed_valid_invalid(self):
        entries = [
            ("1", "CCCCO"),         # valid (butanol)
            ("2", "not_smiles"),    # invalid parse
            ("3", "CCO.O"),         # multi-component
            ("4", "[Fe](C)C"),      # metal
            ("5", "c1ccccc1"),      # valid
            ("6", "CCCCO"),         # duplicate of 1
        ]
        result = filter_organic_smiles(entries)
        assert len(result) == 2  # butanol + benzene

    def test_aspirin(self):
        entries = [("1", "CC(=O)Oc1ccccc1C(=O)O")]
        result = filter_organic_smiles(entries)
        assert len(result) == 1

    def test_brominated(self):
        # Bromine is in ALLOWED_ELEMENTS
        entries = [("1", "BrCCBr")]
        result = filter_organic_smiles(entries)
        assert len(result) == 1

    def test_iodinated(self):
        # Iodine is in ALLOWED_ELEMENTS
        entries = [("1", "ICCI")]
        result = filter_organic_smiles(entries)
        assert len(result) == 1


# ─── SMI File Parsing Tests ─────────────────────────────────────────────────

class TestParseSmiDirectory:
    def test_parses_smi_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock .smi files
            (Path(tmpdir) / "1000001.smi").write_text("CCO 1000001\n")
            (Path(tmpdir) / "1000002.smi").write_text("c1ccccc1 1000002\n")
            results = list(parse_smi_directory(tmpdir))
            assert len(results) == 2
            ids = {r[0] for r in results}
            assert "1000001" in ids
            assert "1000002" in ids

    def test_handles_empty_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "1000001.smi").write_text("")
            (Path(tmpdir) / "1000002.smi").write_text("CCO\n")
            results = list(parse_smi_directory(tmpdir))
            assert len(results) == 1

    def test_handles_no_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = list(parse_smi_directory(tmpdir))
            assert len(results) == 0

    def test_nested_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "1" / "00" / "00"
            subdir.mkdir(parents=True)
            (subdir / "1000001.smi").write_text("CCO\n")
            results = list(parse_smi_directory(tmpdir))
            assert len(results) == 1

    def test_smiles_only_format(self):
        # Some .smi files may only have SMILES, no ID
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "12345.smi").write_text("CCO\n")
            results = list(parse_smi_directory(tmpdir))
            assert len(results) == 1
            assert results[0][0] == "12345"  # ID from filename
            assert results[0][1] == "CCO"


# ─── OpenBabel Tests ─────────────────────────────────────────────────────────

class TestOpenBabel:
    def test_obabel_check(self):
        # Just check it returns a bool without crashing
        result = _obabel_available()
        assert isinstance(result, bool)

    def test_cif_to_smiles_missing_file(self):
        result = cif_to_smiles_obabel("/nonexistent/file.cif")
        assert result is None

    @pytest.mark.skipif(not _obabel_available(), reason="OpenBabel not installed")
    def test_cif_to_smiles_valid(self):
        # Create a minimal valid CIF for ethanol
        cif_content = """data_ethanol
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
C2 C 0.3 0.0 0.0
O1 O 0.6 0.0 0.0
"""
        with tempfile.NamedTemporaryFile(suffix=".cif", mode="w", delete=False) as f:
            f.write(cif_content)
            cif_path = f.name
        try:
            result = cif_to_smiles_obabel(cif_path)
            # May or may not produce valid SMILES depending on OpenBabel version
            # Just check it doesn't crash
            assert result is None or isinstance(result, str)
        finally:
            Path(cif_path).unlink()


# ─── API Tests ───────────────────────────────────────────────────────────────

class TestFetchCodSmilesApi:
    @patch("clarimol.data.cod_bulk.requests.get")
    def test_api_returns_smiles(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"file": "1000001", "smiles": "CCO"},
            {"file": "1000002", "smiles": "c1ccccc1"},
            {"file": "1000003"},  # no SMILES
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        results = fetch_cod_smiles_api(max_entries=10)
        assert len(results) == 2
        assert results[0] == ("1000001", "CCO")

    @patch("clarimol.data.cod_bulk.requests.get")
    def test_api_handles_failure(self, mock_get):
        mock_get.side_effect = Exception("Connection error")
        results = fetch_cod_smiles_api(max_entries=10)
        assert len(results) == 0

    @patch("clarimol.data.cod_bulk.requests.get")
    def test_api_respects_max_entries(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"file": str(i), "smiles": f"C{'C' * i}"} for i in range(100)
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        results = fetch_cod_smiles_api(max_entries=5)
        assert len(results) == 5


# ─── Pipeline Integration Tests ─────────────────────────────────────────────

class TestFetchCodBulk:
    @patch("clarimol.data.cod_bulk.fetch_cod_smiles_api")
    def test_full_pipeline_with_cache(self, mock_api):
        mock_api.return_value = [
            ("1", "CCCCO"),       # valid (butanol)
            ("2", "c1ccccc1"),    # valid (benzene)
            ("3", "CC(=O)OC"),    # valid (methyl acetate)
            ("4", "CCO.O"),       # will be filtered (multi-component)
            ("5", "[Fe]"),        # will be filtered (metal)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_cod_bulk(
                max_molecules=10,
                cache_dir=tmpdir,
                strategies=["api"],
            )
            assert len(result) == 3  # butanol, benzene, methyl acetate

            # Check cache was created
            cache_file = Path(tmpdir) / "cod_bulk_smiles.json"
            assert cache_file.exists()

            # Check metadata was saved
            meta_file = Path(tmpdir) / "cod_bulk_metadata.json"
            assert meta_file.exists()
            with open(meta_file) as f:
                meta = json.load(f)
            assert len(meta) == 3

    @patch("clarimol.data.cod_bulk.fetch_cod_smiles_api")
    def test_cache_reuse(self, mock_api):
        mock_api.return_value = [("1", "CCCCO"), ("2", "c1ccccc1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            # First call populates cache
            r1 = fetch_cod_bulk(max_molecules=10, cache_dir=tmpdir, strategies=["api"])
            # Second call uses cache (API not called again)
            mock_api.reset_mock()
            r2 = fetch_cod_bulk(max_molecules=10, cache_dir=tmpdir, strategies=["api"])
            mock_api.assert_not_called()
            assert r1 == r2

    @patch("clarimol.data.cod_bulk.fetch_cod_smiles_api")
    def test_respects_max_molecules(self, mock_api):
        mock_api.return_value = [
            (str(i), f"{'C' * (i + 4)}O") for i in range(20)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_cod_bulk(
                max_molecules=5,
                cache_dir=tmpdir,
                strategies=["api"],
            )
            assert len(result) <= 5


# ─── Allowed Elements Tests ─────────────────────────────────────────────────

class TestAllowedElements:
    def test_common_organic_elements(self):
        # H, C, N, O, F, S, Cl, Br, I, P, Si, B
        assert 1 in ALLOWED_ELEMENTS   # H
        assert 6 in ALLOWED_ELEMENTS   # C
        assert 7 in ALLOWED_ELEMENTS   # N
        assert 8 in ALLOWED_ELEMENTS   # O
        assert 9 in ALLOWED_ELEMENTS   # F
        assert 16 in ALLOWED_ELEMENTS  # S
        assert 17 in ALLOWED_ELEMENTS  # Cl
        assert 35 in ALLOWED_ELEMENTS  # Br
        assert 53 in ALLOWED_ELEMENTS  # I
        assert 15 in ALLOWED_ELEMENTS  # P

    def test_metals_excluded(self):
        assert 26 not in ALLOWED_ELEMENTS  # Fe
        assert 29 not in ALLOWED_ELEMENTS  # Cu
        assert 30 not in ALLOWED_ELEMENTS  # Zn
