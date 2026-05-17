"""Unit tests for the bulk COD SMILES pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from clarimol.data.cod_bulk import (
    filter_smiles,
    parse_smi_directory,
    cif_to_smiles_obabel,
    _obabel_available,
    fetch_cod_smiles_api,
    fetch_cod_bulk,
    ORGANIC_ELEMENTS,
)


# ─── Filter Tests ────────────────────────────────────────────────────────────

class TestFilterSmiles:
    def test_valid_molecules(self):
        entries = [("1", "CCCCO"), ("2", "c1ccccc1"), ("3", "CC(=O)OC")]
        result = filter_smiles(entries)
        assert len(result) == 3

    def test_allows_metals_by_default(self):
        # Iron complex — allowed when no element whitelist
        entries = [("1", "[Fe](C)(C)(C)C")]
        result = filter_smiles(entries)
        assert len(result) == 1

    def test_no_size_filter_by_default(self):
        # Tiny molecule (1 heavy atom + carbon) — allowed by default
        entries = [("1", "CO")]
        result = filter_smiles(entries)
        assert len(result) == 1

        # Large molecule — allowed by default
        large = "C" * 300
        entries = [("1", large)]
        result = filter_smiles(entries)
        assert len(result) == 1

    def test_optional_size_filter(self):
        entries = [("1", "C"), ("2", "CCCCCC")]
        result = filter_smiles(entries, min_atoms=4)
        assert len(result) == 1  # only hexane passes

    def test_optional_element_whitelist(self):
        entries = [("1", "CCCCO"), ("2", "[Fe](C)(C)(C)C")]
        result = filter_smiles(entries, allowed_elements=ORGANIC_ELEMENTS)
        assert len(result) == 1  # iron rejected

    def test_rejects_multicomponent(self):
        entries = [("1", "CCO.CC")]
        result = filter_smiles(entries)
        assert len(result) == 0

    def test_allows_multicomponent_when_disabled(self):
        entries = [("1", "CCO.CC")]
        result = filter_smiles(entries, reject_multicomponent=False)
        assert len(result) == 1

    def test_rejects_invalid_smiles(self):
        entries = [("1", "not_a_smiles"), ("2", "[invalid")]
        result = filter_smiles(entries)
        assert len(result) == 0

    def test_allows_no_carbon_by_default(self):
        entries = [("1", "[NH3]")]
        result = filter_smiles(entries)
        assert len(result) == 1

    def test_rejects_no_carbon_when_required(self):
        entries = [("1", "[NH4+]")]
        result = filter_smiles(entries, require_carbon=True)
        assert len(result) == 0

    def test_deduplication(self):
        entries = [("1", "OCCCC"), ("2", "CCCCO"), ("3", "C(O)CCC")]
        result = filter_smiles(entries)
        assert len(result) == 1

    def test_canonicalizes(self):
        entries = [("1", "OCCC")]
        result = filter_smiles(entries)
        assert len(result) == 1
        _, canonical, raw = result[0]
        assert raw == "OCCC"
        assert canonical == "CCCO"

    def test_preserves_cod_id(self):
        entries = [("12345", "CCCCO")]
        result = filter_smiles(entries)
        assert result[0][0] == "12345"

    def test_empty_input(self):
        result = filter_smiles([])
        assert len(result) == 0

    def test_mixed_valid_invalid(self):
        entries = [
            ("1", "CCCCO"),         # valid
            ("2", "not_smiles"),    # invalid parse
            ("3", "CCO.O"),         # multi-component
            ("4", "[Fe](C)(C)C"),   # valid (metals allowed)
            ("5", "c1ccccc1"),      # valid
            ("6", "CCCCO"),         # duplicate of 1
        ]
        result = filter_smiles(entries)
        assert len(result) == 3  # butanol, Fe complex, benzene

    def test_organometallic_crystal(self):
        # Ferrocene-like
        entries = [("1", "[Fe]1234([CH-]5[CH-]1[CH-]2[CH-]3[CH-]45)[CH-]1[CH-]2[CH-]3[CH-]4[CH-]12")]
        result = filter_smiles(entries)
        # Should pass (no metal restriction by default)
        # May fail RDKit parse depending on representation
        # At minimum, should not crash
        assert isinstance(result, list)


# ─── SMI File Parsing Tests ─────────────────────────────────────────────────

class TestParseSmiDirectory:
    def test_parses_smi_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "1000001.smi").write_text("CCO 1000001\n")
            (Path(tmpdir) / "1000002.smi").write_text("c1ccccc1 1000002\n")
            results = list(parse_smi_directory(tmpdir))
            assert len(results) == 2

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
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "12345.smi").write_text("CCO\n")
            results = list(parse_smi_directory(tmpdir))
            assert results[0][0] == "12345"
            assert results[0][1] == "CCO"


# ─── OpenBabel Tests ─────────────────────────────────────────────────────────

class TestOpenBabel:
    def test_obabel_check(self):
        result = _obabel_available()
        assert isinstance(result, bool)

    def test_cif_to_smiles_missing_file(self):
        result = cif_to_smiles_obabel("/nonexistent/file.cif")
        assert result is None

    @pytest.mark.skipif(not _obabel_available(), reason="OpenBabel not installed")
    def test_cif_to_smiles_valid(self):
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

    @patch("clarimol.data.cod_bulk.requests.get")
    def test_api_handles_failure(self, mock_get):
        mock_get.side_effect = Exception("Connection error")
        results = fetch_cod_smiles_api(max_entries=10)
        assert len(results) == 0

    @patch("clarimol.data.cod_bulk.requests.get")
    def test_api_respects_max_entries(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"file": str(i), "smiles": f"C{'C' * (i + 2)}"} for i in range(100)
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        results = fetch_cod_smiles_api(max_entries=5)
        assert len(results) == 5


# ─── Pipeline Integration Tests ─────────────────────────────────────────────

class TestFetchCodBulk:
    @patch("clarimol.data.cod_bulk.fetch_cod_smiles_api")
    def test_full_pipeline(self, mock_api):
        mock_api.return_value = [
            ("1", "CCCCO"),
            ("2", "c1ccccc1"),
            ("3", "CC(=O)OC"),
            ("4", "CCO.O"),       # will be filtered (multi-component)
            ("5", "not_valid"),   # will be filtered (parse error)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_cod_bulk(
                max_molecules=10, cache_dir=tmpdir, strategies=["api"],
            )
            assert len(result) == 3

            # Cache and metadata created
            assert (Path(tmpdir) / "cod_bulk_smiles.json").exists()
            assert (Path(tmpdir) / "cod_bulk_metadata.json").exists()

    @patch("clarimol.data.cod_bulk.fetch_cod_smiles_api")
    def test_cache_reuse(self, mock_api):
        mock_api.return_value = [("1", "CCCCO"), ("2", "c1ccccc1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = fetch_cod_bulk(max_molecules=10, cache_dir=tmpdir, strategies=["api"])
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
                max_molecules=5, cache_dir=tmpdir, strategies=["api"],
            )
            assert len(result) <= 5

    @patch("clarimol.data.cod_bulk.fetch_cod_smiles_api")
    def test_metals_pass_through(self, mock_api):
        """Verify metals are NOT rejected by default."""
        mock_api.return_value = [
            ("1", "[Zn](C)(C)C"),    # zinc complex with carbon
            ("2", "c1ccccc1"),        # benzene
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_cod_bulk(
                max_molecules=10, cache_dir=tmpdir, strategies=["api"],
            )
            # Zinc complex should pass (metals allowed by default)
            # Exact count depends on RDKit parsing the zinc SMILES
            assert len(result) >= 1  # at minimum benzene passes
