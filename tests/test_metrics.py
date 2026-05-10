"""Unit tests for evaluation metrics."""

import pytest

from clarimol.eval.metrics import (
    _extract_yes_no,
    _extract_integer,
    _extract_smiles,
    _is_selfies,
    evaluate_parsing,
    evaluate_generation,
    ParsingResult,
    GenerationResult,
)


class TestExtractYesNo:
    def test_simple_yes(self):
        assert _extract_yes_no("Yes") == "Yes"

    def test_simple_no(self):
        assert _extract_yes_no("No") == "No"

    def test_case_insensitive(self):
        assert _extract_yes_no("yes") == "Yes"
        assert _extract_yes_no("NO") == "No"

    def test_embedded_in_text(self):
        assert _extract_yes_no("The answer is yes, the molecule contains it.") == "Yes"

    def test_no_match(self):
        assert _extract_yes_no("maybe") is None
        assert _extract_yes_no("") is None

    def test_prefers_first_match(self):
        assert _extract_yes_no("Yes, no, maybe") == "Yes"


class TestExtractInteger:
    def test_simple_integer(self):
        assert _extract_integer("3") == "3"

    def test_embedded_in_text(self):
        assert _extract_integer("The count is 5 rings.") == "5"

    def test_zero(self):
        assert _extract_integer("0") == "0"

    def test_no_match(self):
        assert _extract_integer("none") is None
        assert _extract_integer("") is None

    def test_prefers_first_integer(self):
        assert _extract_integer("There are 2 or 3 rings") == "2"


class TestExtractSmiles:
    def test_simple_smiles(self):
        assert _extract_smiles("CCO") == "CCO"

    def test_multiline_takes_first(self):
        assert _extract_smiles("CCO\nCC(=O)O") == "CCO"

    def test_strips_whitespace(self):
        assert _extract_smiles("  CCO  ") == "CCO"

    def test_empty_returns_empty(self):
        assert _extract_smiles("") == ""


class TestIsSelfies:
    def test_selfies_string(self):
        assert _is_selfies("[C][C][O]") is True

    def test_smiles_string(self):
        assert _is_selfies("CCO") is False

    def test_empty(self):
        assert _is_selfies("") is False


class TestEvaluateParsing:
    def test_functional_group_all_correct(self):
        preds = ["Yes", "No", "Yes"]
        refs = ["Yes", "No", "Yes"]
        result = evaluate_parsing(preds, refs, "functional_group")
        assert result.accuracy == 1.0
        assert result.correct == 3
        assert result.total == 3

    def test_functional_group_half_correct(self):
        preds = ["Yes", "Yes"]
        refs = ["Yes", "No"]
        result = evaluate_parsing(preds, refs, "functional_group")
        assert result.accuracy == 0.5

    def test_ring_counting(self):
        preds = ["1", "2", "3"]
        refs = ["1", "2", "4"]
        result = evaluate_parsing(preds, refs, "ring_counting")
        assert result.correct == 2
        assert result.accuracy == pytest.approx(2 / 3)

    def test_canonicalization_exact_match(self):
        preds = ["CCO", "c1ccccc1"]
        refs = ["CCO", "c1ccccc1"]
        result = evaluate_parsing(preds, refs, "canonicalization")
        assert result.accuracy == 1.0
        assert result.validity > 0

    def test_canonicalization_equivalent_smiles(self):
        # OCC and CCO represent the same molecule (ethanol)
        preds = ["OCC"]
        refs = ["CCO"]
        result = evaluate_parsing(preds, refs, "canonicalization")
        assert result.correct == 1

    def test_extraction_failure_counted(self):
        preds = ["maybe", "Yes"]
        refs = ["Yes", "Yes"]
        result = evaluate_parsing(preds, refs, "functional_group")
        assert result.extraction_failures == 1
        assert result.correct == 1

    def test_breakdown_with_metadata(self):
        preds = ["Yes", "No", "Yes"]
        refs = ["Yes", "No", "No"]
        meta = [
            {"fg_name": "hydroxyl"},
            {"fg_name": "hydroxyl"},
            {"fg_name": "amine_primary"},
        ]
        result = evaluate_parsing(preds, refs, "functional_group", metadata=meta)
        assert "hydroxyl" in result.breakdown
        assert "amine_primary" in result.breakdown
        assert result.breakdown["hydroxyl"]["accuracy"] == 1.0
        assert result.breakdown["amine_primary"]["accuracy"] == 0.0

    def test_ring_counting_breakdown_by_count(self):
        preds = ["1", "2", "1"]
        refs = ["1", "3", "1"]
        result = evaluate_parsing(preds, refs, "ring_counting")
        assert "count=1" in result.breakdown
        assert result.breakdown["count=1"]["accuracy"] == 1.0

    def test_chain_length_breakdown_buckets(self):
        preds = ["2", "5", "10"]
        refs = ["2", "5", "10"]
        result = evaluate_parsing(preds, refs, "chain_length")
        assert "short (1-3)" in result.breakdown
        assert "medium (4-7)" in result.breakdown
        assert "long (8+)" in result.breakdown

    def test_empty_predictions(self):
        result = evaluate_parsing([], [], "functional_group")
        assert result.accuracy == 0.0
        assert result.total == 0


class TestEvaluateGeneration:
    def test_perfect_match(self):
        preds = ["CCO", "c1ccccc1"]
        refs = ["CCO", "c1ccccc1"]
        result = evaluate_generation(preds, refs)
        assert result.exact_match == 1.0
        assert result.validity == 1.0

    def test_no_match(self):
        preds = ["CCCC", "CCCCC"]
        refs = ["CCO", "c1ccccc1"]
        result = evaluate_generation(preds, refs)
        assert result.exact_match == 0.0

    def test_levenshtein_normalized(self):
        preds = ["CCO"]
        refs = ["CCCO"]
        result = evaluate_generation(preds, refs)
        assert result.levenshtein > 0
        assert 0.0 < result.levenshtein_norm <= 1.0

    def test_empty_predictions(self):
        result = evaluate_generation([], [])
        assert result.exact_match == 0.0

    def test_invalid_smiles_lowers_validity(self):
        preds = ["CCO", "not_a_smiles"]
        refs = ["CCO", "c1ccccc1"]
        result = evaluate_generation(preds, refs)
        assert result.validity == 0.5

    def test_fingerprint_similarity_valid_pair(self):
        preds = ["CCO"]
        refs = ["CCO"]
        result = evaluate_generation(preds, refs)
        assert result.maccs_fps == 1.0
        assert result.rdk_fps == 1.0
        assert result.morgan_fps == 1.0
