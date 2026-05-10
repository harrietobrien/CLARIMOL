"""Unit tests for data pruning and curriculum ordering."""

import pytest

from clarimol.data.sample import Sample
from clarimol.data.pruning import PruningConfig, prune_and_sort, CURRICULUM_ORDERS


def _make_samples(n: int, task: str = "test") -> list[Sample]:
    """Create n samples with difficulty 0..n-1."""
    return [
        Sample(smiles=f"C{'C' * i}", task=task, question="q",
               answer="a", difficulty=float(i))
        for i in range(n)
    ]


class TestPruningConfig:
    def test_defaults(self):
        cfg = PruningConfig()
        assert cfg.curriculum_order == "easy-hard"
        assert cfg.keep_n == 50_000

    def test_legacy_sort_curriculum_true(self):
        cfg = PruningConfig(sort_curriculum=True)
        assert cfg.curriculum_order == "easy-hard"

    def test_legacy_sort_curriculum_false(self):
        cfg = PruningConfig(sort_curriculum=False)
        assert cfg.curriculum_order == "none"

    def test_invalid_curriculum_order(self):
        with pytest.raises(ValueError, match="Unknown curriculum_order"):
            PruningConfig(curriculum_order="backwards")

    def test_valid_orders(self):
        for order in CURRICULUM_ORDERS:
            cfg = PruningConfig(curriculum_order=order)
            assert cfg.curriculum_order == order


class TestPruneAndSort:
    def test_empty_input(self):
        assert prune_and_sort([], PruningConfig()) == []

    def test_keep_n_respected(self):
        samples = _make_samples(100)
        cfg = PruningConfig(keep_n=10, subsample="random")
        result = prune_and_sort(samples, cfg)
        assert len(result) == 10

    def test_keep_n_when_fewer_samples(self):
        samples = _make_samples(5)
        cfg = PruningConfig(keep_n=100, subsample="random")
        result = prune_and_sort(samples, cfg)
        assert len(result) == 5

    def test_easy_hard_ordering(self):
        samples = _make_samples(20)
        cfg = PruningConfig(keep_n=20, subsample="random", curriculum_order="easy-hard")
        result = prune_and_sort(samples, cfg)
        diffs = [s.difficulty for s in result]
        assert diffs == sorted(diffs)

    def test_hard_easy_ordering(self):
        samples = _make_samples(20)
        cfg = PruningConfig(keep_n=20, subsample="random", curriculum_order="hard-easy")
        result = prune_and_sort(samples, cfg)
        diffs = [s.difficulty for s in result]
        assert diffs == sorted(diffs, reverse=True)

    def test_random_ordering(self):
        samples = _make_samples(100)
        cfg = PruningConfig(keep_n=100, subsample="bottom", curriculum_order="random")
        result = prune_and_sort(samples, cfg, seed=42)
        diffs = [s.difficulty for s in result]
        # Very unlikely to be sorted with 100 elements
        assert diffs != sorted(diffs)

    def test_none_ordering_preserves_subsample_order(self):
        samples = _make_samples(20)
        cfg = PruningConfig(keep_n=10, subsample="bottom", curriculum_order="none")
        result = prune_and_sort(samples, cfg)
        # "bottom" selects easiest, "none" preserves that order
        diffs = [s.difficulty for s in result]
        assert diffs == sorted(diffs)

    def test_middle_subsample(self):
        samples = _make_samples(100)
        cfg = PruningConfig(keep_n=50, subsample="middle", trim_fraction=0.1,
                            curriculum_order="none")
        result = prune_and_sort(samples, cfg)
        # Should not contain the very easiest or hardest
        diffs = [s.difficulty for s in result]
        assert min(diffs) > 0  # trimmed bottom
        assert max(diffs) < 99  # trimmed top

    def test_top_subsample(self):
        samples = _make_samples(100)
        cfg = PruningConfig(keep_n=10, subsample="top", curriculum_order="none")
        result = prune_and_sort(samples, cfg)
        diffs = [s.difficulty for s in result]
        assert min(diffs) >= 90

    def test_bottom_subsample(self):
        samples = _make_samples(100)
        cfg = PruningConfig(keep_n=10, subsample="bottom", curriculum_order="none")
        result = prune_and_sort(samples, cfg)
        diffs = [s.difficulty for s in result]
        assert max(diffs) <= 9

    def test_deterministic_with_seed(self):
        samples = _make_samples(100)
        cfg = PruningConfig(keep_n=10, subsample="random", curriculum_order="random")
        r1 = prune_and_sort(samples, cfg, seed=42)
        r2 = prune_and_sort(samples, cfg, seed=42)
        assert [s.difficulty for s in r1] == [s.difficulty for s in r2]

    def test_different_seeds_differ(self):
        samples = _make_samples(100)
        cfg = PruningConfig(keep_n=10, subsample="random", curriculum_order="random")
        r1 = prune_and_sort(samples, cfg, seed=42)
        r2 = prune_and_sort(samples, cfg, seed=99)
        assert [s.difficulty for s in r1] != [s.difficulty for s in r2]

    def test_does_not_mutate_input(self):
        samples = _make_samples(10)
        original_order = [s.difficulty for s in samples]
        cfg = PruningConfig(keep_n=5, subsample="random", curriculum_order="hard-easy")
        prune_and_sort(samples, cfg)
        assert [s.difficulty for s in samples] == original_order
