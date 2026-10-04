#!/usr/bin/env python3
"""
Prepare SMILES-augmented training data for the robustness intervention experiment.

Reads canonical training data from data/clarimol_clean/ and produces four variants:
  1) augmented_canonical  -- original data unchanged (control)
  2) augmented_50_50      -- 50% canonical, 50% randomized (per-sample coin flip)
  3) augmented_random     -- 100% randomized SMILES
  4) augmented_curriculum -- first half canonical, second half randomized (by index)

The `smiles` field is modified; the `answer` field is never changed.
Fragment assembly samples (containing ' . ') have each fragment randomized
independently and rejoined.

RDKit's Chem.MolToSmiles(doRandom=True) is not seed-deterministic, so all
randomized variants are pre-generated here rather than on-the-fly during
training. Samples that fail to parse or randomize keep their canonical form.

Usage:
    python scripts/prepare_augmented_data.py [--input-dir data/clarimol_clean]
                                              [--output-base data]
                                              [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from copy import deepcopy
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TASKS = [
    "functional_group",
    "ring_counting",
    "chain_length",
    "canonicalization",
    "fragment_assembly",
]


def randomize_smiles(mol: Chem.Mol, max_attempts: int = 50) -> str | None:
    """Generate a single random (non-canonical) SMILES for a molecule.

    Returns None if randomization fails after max_attempts tries.
    """
    canonical = Chem.MolToSmiles(mol, canonical=True)
    for _ in range(max_attempts):
        try:
            rand = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            if rand and rand != canonical:
                return rand
        except Exception:
            continue
    # If the molecule has only one possible SMILES (e.g., single atom),
    # return the canonical form rather than failing.
    return canonical


def randomize_fragment_smiles(smiles: str, max_attempts: int = 50) -> str | None:
    """Randomize a dot-separated fragment SMILES string.

    Each fragment (separated by ' . ') is parsed and randomized independently.
    Dummy atoms like [10*] are preserved by RDKit's SMILES writer.
    Returns None only if any fragment fails to parse.
    """
    parts = smiles.split(" . ")
    mols = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        mols.append(mol)

    canonical_parts = [Chem.MolToSmiles(m, canonical=True) for m in mols]
    canonical_whole = " . ".join(canonical_parts)

    for _ in range(max_attempts):
        try:
            rand_parts = [
                Chem.MolToSmiles(m, doRandom=True, canonical=False) for m in mols
            ]
            combined = " . ".join(rand_parts)
            if combined != canonical_whole:
                return combined
        except Exception:
            continue

    return canonical_whole


def load_task_data(input_dir: Path, task: str) -> list[dict]:
    """Load JSON samples for a single task."""
    path = input_dir / f"{task}.json"
    if not path.exists():
        logger.warning("Missing task file: %s", path)
        return []
    with open(path) as f:
        return json.load(f)


def save_task_data(samples: list[dict], output_dir: Path, task: str) -> None:
    """Save JSON samples for a single task."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task}.json"
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)


def augment_sample(
    sample: dict,
    randomize: bool,
) -> tuple[dict, bool]:
    """Produce an augmented copy of a sample.

    If randomize=True, attempts to randomize the SMILES field.
    Returns (augmented_sample, was_randomized).
    The answer field is never modified.
    """
    out = deepcopy(sample)
    if not randomize:
        return out, False

    smiles = sample["smiles"]
    is_fragment = " . " in smiles

    if is_fragment:
        rand = randomize_fragment_smiles(smiles)
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return out, False
        rand = randomize_smiles(mol)

    if rand is None:
        return out, False

    out["smiles"] = rand

    # Also update the question field if it contains the original SMILES.
    # In the current dataset the question field does NOT embed the SMILES
    # (the SMILES is injected via instruction templates at training time
    # from the sample.smiles attribute), but handle this defensively in
    # case future data formats change.
    if smiles in out.get("question", ""):
        out["question"] = out["question"].replace(smiles, rand)

    return out, True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SMILES-augmented training data"
    )
    parser.add_argument(
        "--input-dir",
        default="data/clarimol_clean",
        help="Source training data directory (default: data/clarimol_clean)",
    )
    parser.add_argument(
        "--output-base",
        default="data",
        help="Base directory for output (default: data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the 50/50 coin flip and curriculum ordering",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_base = Path(args.output_base)
    rng = random.Random(args.seed)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    conditions = {
        "augmented_canonical": "control",
        "augmented_50_50": "50/50 mix",
        "augmented_random": "100% randomized",
        "augmented_curriculum": "canonical-first curriculum",
    }

    # Pre-generate randomized SMILES for every sample across all tasks.
    # A single randomized form per sample is stored and reused across
    # conditions that need it, ensuring consistency.
    logger.info("Loading source data from %s", input_dir)
    all_task_data: dict[str, list[dict]] = {}
    all_task_random: dict[str, list[str | None]] = {}

    for task in TASKS:
        samples = load_task_data(input_dir, task)
        if not samples:
            continue
        all_task_data[task] = samples
        logger.info("  %s: %d samples loaded", task, len(samples))

        # Pre-generate one randomized SMILES per sample
        randomized = []
        n_success = 0
        n_fail = 0
        for sample in samples:
            smiles = sample["smiles"]
            is_fragment = " . " in smiles

            if is_fragment:
                rand = randomize_fragment_smiles(smiles)
            else:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    rand = None
                else:
                    rand = randomize_smiles(mol)

            randomized.append(rand)
            if rand is not None and rand != smiles:
                n_success += 1
            else:
                n_fail += 1

        all_task_random[task] = randomized
        logger.info(
            "  %s: %d successfully randomized, %d kept canonical (parse failure or single-form molecule)",
            task,
            n_success,
            n_fail,
        )

    # Build each condition
    total_stats: dict[str, dict[str, int]] = {}

    for condition_name, description in conditions.items():
        out_dir = output_base / condition_name
        logger.info("Building condition: %s (%s) -> %s", condition_name, description, out_dir)
        condition_stats: dict[str, int] = {}

        for task in TASKS:
            if task not in all_task_data:
                continue

            samples = all_task_data[task]
            randomized = all_task_random[task]
            n_samples = len(samples)
            output_samples = []
            n_randomized = 0

            if condition_name == "augmented_canonical":
                # Control: keep everything canonical
                output_samples = deepcopy(samples)

            elif condition_name == "augmented_random":
                # 100% randomized
                for i, sample in enumerate(samples):
                    out = deepcopy(sample)
                    rand = randomized[i]
                    if rand is not None:
                        original_smiles = out["smiles"]
                        out["smiles"] = rand
                        if original_smiles in out.get("question", ""):
                            out["question"] = out["question"].replace(
                                original_smiles, rand
                            )
                        n_randomized += 1
                    output_samples.append(out)

            elif condition_name == "augmented_50_50":
                # 50% canonical, 50% randomized (coin flip per sample)
                # Use a deterministic RNG seeded per-task so the selection
                # is reproducible and independent of task processing order.
                task_rng = random.Random(args.seed + hash(task))
                flip_choices = [task_rng.random() < 0.5 for _ in range(n_samples)]
                for i, sample in enumerate(samples):
                    out = deepcopy(sample)
                    if flip_choices[i]:
                        rand = randomized[i]
                        if rand is not None:
                            original_smiles = out["smiles"]
                            out["smiles"] = rand
                            if original_smiles in out.get("question", ""):
                                out["question"] = out["question"].replace(
                                    original_smiles, rand
                                )
                            n_randomized += 1
                    output_samples.append(out)

            elif condition_name == "augmented_curriculum":
                # First half canonical, second half randomized
                midpoint = n_samples // 2
                for i, sample in enumerate(samples):
                    out = deepcopy(sample)
                    if i >= midpoint:
                        rand = randomized[i]
                        if rand is not None:
                            original_smiles = out["smiles"]
                            out["smiles"] = rand
                            if original_smiles in out.get("question", ""):
                                out["question"] = out["question"].replace(
                                    original_smiles, rand
                                )
                            n_randomized += 1
                    output_samples.append(out)

            save_task_data(output_samples, out_dir, task)
            condition_stats[task] = n_randomized
            logger.info(
                "    %s: %d samples (%d randomized, %d canonical)",
                task,
                len(output_samples),
                n_randomized,
                len(output_samples) - n_randomized,
            )

        total_stats[condition_name] = condition_stats

    # Verification pass
    logger.info("")
    logger.info("=== Verification ===")
    all_ok = True

    for condition_name in conditions:
        out_dir = output_base / condition_name
        for task in TASKS:
            path = out_dir / f"{task}.json"
            if not path.exists():
                logger.error("MISSING: %s", path)
                all_ok = False
                continue
            with open(path) as f:
                data = json.load(f)
            expected = len(all_task_data.get(task, []))
            if len(data) != expected:
                logger.error(
                    "COUNT MISMATCH: %s has %d samples (expected %d)",
                    path,
                    len(data),
                    expected,
                )
                all_ok = False

    # Verify fragment assembly integrity
    for condition_name in conditions:
        out_dir = output_base / condition_name
        fa_path = out_dir / "fragment_assembly.json"
        if not fa_path.exists():
            continue
        with open(fa_path) as f:
            fa_data = json.load(f)
        n_bad_separator = 0
        n_bad_parse = 0
        for sample in fa_data:
            smiles = sample["smiles"]
            if " . " not in smiles:
                n_bad_separator += 1
                continue
            parts = smiles.split(" . ")
            for part in parts:
                mol = Chem.MolFromSmiles(part)
                if mol is None:
                    n_bad_parse += 1
                    break
        if n_bad_separator > 0:
            logger.warning(
                "%s/fragment_assembly: %d samples missing ' . ' separator",
                condition_name,
                n_bad_separator,
            )
        if n_bad_parse > 0:
            logger.warning(
                "%s/fragment_assembly: %d samples with unparseable fragments",
                condition_name,
                n_bad_parse,
            )

    # Verify answers are unchanged
    for condition_name in conditions:
        out_dir = output_base / condition_name
        for task in TASKS:
            path = out_dir / f"{task}.json"
            if not path.exists():
                continue
            with open(path) as f:
                aug_data = json.load(f)
            orig_data = all_task_data.get(task, [])
            n_answer_changed = 0
            for orig, aug in zip(orig_data, aug_data):
                if orig["answer"] != aug["answer"]:
                    n_answer_changed += 1
            if n_answer_changed > 0:
                logger.error(
                    "ANSWER CHANGED: %s/%s has %d samples with modified answers",
                    condition_name,
                    task,
                    n_answer_changed,
                )
                all_ok = False

    # Verify that randomized conditions actually changed SMILES
    for condition_name in ["augmented_random", "augmented_50_50", "augmented_curriculum"]:
        out_dir = output_base / condition_name
        for task in TASKS:
            path = out_dir / f"{task}.json"
            if not path.exists():
                continue
            with open(path) as f:
                aug_data = json.load(f)
            orig_data = all_task_data.get(task, [])
            n_changed = sum(
                1
                for orig, aug in zip(orig_data, aug_data)
                if orig["smiles"] != aug["smiles"]
            )
            logger.info(
                "  %s/%s: %d/%d SMILES changed",
                condition_name,
                task,
                n_changed,
                len(aug_data),
            )

    if all_ok:
        logger.info("All verification checks passed.")
    else:
        logger.error("Some verification checks FAILED. Review output above.")
        sys.exit(1)

    # Print summary table
    print()
    print("=" * 70)
    print("SMILES Augmentation Data Preparation Summary")
    print("=" * 70)
    print(f"{'Condition':<25s} {'Task':<25s} {'Randomized':>10s} {'Total':>8s}")
    print("-" * 70)
    for condition_name, stats in total_stats.items():
        for task in TASKS:
            if task not in stats:
                continue
            total = len(all_task_data.get(task, []))
            print(
                f"{condition_name:<25s} {task:<25s} {stats[task]:>10d} {total:>8d}"
            )
        print("-" * 70)
    print()


if __name__ == "__main__":
    main()
