from __future__ import annotations
import json
import logging
import random
from pathlib import Path
from typing import Iterator
from rdkit import Chem
from datasets import load_dataset
from clarimol.data.pruning import PruningConfig, prune_and_sort
from clarimol.data.parsing import Parsing
from clarimol.data.sample import Sample
from clarimol.data.tasks import get_task

logger = logging.getLogger(__name__)


# Load ZINC250K - https://huggingface.co/datasets/yairschiff/zinc250k
def load_zinc250k(split: str = "train") -> list[str]:
    ds = load_dataset("yairschiff/zinc250k", split=split)
    return [row["smiles"] for row in ds]


def load_smiles_file(path: str | Path) -> list[str]:
    """Load SMILES from text file (one per line) or JSON"""
    p = Path(path)
    if p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [r["smiles"] if isinstance(r, dict) else r for r in data]
        raise ValueError(f"Unexpected JSON structure in {p}")
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


# Sample Generation
def _generate_for_molecule(
    smiles: str,
    tasks: list[Parsing],
    rng: random.Random,
) -> Iterator[Sample]:
    """Generate samples across all tasks for a molecule"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    for task in tasks:
        try:
            for sample in task.generate(smiles, mol, rng):
                yield sample
        except Exception as e:
            logger.debug("Task %s failed on %s: %s", task.name, smiles, e)


def build_dataset(
    smiles_list: list[str],
    task_names: list[str] | None = None,
    pruning: PruningConfig | None = None,
    seed: int = 42,
    max_molecules: int | None = None,
) -> dict[str, list[Sample]]:
    """
    Build the CLARIMOL dataset from a list of SMILES.

    :param: smiles_list: Raw SMILES strings (e.g. from ZINC250K)
    :param: task_names: Which parsing tasks to generate. None → all five
    :param: pruning: Pruning/curriculum config. None → defaults (50K middle, curriculum)
    :param: seed: Random seed for reproducibility
    :param: max_molecules: Cap on molecules to process (for testing)

    :return: Dictionary of task name -> list of pruned samples
    """
    if not task_names:
        task_names = [
            "functional_group",
            "ring_counting",
            "chain_length",
            "canonicalization",
            "fragment_assembly",
        ]
    tasks = [get_task(name) for name in task_names]
    rng = random.Random(seed)
    if max_molecules is not None:
        smiles_list = smiles_list[:max_molecules]
    # Collect raw samples per task
    raw: dict[str, list[Sample]] = {t.name: [] for t in tasks}
    n_total = len(smiles_list)
    for i, smi in enumerate(smiles_list):
        if (i + 1) % 10_000 == 0:
            logger.info("Processing molecule %d / %d", i + 1, n_total)
        for sample in _generate_for_molecule(smi, tasks, rng):
            raw[sample.task].append(sample)
    logger.info("Raw samples per task: %s", {k: len(v) for k, v in raw.items()})
    # Prune + sort each task independently
    if pruning is None:
        pruning = PruningConfig()
    result: dict[str, list[Sample]] = {}
    for task_name, samples in raw.items():
        result[task_name] = prune_and_sort(samples, pruning)
        logger.info(
            "Task %s: %d → %d samples after pruning",
            task_name,
            len(samples),
            len(result[task_name]),
        )
    return result


# Serialization
def save_dataset(data: dict[str, list[Sample]], out_dir: str | Path) -> None:
    """Save dataset as JSON files (one per task)"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for task_name, samples in data.items():
        path = out / f"{task_name}.json"
        records = [
            {
                "smiles": s.smiles,
                "task": s.task,
                "question": s.question,
                "answer": s.answer,
                "difficulty": s.difficulty,
                "metadata": s.metadata,
            }
            for s in samples
        ]
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        logger.info("Saved %d samples to %s", len(records), path)


def load_dataset_from_disk(data_dir: str | Path) -> dict[str, list[Sample]]:
    """Load a previously saved dataset"""
    d = Path(data_dir)
    result: dict[str, list[Sample]] = {}
    for path in sorted(d.glob("*.json")):
        task_name = path.stem
        with open(path) as f:
            records = json.load(f)
        result[task_name] = [
            Sample(
                smiles=r["smiles"],
                task=r["task"],
                question=r["question"],
                answer=r["answer"],
                difficulty=r.get("difficulty", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in records
        ]
    return result
