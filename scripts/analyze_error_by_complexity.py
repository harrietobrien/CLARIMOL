"""
Stratify per-sample model correctness by molecular complexity metrics computed via RDKit.

For each model × task, per-sample predictions are loaded from predictions.jsonl.
Five RDKit-derived complexity metrics are computed per SMILES:
    - smiles_length   : character count of the raw SMILES string
    - heavy_atoms     : number of heavy (non-hydrogen) atoms
    - num_rings       : total ring count (RingInfo.NumRings)
    - rot_bonds       : number of rotatable bonds (RDKit definition)
    - mol_weight      : molecular weight (Da)

Each metric is binned into quartiles (Q1–Q4) across the combined test population
for that task. Accuracy within each quartile is computed and written to a JSON
summary. A companion figure script reads that summary to produce the plot.

Output
------
output/complexity_analysis/complexity_summary.json
    Nested dict: {model -> task -> metric -> quartile_label -> {"accuracy", "n"}}

output/complexity_analysis/complexity_distributions.json
    Per-task metric distributions and quartile boundaries (for figure annotation).

Usage
-----
    python scripts/analyze_error_by_complexity.py \\
        --output-dir output/complexity_analysis
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS: dict[str, str] = {
    "LLaMA-8B": "llama-8b",
    "Mistral-7B": "mistral-7b",
    "OLMo-7B": "olmo-7b",
    "Qwen-7B": "qwen-7b",
    "Qwen3-8B": "qwen3-8b",
}

TASKS: list[str] = [
    "functional_group",
    "ring_counting",
    "chain_length",
    "canonicalization",
    "fragment_assembly",
]

METRICS: list[str] = [
    "smiles_length",
    "heavy_atoms",
    "num_rings",
    "rot_bonds",
    "mol_weight",
]

METRIC_LABELS: dict[str, str] = {
    "smiles_length": "SMILES Length",
    "heavy_atoms": "Heavy Atoms",
    "num_rings": "Ring Count",
    "rot_bonds": "Rotatable Bonds",
    "mol_weight": "Mol. Weight (Da)",
}

QUARTILE_LABELS: list[str] = ["Q1", "Q2", "Q3", "Q4"]


def rdkit_metrics(smiles: str) -> dict[str, float] | None:
    """Compute RDKit-derived complexity metrics for a single SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES string.  For fragment_assembly the input SMILES may contain
        attachment-point tokens ([1*], [2*]) or a fragment separator (' . ');
        the canonical molecule is inferred from the largest connected component
        after parsing.

    Returns
    -------
    dict[str, float] or None
        Mapping of metric name to numeric value, or None if RDKit parsing fails.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    # Strip attachment tokens and take the largest fragment for multi-component
    # SMILES (fragment_assembly inputs).
    cleaned = smiles.replace("[1*]", "C").replace("[2*]", "C")
    parts = cleaned.split(" . ")
    mol = None
    for part in sorted(parts, key=len, reverse=True):
        mol = Chem.MolFromSmiles(part)
        if mol is not None:
            break

    if mol is None:
        return None

    ring_info = mol.GetRingInfo()
    return {
        "smiles_length": float(len(smiles)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "num_rings": float(ring_info.NumRings()),
        "rot_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "mol_weight": float(Descriptors.MolWt(mol)),
    }


def load_predictions(model_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load per-sample predictions from a model's predictions.jsonl file.

    Parameters
    ----------
    model_dir : Path
        Directory containing ``predictions.jsonl``.

    Returns
    -------
    dict[str, list[dict]]
        Mapping of task name to list of prediction records.  Each record
        contains at minimum ``smiles`` (str) and ``correct`` (bool).
    """
    jsonl_path = model_dir / "predictions.jsonl"
    if not jsonl_path.exists():
        log.warning("predictions.jsonl not found: %s", jsonl_path)
        return {}

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(jsonl_path) as fh:
        for line in fh:
            record = json.loads(line)
            by_task[record["task"]].append(record)

    total = sum(len(v) for v in by_task.values())
    log.info("Loaded %d predictions from %s", total, jsonl_path)
    return dict(by_task)


def assign_quartiles(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Assign each value to a quartile bin (0-indexed: 0=Q1, 3=Q4).

    Boundaries are computed on the full array (not per-split) so that quartile
    definitions are consistent across models for the same task.

    Parameters
    ----------
    values : np.ndarray
        1-D array of numeric values.

    Returns
    -------
    quartile_indices : np.ndarray
        Integer array of quartile assignments (0–3).
    boundaries : list[float]
        The three boundary values [25th, 50th, 75th percentiles].
    """
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    bins = np.digitize(values, [q25, q50, q75])  # 0,1,2,3
    return bins, [float(q25), float(q50), float(q75)]


def compute_complexity_metrics_for_task(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, float] | None], list[bool]]:
    """Compute RDKit metrics for every record in a task's prediction list.

    Parameters
    ----------
    records : list[dict]
        Prediction records, each containing ``smiles`` and ``correct``.

    Returns
    -------
    metrics_list : list[dict | None]
        Per-record RDKit metrics (None on parse failure).
    correct_list : list[bool]
        Per-record correctness flag.
    """
    metrics_list: list[dict[str, float] | None] = []
    correct_list: list[bool] = []
    n_fail = 0
    for rec in records:
        m = rdkit_metrics(rec["smiles"])
        if m is None:
            n_fail += 1
        metrics_list.append(m)
        correct_list.append(bool(rec["correct"]))
    if n_fail:
        log.warning("%d / %d SMILES failed RDKit parsing.", n_fail, len(records))
    return metrics_list, correct_list


def accuracy_by_quartile(
    metrics_list: list[dict[str, float] | None],
    correct_list: list[bool],
    metric: str,
    boundaries: list[float],
) -> dict[str, dict[str, float | int]]:
    """Compute accuracy within each quartile for a single metric.

    Parameters
    ----------
    metrics_list : list[dict | None]
        Per-sample metric dicts (None entries are skipped).
    correct_list : list[bool]
        Per-sample correctness flags.
    metric : str
        The metric key to stratify on.
    boundaries : list[float]
        Three boundary values defining four quartile bins.

    Returns
    -------
    dict[str, dict]
        Mapping of quartile label (Q1–Q4) to {"accuracy": float, "n": int}.
    """
    bins: dict[str, list[bool]] = {q: [] for q in QUARTILE_LABELS}
    q25, q50, q75 = boundaries
    for m, c in zip(metrics_list, correct_list):
        if m is None:
            continue
        v = m[metric]
        if v <= q25:
            bins["Q1"].append(c)
        elif v <= q50:
            bins["Q2"].append(c)
        elif v <= q75:
            bins["Q3"].append(c)
        else:
            bins["Q4"].append(c)

    result: dict[str, dict[str, float | int]] = {}
    for label, vals in bins.items():
        n = len(vals)
        acc = float(np.mean(vals)) if n > 0 else float("nan")
        result[label] = {"accuracy": acc, "n": n}
    return result


def build_task_metric_boundaries(
    all_model_records: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute quartile boundaries from the union of all models' SMILES for each task.

    Using the union ensures boundaries are consistent across models — the same
    molecule always falls in the same quartile regardless of which model is
    being evaluated.

    Parameters
    ----------
    all_model_records : dict[str, dict[str, list[dict]]]
        {model_label -> {task -> records}}.

    Returns
    -------
    dict[str, dict[str, dict]]
        {task -> {metric -> {"boundaries": list[float], "all_values": list[float]}}}.
    """
    log.info("Computing quartile boundaries from union of all model SMILES.")
    # Collect unique SMILES per task (predictions across models overlap heavily;
    # use the first model that has the task to avoid redundant RDKit calls).
    task_smiles: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for _label, by_task in all_model_records.items():
        for task, records in by_task.items():
            for rec in records:
                s = rec["smiles"]
                if s not in seen[task]:
                    seen[task].add(s)
                    task_smiles[task].append(s)

    boundaries: dict[str, dict[str, dict[str, Any]]] = {}
    for task, smiles_list in task_smiles.items():
        log.info("Computing metrics for task=%s (%d unique SMILES).", task, len(smiles_list))
        metric_values: dict[str, list[float]] = defaultdict(list)
        for s in smiles_list:
            m = rdkit_metrics(s)
            if m is None:
                continue
            for k, v in m.items():
                metric_values[k].append(v)
        boundaries[task] = {}
        for metric, vals in metric_values.items():
            arr = np.array(vals)
            _, bnd = assign_quartiles(arr)
            boundaries[task][metric] = {
                "boundaries": bnd,
                "all_values": vals,
            }
    return boundaries


def run_analysis(
    output_dir: Path,
    seed: str = "seed_42",
) -> None:
    """Execute the full complexity stratification analysis.

    Parameters
    ----------
    output_dir : Path
        Directory for JSON output files.
    seed : str
        Seed subdirectory name within each model's multi_seed directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_root = REPO_ROOT / "output" / "multi_seed"

    # Load all model predictions.
    all_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for label, model_dir_name in MODELS.items():
        seed_dir = predictions_root / model_dir_name / seed
        if not seed_dir.exists():
            log.warning("Seed directory not found, skipping model %s: %s", label, seed_dir)
            continue
        by_task = load_predictions(seed_dir)
        if by_task:
            all_records[label] = by_task

    if not all_records:
        log.error("No prediction data loaded. Exiting.")
        sys.exit(1)

    # Compute quartile boundaries from the union of all SMILES.
    task_boundaries = build_task_metric_boundaries(all_records)

    # Save distribution data for figure annotation.
    distributions: dict[str, Any] = {}
    for task, metric_data in task_boundaries.items():
        distributions[task] = {}
        for metric, info in metric_data.items():
            distributions[task][metric] = {
                "boundaries": info["boundaries"],
                "label": METRIC_LABELS[metric],
                "n_unique": len(info["all_values"]),
                "mean": float(np.mean(info["all_values"])),
                "std": float(np.std(info["all_values"])),
            }

    dist_path = output_dir / "complexity_distributions.json"
    with open(dist_path, "w") as fh:
        json.dump(distributions, fh, indent=2)
    log.info("Distribution data written to %s.", dist_path)

    # Compute per-model, per-task, per-metric accuracy by quartile.
    summary: dict[str, Any] = {}

    for label, by_task in all_records.items():
        summary[label] = {}
        log.info("Stratifying model: %s", label)

        for task in TASKS:
            if task not in by_task:
                log.warning("Task %s not found for model %s.", task, label)
                continue
            if task not in task_boundaries:
                log.warning("No boundary data for task %s.", task)
                continue

            records = by_task[task]
            metrics_list, correct_list = compute_complexity_metrics_for_task(records)

            summary[label][task] = {}
            for metric in METRICS:
                if metric not in task_boundaries[task]:
                    continue
                bnd = task_boundaries[task][metric]["boundaries"]
                q_result = accuracy_by_quartile(metrics_list, correct_list, metric, bnd)
                summary[label][task][metric] = q_result

    summary_path = output_dir / "complexity_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Complexity summary written to %s.", summary_path)

    # Print a brief tabular overview to stdout.
    _print_overview(summary)


def _print_overview(summary: dict[str, Any]) -> None:
    """Print a compact accuracy-by-quartile table for the primary metric (heavy_atoms).

    Parameters
    ----------
    summary : dict
        Nested summary dict produced by run_analysis.
    """
    metric = "heavy_atoms"
    print(f"\nAccuracy by quartile — metric: {METRIC_LABELS[metric]}")
    header = f"{'Model':<14}" + "".join(f"  {t[:10]:<10}" for t in TASKS)
    print(header)
    print("-" * len(header))

    for label in MODELS:
        if label not in summary:
            continue
        row = f"{label:<14}"
        for task in TASKS:
            if task not in summary[label] or metric not in summary[label][task]:
                row += f"  {'—':<10}"
                continue
            q_data = summary[label][task][metric]
            accs = [
                q_data[q]["accuracy"]
                for q in QUARTILE_LABELS
                if not np.isnan(q_data[q]["accuracy"])
            ]
            if not accs:
                row += f"  {'—':<10}"
                continue
            # Show Q1 and Q4 accuracy to highlight the gradient.
            q1 = q_data["Q1"]["accuracy"]
            q4 = q_data["Q4"]["accuracy"]
            row += f"  {q1:.2f}→{q4:.2f}  "
        print(row)
    print()


def main() -> None:
    """Entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description="Stratify model accuracy by molecular complexity metrics."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "complexity_analysis",
        help="Directory for JSON output files.",
    )
    parser.add_argument(
        "--seed",
        default="seed_42",
        help="Seed subdirectory to load from each model directory.",
    )
    args = parser.parse_args()
    run_analysis(output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
