"""
Comparison table: few-shot frontier LLMs vs fine-tuned models on SMILES parsing tasks.

Loads results from output/few_shot/{model}/{n_shot}/results.json (frontier, few-shot)
and one or more fine-tuned results JSON files, then prints a formatted accuracy table.

Usage:
    python scripts/compare_fewshot_vs_finetuned.py
    python scripts/compare_fewshot_vs_finetuned.py --finetuned output/baselines/zero_shot_llama-3.1-8b-instruct/results.json
    python scripts/compare_fewshot_vs_finetuned.py --few-shot-dir output/few_shot --finetuned results_a.json results_b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

TASKS = [
    "functional_group",
    "ring_counting",
    "chain_length",
    "canonicalization",
    "fragment_assembly",
]

TASK_ABBREV = {
    "functional_group": "FG",
    "ring_counting": "Ring",
    "chain_length": "Chain",
    "canonicalization": "Canon.",
    "fragment_assembly": "Frag.",
}


def _load_results(path: Path) -> dict[str, float]:
    """
    Load a results JSON and return a flat dict mapping task_name → accuracy.

    Accepts both the fine-tuned format (task → {accuracy: float, ...}) and
    the few-shot format (same schema, produced by eval_few_shot_frontier.py).
    """
    with open(path) as f:
        raw = json.load(f)
    out: dict[str, float] = {}
    for task, data in raw.items():
        if isinstance(data, dict) and "accuracy" in data:
            out[task] = float(data["accuracy"])
        elif isinstance(data, (int, float)):
            out[task] = float(data)
    return out


def _discover_few_shot_results(few_shot_dir: Path) -> list[tuple[str, int, dict[str, float]]]:
    """
    Walk the few-shot output directory tree and collect all results.

    Expected structure: {few_shot_dir}/{model_name}/{n_shot}/results.json

    Returns list of (model_display_name, n_shot, accuracy_dict) tuples,
    sorted by model name then n_shot ascending.
    """
    entries: list[tuple[str, int, dict[str, float]]] = []
    if not few_shot_dir.exists():
        return entries
    for model_dir in sorted(few_shot_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for shot_dir in sorted(model_dir.iterdir()):
            if not shot_dir.is_dir():
                continue
            results_file = shot_dir / "results.json"
            if not results_file.exists():
                continue
            try:
                n_shot = int(shot_dir.name)
            except ValueError:
                continue
            accs = _load_results(results_file)
            if accs:
                entries.append((model_dir.name, n_shot, accs))
    # Sort: model name first, then n_shot
    entries.sort(key=lambda e: (e[0], e[1]))
    return entries


def _format_row(
    label: str,
    accs: dict[str, float],
    tasks: list[str],
    col_width: int,
    label_width: int,
) -> str:
    """Format a single table row."""
    parts = [label.ljust(label_width)]
    task_accs = []
    for task in tasks:
        if task in accs:
            cell = f"{accs[task]:.3f}"
        else:
            cell = "  —  "
        task_accs.append(cell.center(col_width))
    avg_accs = [accs[t] for t in tasks if t in accs]
    if avg_accs:
        avg_cell = f"{sum(avg_accs) / len(avg_accs):.3f}".center(col_width)
    else:
        avg_cell = "  —  ".center(col_width)
    parts.extend(task_accs)
    parts.append(avg_cell)
    return " | ".join(parts)


def print_table(
    few_shot_entries: list[tuple[str, int, dict[str, float]]],
    finetuned_entries: list[tuple[str, dict[str, float]]],
    tasks: list[str],
) -> None:
    """Print a formatted accuracy comparison table to stdout."""
    col_width = 8
    label_width = 28

    # Header
    header_abbrevs = [TASK_ABBREV.get(t, t).center(col_width) for t in tasks]
    header = " | ".join(
        ["Model".ljust(label_width)] + header_abbrevs + ["Avg.".center(col_width)]
    )
    sep = "-" * len(header)

    print()
    print("Accuracy Comparison: Few-Shot Frontier vs Fine-Tuned Models")
    print(sep)
    print(header)
    print(sep)

    if few_shot_entries:
        print("Few-shot (frontier models, no fine-tuning):")
        for model_name, n_shot, accs in few_shot_entries:
            label = f"  {model_name} [{n_shot}-shot]"
            print(_format_row(label, accs, tasks, col_width, label_width))
        print(sep)

    if finetuned_entries:
        print("Fine-tuned models:")
        for label, accs in finetuned_entries:
            row_label = f"  {label}"
            print(_format_row(row_label, accs, tasks, col_width, label_width))
        print(sep)

    print()

    # Per-task best highlighting
    print("Per-task best accuracy:")
    all_entries: list[tuple[str, dict[str, float]]] = []
    for model_name, n_shot, accs in few_shot_entries:
        all_entries.append((f"{model_name} [{n_shot}-shot]", accs))
    all_entries.extend(finetuned_entries)

    for task in tasks:
        task_scores = [
            (label, accs[task])
            for label, accs in all_entries
            if task in accs
        ]
        if not task_scores:
            continue
        best_label, best_score = max(task_scores, key=lambda x: x[1])
        print(f"  {TASK_ABBREV.get(task, task):<10} {best_score:.3f}  ({best_label})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print accuracy comparison table for few-shot frontier vs fine-tuned models."
    )
    parser.add_argument(
        "--few-shot-dir",
        type=Path,
        default=_REPO_ROOT / "output" / "few_shot",
        help="Root directory of few-shot results (default: output/few_shot).",
    )
    parser.add_argument(
        "--finetuned",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Paths to fine-tuned results JSON files. Accepts both the flat "
            "{task: {accuracy: ...}} format and the legacy {task: accuracy} format. "
            "If not specified, auto-discovers from output/baselines/ and "
            "output/legacy_results/."
        ),
    )
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="Comma-separated list of tasks to include in the table.",
    )
    parser.add_argument(
        "--label-prefix",
        default=None,
        help="Optional prefix for fine-tuned result labels (e.g., model family name).",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # Discover few-shot results
    few_shot_entries = _discover_few_shot_results(args.few_shot_dir)
    if not few_shot_entries:
        print(
            f"No few-shot results found under {args.few_shot_dir}. "
            "Run eval_few_shot_frontier.py first.",
            file=sys.stderr,
        )

    # Collect fine-tuned results
    finetuned_entries: list[tuple[str, dict[str, float]]] = []

    if args.finetuned is not None:
        ft_paths = args.finetuned
    else:
        # Auto-discover from standard output locations
        ft_paths = []
        for candidate_dir in [
            _REPO_ROOT / "output" / "baselines",
            _REPO_ROOT / "output" / "legacy_results",
        ]:
            if candidate_dir.is_dir():
                for p in sorted(candidate_dir.rglob("results.json")):
                    ft_paths.append(p)
                for p in sorted(candidate_dir.glob("*.json")):
                    if "checkpoint" not in str(p):
                        ft_paths.append(p)

    seen_paths: set[Path] = set()
    for path in ft_paths:
        path = Path(path).resolve()
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        accs = _load_results(path)
        if not any(t in accs for t in tasks):
            continue
        # Derive a short label from the path
        try:
            rel = path.relative_to(_REPO_ROOT)
            label_parts = list(rel.parts)
            # Drop "output" prefix and "results.json" suffix for conciseness
            if label_parts and label_parts[0] == "output":
                label_parts = label_parts[1:]
            if label_parts and label_parts[-1].endswith(".json"):
                label_parts[-1] = label_parts[-1].replace(".json", "")
            label = "/".join(label_parts)
        except ValueError:
            label = path.stem
        if args.label_prefix:
            label = f"{args.label_prefix}/{label}"
        finetuned_entries.append((label, accs))

    if not few_shot_entries and not finetuned_entries:
        print("No results found. Check paths and re-run evaluations.", file=sys.stderr)
        sys.exit(1)

    print_table(few_shot_entries, finetuned_entries, tasks)

    # Also print a raw CSV for easy import into spreadsheets
    print("CSV output:")
    header_cols = ["Model", "Type"] + [TASK_ABBREV.get(t, t) for t in tasks] + ["Avg"]
    print(",".join(header_cols))

    all_rows: list[tuple[str, str, dict[str, float]]] = []
    for model_name, n_shot, accs in few_shot_entries:
        all_rows.append((f"{model_name} [{n_shot}-shot]", "few-shot", accs))
    for label, accs in finetuned_entries:
        all_rows.append((label, "fine-tuned", accs))

    for label, row_type, accs in all_rows:
        task_vals = []
        for t in tasks:
            task_vals.append(f"{accs[t]:.4f}" if t in accs else "")
        valid = [accs[t] for t in tasks if t in accs]
        avg = f"{sum(valid)/len(valid):.4f}" if valid else ""
        print(",".join([f'"{label}"', row_type] + task_vals + [avg]))
    print()


if __name__ == "__main__":
    main()
