#!/usr/bin/env python3
"""
Analyze cross-task transfer results and build a 5x5 transfer matrix.

Reads output/cross_task_transfer/holdout_{task}/results.json for each
held-out task, then builds a matrix where:
  rows  = training configuration (which task was held out)
  cols  = evaluation task
  diagonal = held-out task accuracy (true transfer performance)
  off-diagonal = trained tasks (in-distribution performance)

Outputs:
  output/cross_task_transfer/transfer_matrix.json  — machine-readable matrix
  stdout — formatted table with transfer diagonal highlighted

Usage:
    python scripts/analyze_cross_task_transfer.py [--base-dir <path>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


ALL_TASKS: list[str] = [
    "functional_group",
    "ring_counting",
    "chain_length",
    "canonicalization",
    "fragment_assembly",
]

SHORT: dict[str, str] = {
    "functional_group": "func_grp",
    "ring_counting":    "ring_cnt",
    "chain_length":     "chain_len",
    "canonicalization": "canon",
    "fragment_assembly": "frag_asm",
}


def load_results(base: pathlib.Path) -> dict[str, dict[str, float]]:
    """Load per-task accuracy for each holdout run.

    Returns a dict keyed by held-out task name. Each value is a dict
    mapping task name to accuracy (or NaN if data is missing).
    """
    matrix: dict[str, dict[str, float]] = {}
    for holdout in ALL_TASKS:
        rpath = base / f"holdout_{holdout}" / "results.json"
        row: dict[str, float] = {}
        if rpath.exists():
            with open(rpath) as fh:
                data: dict = json.load(fh)
            for task in ALL_TASKS:
                row[task] = data.get(task, {}).get("accuracy", float("nan"))
        else:
            for task in ALL_TASKS:
                row[task] = float("nan")
        matrix[holdout] = row
    return matrix


def print_table(matrix: dict[str, dict[str, float]]) -> None:
    """Print a formatted transfer matrix table to stdout."""
    col_w = 10
    row_label_w = 14

    header = f"  {'holdout':>{row_label_w}s}"
    for task in ALL_TASKS:
        header += f"  {SHORT[task]:>{col_w}s}"
    print(header)
    print("  " + "-" * (row_label_w + (col_w + 2) * len(ALL_TASKS)))

    for holdout in ALL_TASKS:
        row = matrix[holdout]
        line = f"  {SHORT[holdout]:>{row_label_w}s}"
        for task in ALL_TASKS:
            acc = row[task]
            if acc != acc:  # NaN check
                cell = f"{'N/A':>{col_w}s}"
            else:
                cell = f"{acc:>{col_w}.4f}"
            # Mark the transfer (held-out) cell with an asterisk.
            marker = "*" if task == holdout else " "
            line += f"  {cell}{marker}"
        print(line)

    print()
    print("  * = held-out task (true zero-transfer performance)")
    print()


def compute_summary(matrix: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute scalar summary statistics from the transfer matrix."""
    transfer_accs: list[float] = []
    indist_accs: list[float] = []

    for holdout in ALL_TASKS:
        row = matrix[holdout]
        for task in ALL_TASKS:
            acc = row[task]
            if acc != acc:
                continue
            if task == holdout:
                transfer_accs.append(acc)
            else:
                indist_accs.append(acc)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "mean_transfer": mean(transfer_accs),
        "mean_in_distribution": mean(indist_accs),
        "transfer_gap": mean(indist_accs) - mean(transfer_accs),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default="output/cross_task_transfer",
        help="Directory containing holdout_<task>/ subdirectories.",
    )
    args = parser.parse_args(argv)

    base = pathlib.Path(args.base_dir)
    if not base.exists():
        print(f"ERROR: base directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    matrix = load_results(base)

    missing = [h for h in ALL_TASKS if not (base / f"holdout_{h}" / "results.json").exists()]
    if missing:
        print(f"Warning: results not yet available for: {', '.join(missing)}")
        print()

    print("=== Cross-Task Transfer Matrix ===")
    print("Rows = training config (held-out task), Cols = evaluation task")
    print()
    print_table(matrix)

    summary = compute_summary(matrix)
    print("=== Summary ===")
    print(f"  Mean transfer (diagonal)     : {summary['mean_transfer']:.4f}")
    print(f"  Mean in-distribution (off-diag): {summary['mean_in_distribution']:.4f}")
    print(f"  Transfer gap (in-dist - transfer): {summary['transfer_gap']:.4f}")
    print()

    out_path = base / "transfer_matrix.json"
    payload = {
        "matrix": matrix,
        "summary": summary,
        "tasks": ALL_TASKS,
        "note": (
            "diagonal = held-out task accuracy (transfer performance); "
            "off-diagonal = in-distribution performance on trained tasks"
        ),
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Matrix saved to {out_path}")


if __name__ == "__main__":
    main()
