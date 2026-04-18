"""
Plot all CLARIMOL results: training curves, parsing accuracy, downstream metrics.
Usage: python scripts/plot_results.py
"""
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

OUTPUT_DIR = Path("output")
PLOT_DIR = Path("output/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Colors
BASELINE_COLOR = "#EF476F"
TRAINED_COLOR = "#06D6A0"
P100_COLOR = "#00B4D8"
ADA_COLOR = "#FFD166"
TASK_COLORS = ["#00B4D8", "#06D6A0", "#FFD166", "#EF476F", "#AB47BC"]


def parse_training_log(log_path: Path) -> list[dict]:
    """Extract training metrics from log file."""
    metrics = []
    with open(log_path) as f:
        for line in f:
            if line.startswith("{") and "'loss'" in line:
                # Convert single quotes to double quotes for JSON
                entry = line.strip().replace("'", '"')
                try:
                    metrics.append(json.loads(entry))
                except json.JSONDecodeError:
                    continue
    return metrics


def plot_training_curves(logs: dict[str, list[dict]], filename: str = "training_curves.png"):
    """Plot loss and accuracy curves for one or more training runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves", fontsize=16, fontweight="bold")

    colors = [P100_COLOR, ADA_COLOR, "#AB47BC", "#EF476F"]
    for i, (name, metrics) in enumerate(logs.items()):
        # Skip final summary entry
        steps = [m for m in metrics if "epoch" in m and "train_runtime" not in m]
        epochs = [float(s["epoch"]) for s in steps]
        losses = [float(s["loss"]) for s in steps]
        accs = [float(s["mean_token_accuracy"]) * 100 for s in steps]
        color = colors[i % len(colors)]

        ax1.plot(epochs, losses, marker="o", markersize=4, label=name, color=color, linewidth=2)
        ax2.plot(epochs, accs, marker="o", markersize=4, label=name, color=color, linewidth=2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Token Accuracy (%)")
    ax2.set_title("Mean Token Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOT_DIR / filename}")


def plot_parsing_comparison(baseline_path: Path, trained_path: Path,
                             filename: str = "parsing_comparison.png"):
    """Bar chart comparing baseline vs pre-trained parsing accuracy."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(trained_path) as f:
        trained = json.load(f)

    tasks = list(baseline.keys())
    task_labels = [t.replace("_", " ").title() for t in tasks]
    baseline_accs = [baseline[t]["accuracy"] * 100 for t in tasks]
    trained_accs = [trained[t]["accuracy"] * 100 for t in tasks]

    x = range(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width / 2 for i in x], baseline_accs, width,
                   label="Baseline (Qwen 7B)", color=BASELINE_COLOR, alpha=0.85)
    bars2 = ax.bar([i + width / 2 for i in x], trained_accs, width,
                   label="Pre-trained", color=TRAINED_COLOR, alpha=0.85)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("SMILES Parsing Task Accuracy: Baseline vs Pre-trained", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=15, ha="right")
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOT_DIR / filename}")


def plot_improvement_chart(baseline_path: Path, trained_path: Path,
                            filename: str = "improvement.png"):
    """Horizontal bar chart showing improvement per task."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(trained_path) as f:
        trained = json.load(f)

    tasks = list(baseline.keys())
    task_labels = [t.replace("_", " ").title() for t in tasks]
    improvements = [(trained[t]["accuracy"] - baseline[t]["accuracy"]) * 100 for t in tasks]

    # Sort by improvement
    sorted_pairs = sorted(zip(task_labels, improvements), key=lambda p: p[1])
    labels, imps = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [TRAINED_COLOR if v > 0 else BASELINE_COLOR for v in imps]
    bars = ax.barh(labels, imps, color=colors, alpha=0.85)

    for bar, val in zip(bars, imps):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"+{val:.1f}%", ha="left", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Accuracy Improvement (percentage points)", fontsize=12)
    ax.set_title("Improvement from SMILES Parsing Pre-training", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.axvline(x=0, color="gray", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOT_DIR / filename}")


def plot_downstream_results(result_files: dict[str, Path],
                             filename: str = "downstream_results.png"):
    """Plot downstream task metrics (exact match, BLEU, validity, fingerprint similarity)."""
    results = {}
    for task, path in result_files.items():
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                # Handle nested structure
                if task in data:
                    results[task] = data[task]
                else:
                    results[task] = data

    if not results:
        print("No downstream results found, skipping plot")
        return

    metrics = ["exact_match", "bleu", "validity", "morgan_fps"]
    metric_labels = ["Exact Match", "BLEU", "Validity", "Morgan FPS"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    fig.suptitle("Downstream Task Performance", fontsize=16, fontweight="bold")

    tasks = list(results.keys())
    task_labels = [t.replace("_", " ").title() for t in tasks]

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [results[t].get(metric, 0) * 100 for t in tasks]
        bars = ax.bar(range(len(tasks)), values, color=TASK_COLORS[:len(tasks)], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title(label, fontsize=12)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(task_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOT_DIR / filename}")


def plot_cod_comparison(zinc_path: Path, cod_path: Path,
                        filename: str = "cod_comparison.png"):
    """Compare model performance on ZINC vs COD molecules."""
    if not zinc_path.exists() or not cod_path.exists():
        print("Missing ZINC or COD results, skipping COD comparison plot")
        return

    with open(zinc_path) as f:
        zinc = json.load(f)
    with open(cod_path) as f:
        cod = json.load(f)

    # Find common tasks
    tasks = [t for t in zinc if t in cod]
    task_labels = [t.replace("_", " ").title() for t in tasks]
    zinc_accs = [zinc[t]["accuracy"] * 100 for t in tasks]
    cod_accs = [cod[t]["accuracy"] * 100 for t in tasks]

    x = range(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width / 2 for i in x], zinc_accs, width,
           label="ZINC250K (drug-like)", color=P100_COLOR, alpha=0.85)
    ax.bar([i + width / 2 for i in x], cod_accs, width,
           label="COD (crystal)", color=ADA_COLOR, alpha=0.85)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Generalization: ZINC250K vs COD Crystal Molecules", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=15, ha="right")
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOT_DIR / filename}")


def main():
    print("=" * 60)
    print("CLARIMOL Results Plotting")
    print("=" * 60)

    # 1. Training curves
    print("\n--- Training Curves ---")
    logs = {}
    p100_log = OUTPUT_DIR / "train_p100_fast.log"
    if p100_log.exists():
        metrics = parse_training_log(p100_log)
        if metrics:
            logs["P100 (LLaMA 8B)"] = metrics

    # Search for Ada training logs
    for log_file in sorted(OUTPUT_DIR.glob("duke_7b_*.log")):
        metrics = parse_training_log(log_file)
        if metrics and len(metrics) > 3:  # Skip failed runs
            # Check if loss is stable (not diverged)
            final = metrics[-1]
            if "train_loss" in final and float(final.get("train_loss", 99)) < 1.0:
                logs[f"Ada (Qwen 7B) - {log_file.stem}"] = metrics

    if logs:
        plot_training_curves(logs)
    else:
        print("No training logs found")

    # 2. Parsing task comparison
    print("\n--- Parsing Task Comparison ---")
    baseline_path = OUTPUT_DIR / "results_baseline_qwen7b.json"
    trained_path = OUTPUT_DIR / "results_zinc_test.json"
    if baseline_path.exists() and trained_path.exists():
        plot_parsing_comparison(baseline_path, trained_path)
        plot_improvement_chart(baseline_path, trained_path)
    else:
        print(f"Missing: baseline={baseline_path.exists()}, trained={trained_path.exists()}")

    # 3. Downstream results
    print("\n--- Downstream Results ---")
    downstream_files = {
        "retrosynthesis": OUTPUT_DIR / "results_downstream_retrosynthesis.json",
        "reagent_prediction": OUTPUT_DIR / "results_downstream_reagent_prediction.json",
        "forward_reaction_prediction": OUTPUT_DIR / "results_downstream_forward_reaction_prediction.json",
    }
    plot_downstream_results(downstream_files)

    # 4. COD comparison
    print("\n--- COD Comparison ---")
    plot_cod_comparison(
        OUTPUT_DIR / "results_zinc_test.json",
        OUTPUT_DIR / "results_cod_trained.json",
    )

    print("\n" + "=" * 60)
    print(f"All plots saved to {PLOT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
