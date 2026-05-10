#!/bin/bash
#
# Run training + evaluation across multiple seeds for statistical significance.
#
# Usage:
#   bash scripts/run_multi_seed.sh [--seeds "42 137 2024"] [-- <train args>]
#
# Example:
#   bash scripts/run_multi_seed.sh --seeds "42 137 2024" -- \
#       --model meta-llama/Llama-3.1-8B-Instruct \
#       --data-dir data/clarimol --bf16 --no-4bit --no-unsloth
#
# Results are saved to output/<base_output_dir>/seed_<N>/results.json
# and a summary is written to output/<base_output_dir>/multi_seed_summary.json

set -euo pipefail

SEEDS="42 137 2024"
TRAIN_ARGS=()
OUTPUT_DIR="output/clarimol"
DATA_DIR="data/clarimol"
TEST_DIR="data/test"

# Parse our flags vs passthrough args
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) SEEDS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --test-dir) TEST_DIR="$2"; shift 2 ;;
        --) shift; TRAIN_ARGS=("$@"); break ;;
        *) TRAIN_ARGS+=("$1"); shift ;;
    esac
done

echo "=== Multi-seed training ==="
echo "Seeds: $SEEDS"
echo "Output: $OUTPUT_DIR"
echo "Train args: ${TRAIN_ARGS[*]}"
echo ""

for SEED in $SEEDS; do
    SEED_DIR="${OUTPUT_DIR}/seed_${SEED}"
    echo "--- Seed $SEED → $SEED_DIR ---"

    python -m clarimol train \
        --output-dir "$SEED_DIR" \
        --seed "$SEED" \
        "${TRAIN_ARGS[@]}"

    echo "Evaluating seed $SEED..."
    python -m clarimol evaluate \
        --model-path "$SEED_DIR/final" \
        --data-dir "$TEST_DIR" \
        --output-file "$SEED_DIR/results.json" \
        --no-unsloth

    echo "Seed $SEED complete."
    echo ""
done

echo "=== All seeds complete ==="
echo "Aggregating results..."
python -c "
import json, sys, pathlib, statistics

seeds = '$SEEDS'.split()
base = pathlib.Path('$OUTPUT_DIR')
all_results = {}

for seed in seeds:
    result_path = base / f'seed_{seed}' / 'results.json'
    if result_path.exists():
        with open(result_path) as f:
            all_results[seed] = json.load(f)
    else:
        print(f'WARNING: missing {result_path}', file=sys.stderr)

if not all_results:
    print('No results found.', file=sys.stderr)
    sys.exit(1)

tasks = list(next(iter(all_results.values())).keys())
summary = {}
for task in tasks:
    accs = [all_results[s][task]['accuracy'] for s in all_results]
    summary[task] = {
        'mean_accuracy': statistics.mean(accs),
        'std_accuracy': statistics.stdev(accs) if len(accs) > 1 else 0.0,
        'per_seed': {s: all_results[s][task]['accuracy'] for s in all_results},
    }
    print(f'{task:25s}  {summary[task][\"mean_accuracy\"]:.4f} ± {summary[task][\"std_accuracy\"]:.4f}')

out_path = base / 'multi_seed_summary.json'
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSummary saved to {out_path}')
"
