#!/bin/bash
#SBATCH --job-name=cm_cross_task
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=output/logs/cross_task/cross_task_%j.log
#SBATCH --error=output/logs/cross_task/cross_task_%j.err
#
# Cross-task transfer experiment.
# For each of the 5 SMILES parsing tasks as the held-out task:
#   1. Symlink the 4 remaining tasks' JSON files into a filtered data directory.
#   2. Train LLaMA-3.1-8B-Instruct on those 4 tasks (r64/a128, sweet spot from ablation).
#   3. Evaluate the trained model on ALL 5 tasks.
# Produces a 5x5 transfer matrix: trained-on-4 vs. evaluated-on-each.
#
# Auto-resume: skips completed runs; detects mid-run checkpoints and passes --resume.
#
# Submit: sbatch scripts/train_cross_task_transfer.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/cross_task

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== Cross-Task Transfer Experiment ==="
nvidia-smi
echo "Start: $(date)"

ALL_TASKS=("functional_group" "ring_counting" "chain_length" "canonicalization" "fragment_assembly")
BASE_OUT="output/cross_task_transfer"
SOURCE_DATA="data/clarimol"

for HOLDOUT in "${ALL_TASKS[@]}"; do
    OUT_DIR="${BASE_OUT}/holdout_${HOLDOUT}"
    FILTERED_DATA="data/cross_task_filtered/holdout_${HOLDOUT}"

    echo ""
    echo "=== Holdout: ${HOLDOUT} ==="

    # Skip this holdout entirely if final evaluation already exists.
    if [ -f "${OUT_DIR}/results.json" ]; then
        echo "SKIP: ${OUT_DIR}/results.json exists"
        continue
    fi

    # Build filtered data directory: symlink the 4 included tasks' JSON files.
    mkdir -p "${FILTERED_DATA}"
    for TASK in "${ALL_TASKS[@]}"; do
        if [ "${TASK}" = "${HOLDOUT}" ]; then
            continue
        fi
        TARGET="${FILTERED_DATA}/${TASK}.json"
        if [ ! -L "${TARGET}" ] && [ ! -f "${TARGET}" ]; then
            ln -s "$(realpath "${SOURCE_DATA}/${TASK}.json")" "${TARGET}"
        fi
    done

    # Verify all 4 symlinks/files are present.
    INCLUDED_COUNT=$(ls "${FILTERED_DATA}"/*.json 2>/dev/null | wc -l)
    if [ "${INCLUDED_COUNT}" -ne 4 ]; then
        echo "ERROR: expected 4 task files in ${FILTERED_DATA}, found ${INCLUDED_COUNT}"
        exit 1
    fi

    mkdir -p "${OUT_DIR}"

    # Training: skip if final checkpoint exists, resume if mid-run checkpoint exists.
    if [ ! -d "${OUT_DIR}/final" ]; then
        RESUME_FLAG=""
        if ls "${OUT_DIR}"/checkpoint-* 1>/dev/null 2>&1; then
            echo "Checkpoint detected — resuming training for holdout_${HOLDOUT}"
            RESUME_FLAG="--resume"
        else
            echo "Starting training for holdout_${HOLDOUT}"
        fi
        python -m clarimol train \
            --model meta-llama/Llama-3.1-8B-Instruct \
            --data-dir "${FILTERED_DATA}" \
            --output-dir "${OUT_DIR}" \
            --no-unsloth \
            --max-length 512 \
            --batch-size 16 \
            --grad-accum 1 \
            --lr 1e-4 \
            --epochs 1 \
            --lora-r 64 \
            --lora-alpha 128 \
            --bf16 \
            --no-4bit \
            --no-wandb \
            --save-steps 500 \
            --seed 42 \
            $RESUME_FLAG
    else
        echo "SKIP: training complete for holdout_${HOLDOUT}"
    fi

    # Evaluation: run on all 5 tasks via the standard test split.
    echo "Evaluating holdout_${HOLDOUT} model on all tasks"
    python -m clarimol evaluate \
        --model-path "${OUT_DIR}/final" \
        --data-dir data/test \
        --output-file "${OUT_DIR}/results.json" \
        --no-unsloth \
        --batch-size 16

    # Print per-task summary with held-out task marked.
    echo ""
    echo "--- Results (holdout=${HOLDOUT}) ---"
    python3 -c "
import json
results = json.load(open('${OUT_DIR}/results.json'))
holdout = '${HOLDOUT}'
all_tasks = ['functional_group', 'ring_counting', 'chain_length', 'canonicalization', 'fragment_assembly']
for task in all_tasks:
    if task not in results:
        continue
    acc = results[task].get('accuracy', float('nan'))
    marker = ' [TRANSFER]' if task == holdout else ''
    print(f'  {task:25s}  {acc:.4f}{marker}')
accs = [results[t]['accuracy'] for t in results if 'accuracy' in results[t]]
if accs:
    print(f'  {\"mean\":25s}  {sum(accs)/len(accs):.4f}')
"
done

echo ""
echo "=== All holdout runs complete ==="
echo ""

# Print compact summary of all completed results.
echo "=== Transfer matrix summary ==="
python3 -c "
import json, pathlib

all_tasks = ['functional_group', 'ring_counting', 'chain_length', 'canonicalization', 'fragment_assembly']
base = pathlib.Path('output/cross_task_transfer')

print('Rows = trained-on-4 (held-out task), Cols = evaluated task')
print(f'  {\"\":20s}', end='')
for t in all_tasks:
    print(f'  {t[:10]:>10s}', end='')
print()

for holdout in all_tasks:
    rpath = base / f'holdout_{holdout}' / 'results.json'
    print(f'  holdout_{holdout[:10]:10s}', end='')
    if rpath.exists():
        results = json.load(open(rpath))
        for t in all_tasks:
            acc = results.get(t, {}).get('accuracy', float('nan'))
            marker = '*' if t == holdout else ' '
            print(f'  {acc:9.4f}{marker}', end='')
    else:
        print('  (not complete)', end='')
    print()

print()
print('* = held-out (transfer) task')
"

echo "=== Complete at $(date) ==="
