#!/bin/bash
#SBATCH --job-name=cm_8tv2_olm
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/new_tasks/8task_v2_olmo_%j.log
#SBATCH --error=output/logs/new_tasks/8task_v2_olmo_%j.err
#SBATCH --requeue
#
# 8-task v2 (with system prompts + instruction templates) for OLMo-3-7B.
# Uses existing data at data/clarimol_8task/ — prompts applied at train time.
#
# Submit: sbatch training_duke/job_8task_v2_olmo.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/new_tasks

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="allenai/OLMo-3-7B-Instruct"
TRAIN_DIR="data/clarimol_8task"
TEST_DIR="data/test_8task"
OUT_DIR="output/ablation_8task_v2_olmo"

echo "=== 8-task v2 OLMo-3-7B ==="
nvidia-smi
echo "Start: $(date)"

if [ ! -f "$TRAIN_DIR/stereocenter.json" ]; then
    echo "ERROR: 8-task training data not found at $TRAIN_DIR"
    exit 1
fi

# Train
if [ ! -d "$OUT_DIR/final" ]; then
    echo "=== Training on 8 tasks (v2: system prompts + instruction templates) ==="
    RESUME_FLAG=""
    if ls "$OUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
        echo "Checkpoint(s) detected — resuming"
        RESUME_FLAG="--resume"
    fi
    python -m clarimol train \
        --model "$MODEL" \
        --data-dir "$TRAIN_DIR" \
        --output-dir "$OUT_DIR" \
        --no-unsloth \
        --max-length 512 \
        --batch-size 16 \
        --grad-accum 1 \
        --lr 1e-4 \
        --epochs 1 \
        --lora-r 64 \
        --lora-alpha 16 \
        --bf16 \
        --no-4bit \
        --no-wandb \
        --save-steps 500 \
        --seed 42 \
        $RESUME_FLAG
else
    echo "SKIP: training complete"
fi

# Evaluate 8-task model
if [ ! -f "$OUT_DIR/results.json" ]; then
    echo "=== Evaluating 8-task v2 model ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir "$TEST_DIR" \
        --output-file "$OUT_DIR/results.json" \
        --no-unsloth \
        --batch-size 16
else
    echo "SKIP: results exist"
fi

echo ""
echo "=== Results ==="
if [ -f "$OUT_DIR/results.json" ]; then
    python3 -c "
import json
d = json.load(open('$OUT_DIR/results.json'))
for t in sorted(d.keys()):
    if 'accuracy' in d[t]:
        line = f'  {t}: {d[t][\"accuracy\"]:.4f}'
        if 'details' in d[t] and d[t]['details']:
            extras = ', '.join(f'{k}={v}' for k,v in d[t]['details'].items())
            line += f'  ({extras})'
        print(line)
"
fi

echo "=== Complete at $(date) ==="
