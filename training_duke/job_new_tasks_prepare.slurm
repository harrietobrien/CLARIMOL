#!/bin/bash
#SBATCH --job-name=cm_new_tasks
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/new_tasks/new_tasks_prep_train_%j.log
#SBATCH --error=output/logs/new_tasks/new_tasks_prep_train_%j.err
#SBATCH --requeue
#
# Prepare data for 3 new tasks, then train + eval on all 8 tasks.
#
# Submit: sbatch training_duke/job_new_tasks_prepare.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/new_tasks

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

echo "=== New Tasks: Prepare + Train + Eval ==="
nvidia-smi
echo "Start: $(date)"

# Step 1: Prepare training data for new tasks only (from ZINC250K)
TRAIN_DIR="data/clarimol_8task"
TEST_DIR="data/test_8task"

if [ ! -f "$TRAIN_DIR/stereocenter.json" ]; then
    echo "=== Preparing 8-task training data ==="
    python -m clarimol prepare \
        --output-dir "$TRAIN_DIR" \
        --tasks stereocenter,atom_degree,topological_distance \
        --max-molecules 250000 \
        --keep-n 50000
else
    echo "SKIP: training data exists"
fi

# Copy existing 5 tasks into the 8-task dir
for task in functional_group ring_counting chain_length canonicalization fragment_assembly; do
    if [ ! -f "$TRAIN_DIR/$task.json" ] && [ -f "data/clarimol/$task.json" ]; then
        cp "data/clarimol/$task.json" "$TRAIN_DIR/$task.json"
    fi
done

# Step 2: Prepare test data for new tasks
if [ ! -f "$TEST_DIR/stereocenter.json" ]; then
    echo "=== Preparing 8-task test data ==="
    python -m clarimol prepare \
        --split validation \
        --output-dir "$TEST_DIR" \
        --tasks stereocenter,atom_degree,topological_distance \
        --max-molecules 10000 \
        --subsample random \
        --no-curriculum
else
    echo "SKIP: test data exists"
fi

# Copy existing test tasks
for task in functional_group ring_counting chain_length canonicalization fragment_assembly; do
    if [ ! -f "$TEST_DIR/$task.json" ] && [ -f "data/test/$task.json" ]; then
        cp "data/test/$task.json" "$TEST_DIR/$task.json"
    fi
done

# Step 3: Train on all 8 tasks
OUT_DIR="output/ablation_8task"
if [ ! -d "$OUT_DIR/final" ]; then
    echo "=== Training on 8 tasks ==="
    RESUME_FLAG=""
    if ls "$OUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
        RESUME_FLAG="--resume"
    fi
    python -m clarimol train \
        --model meta-llama/Llama-3.1-8B-Instruct \
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

# Step 4: Evaluate on all 8 tasks
if [ ! -f "$OUT_DIR/results.json" ]; then
    echo "=== Evaluating 8-task model ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir "$TEST_DIR" \
        --output-file "$OUT_DIR/results.json" \
        --no-unsloth \
        --batch-size 16
else
    echo "SKIP: results exist"
fi

# Step 5: Also eval the existing 5-task model on new tasks (zero-shot transfer)
FIVE_TASK_MODEL="output/multi_seed/llama-8b/seed_42/final"
if [ -d "$FIVE_TASK_MODEL" ] && [ ! -f "$OUT_DIR/5task_on_8task.json" ]; then
    echo "=== Eval 5-task model on 8 tasks ==="
    python -m clarimol evaluate \
        --model-path "$FIVE_TASK_MODEL" \
        --data-dir "$TEST_DIR" \
        --output-file "$OUT_DIR/5task_on_8task.json" \
        --no-unsloth \
        --batch-size 16
fi

echo ""
echo "=== Results ==="
for f in "$OUT_DIR"/*.json; do
    echo "--- $(basename $f) ---"
    python3 -c "
import json
d = json.load(open('$f'))
for t in sorted(d.keys()):
    if 'accuracy' in d[t]:
        print(f'  {t}: {d[t][\"accuracy\"]:.4f}')
" 2>/dev/null
done

echo "=== Complete at $(date) ==="
