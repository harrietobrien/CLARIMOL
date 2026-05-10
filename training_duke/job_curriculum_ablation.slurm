#!/bin/bash
#SBATCH --job-name=cm_curriculum
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/curriculum_%j.log
#SBATCH --error=output/curriculum_%j.err
#SBATCH --requeue
#
# P1-8: Curriculum ordering ablation
# Compares: easy-hard (default), hard-easy, random, none
# Uses LLaMA 3.1-8B, full 250K dataset
#
# Pairs well with job_baselines.slurm on the other H200.
#   sbatch training_duke/job_baselines.slurm
#   sbatch training_duke/job_curriculum_ablation.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="meta-llama/Llama-3.1-8B-Instruct"
ORDERS=("easy-hard" "hard-easy" "random" "none")
BASE_DATA="data/curriculum_ablation"
BASE_OUT="output/ablation_curriculum"
SEED=42

echo "=== Curriculum Ordering Ablation ==="
nvidia-smi
echo "Start: $(date)"

# Step 1: Prepare datasets with each ordering
for ORDER in "${ORDERS[@]}"; do
    DATA_DIR="${BASE_DATA}/${ORDER}"
    if [ -d "$DATA_DIR" ] && [ -n "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        echo "Dataset for $ORDER already exists, skipping"
        continue
    fi
    echo "Preparing dataset: curriculum=$ORDER"
    python -m clarimol prepare \
        --output-dir "$DATA_DIR" \
        --curriculum-order "$ORDER" \
        --seed "$SEED"
done

# Step 2: Train + evaluate each
for ORDER in "${ORDERS[@]}"; do
    DATA_DIR="${BASE_DATA}/${ORDER}"
    OUT_DIR="${BASE_OUT}/${ORDER}"
    mkdir -p "$OUT_DIR"

    RESUME_FLAG=""
    if ls "$OUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
        RESUME_FLAG="--resume"
    fi

    echo ""
    echo "=== Curriculum: $ORDER ==="
    echo "Start: $(date)"

    python -m clarimol train \
        --model "$MODEL" \
        --data-dir "$DATA_DIR" \
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
        --seed "$SEED" \
        $RESUME_FLAG

    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/results.json" \
        --no-unsloth \
        --batch-size 16

    echo "Done: $(date)"
done

echo ""
echo "=== Curriculum ablation complete at $(date) ==="
echo "Comparison:"
for ORDER in "${ORDERS[@]}"; do
    echo -n "  $ORDER: "
    python -c "
import json
with open('${BASE_OUT}/${ORDER}/results.json') as f:
    d = json.load(f)
accs = [v['accuracy'] for v in d.values()]
print(f'mean={sum(accs)/len(accs):.4f}  ' + '  '.join(f'{k}={v[\"accuracy\"]:.3f}' for k,v in d.items()))
" 2>/dev/null || echo "no results"
done
