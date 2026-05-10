#!/bin/bash
#SBATCH --job-name=cm_scaling
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/scaling_%j.log
#SBATCH --error=output/scaling_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# P2-17: Dataset size scaling curve
# How much data is enough? Tests {10K, 25K, 50K, 100K, 250K} per task
# 5 runs @ varying time = ~2 hours total
#
# Pairs with job_epochs_instruct.slurm on the other H200.
#   sbatch training_duke/job_dataset_scaling.slurm
#   sbatch training_duke/job_epochs_instruct.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="meta-llama/Llama-3.1-8B-Instruct"
SIZES=(10000 25000 50000 100000 250000)
BASE_DATA="data/scaling"
BASE_OUT="output/ablation_dataset_size"
SEED=42

echo "=== Dataset Size Scaling ==="
nvidia-smi
echo "Start: $(date)"

# Prepare datasets at each scale
for SIZE in "${SIZES[@]}"; do
    DATA_DIR="${BASE_DATA}/keep_${SIZE}"
    if [ -d "$DATA_DIR" ] && [ -n "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        echo "Data for n=$SIZE exists, skipping"
        continue
    fi
    echo "Preparing data: keep_n=$SIZE"
    python -m clarimol prepare \
        --output-dir "$DATA_DIR" \
        --keep-n "$SIZE" \
        --seed "$SEED"
done

# Train + eval at each scale
for SIZE in "${SIZES[@]}"; do
    DATA_DIR="${BASE_DATA}/keep_${SIZE}"
    OUT_DIR="${BASE_OUT}/size_${SIZE}"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "=== n=$SIZE per task ==="
    echo "Start: $(date)"

    python -m clarimol train \
        --model "$MODEL" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUT_DIR" \
        --no-unsloth --max-length 512 \
        --batch-size 16 --grad-accum 1 \
        --lr 1e-4 --epochs 1 \
        --lora-r 64 --lora-alpha 16 \
        --bf16 --no-4bit --no-wandb \
        --save-steps 500 --seed "$SEED"

    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/results.json" \
        --no-unsloth --batch-size 16

    echo "Done: $(date)"
done

echo ""
echo "=== Scaling curve complete at $(date) ==="
echo ""
echo "Results:"
for SIZE in "${SIZES[@]}"; do
    echo -n "  n=$SIZE: "
    python -c "
import json
with open('${BASE_OUT}/size_${SIZE}/results.json') as f:
    d = json.load(f)
accs = [v['accuracy'] for v in d.values()]
print(f'mean={sum(accs)/len(accs):.4f}  ' + '  '.join(f'{k}={v[\"accuracy\"]:.3f}' for k,v in d.items()))
" 2>/dev/null || echo "no results"
done
