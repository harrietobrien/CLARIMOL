#!/bin/bash
#SBATCH --job-name=cm_250k_eval
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/scaling_epochs/size250k_eval_%j.log
#SBATCH --error=output/logs/scaling_epochs/size250k_eval_%j.err
#SBATCH --requeue
#
# Eval-only: dataset size 250K from latest checkpoint (training incomplete).
#
# Submit: sbatch training_duke/job_size250k_eval.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/scaling_epochs

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

CKPT=$(ls -d output/ablation_dataset_size/size_250000/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

echo "=== Eval dataset size 250K from checkpoint ==="
echo "Checkpoint: $CKPT"
nvidia-smi
echo "Start: $(date)"

if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found"
    exit 1
fi

if [ ! -f "output/ablation_dataset_size/size_250000/results.json" ]; then
    python -m clarimol evaluate \
        --model-path "$CKPT" \
        --data-dir data/test \
        --output-file output/ablation_dataset_size/size_250000/results.json \
        --no-unsloth --batch-size 16
else
    echo "SKIP: results.json exists"
fi

echo "=== Complete at $(date) ==="
