#!/bin/bash
#SBATCH --job-name=cm_ep2_eval
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/scaling_epochs/ep2_eval_%j.log
#SBATCH --error=output/logs/scaling_epochs/ep2_eval_%j.err
#SBATCH --requeue
#
# Eval-only: random_ep2 model on ZINC test set.
#
# Submit: sbatch training_duke/job_epochs_ep2_eval.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/scaling_epochs

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="output/ablation_epochs/random_ep2/final"
OUT="output/ablation_epochs/random_ep2/results.json"

echo "=== Eval random_ep2 ==="
nvidia-smi
echo "Start: $(date)"

if [ ! -d "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL"
    exit 1
fi

if [ ! -f "$OUT" ]; then
    python -m clarimol evaluate \
        --model-path "$MODEL" \
        --data-dir data/test \
        --output-file "$OUT" \
        --no-unsloth --batch-size 16
else
    echo "SKIP: results.json exists"
fi

echo "=== Complete at $(date) ==="
