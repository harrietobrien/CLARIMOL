#!/bin/bash
#SBATCH --job-name=cm_persample_seeds
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/persample/persample_seeds_%j.log
#SBATCH --error=output/logs/persample/persample_seeds_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# Per-sample eval on seed_137 and seed_2024 for LLaMA-8B
# Enables cross-seed error correlation analysis
#
# Submit: sbatch scripts/eval_persample_seeds.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/persample

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== Per-sample Eval: LLaMA-8B seeds 137, 2024 ==="
nvidia-smi
echo "Start: $(date)"

for SEED in seed_137 seed_2024; do
    CHECKPOINT="output/multi_seed/llama-8b/${SEED}/final"
    OUTDIR="output/multi_seed/llama-8b/${SEED}"
    PRED_FILE="${OUTDIR}/predictions.jsonl"

    if [ -f "$PRED_FILE" ]; then
        echo "SKIP: predictions exist for llama-8b/${SEED}"
        continue
    fi

    if [ ! -d "$CHECKPOINT" ]; then
        echo "SKIP: checkpoint missing for llama-8b/${SEED}"
        continue
    fi

    echo ""
    echo "=== llama-8b/${SEED} ($(date)) ==="
    python -m clarimol evaluate \
        --model-path "$CHECKPOINT" \
        --data-dir data/test \
        --output-file "${OUTDIR}/results_persample.json" \
        --save-predictions "$PRED_FILE" \
        --no-unsloth \
        --batch-size 16

    echo "llama-8b/${SEED} done: $(date)"
done

echo ""
echo "=== Complete at $(date) ==="
