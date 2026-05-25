#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/persample/persample_%A_%a.log
#SBATCH --error=output/logs/persample/persample_%A_%a.err
#SBATCH --array=0-4
#SBATCH --job-name=cm_persample

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

cd ~/storage/CLARIMOL
mkdir -p output/logs/persample

MODELS=(llama-8b mistral-7b olmo-7b qwen-7b qwen3-8b)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

CHECKPOINT="output/multi_seed/${MODEL}/seed_42/final"
OUTDIR="output/multi_seed/${MODEL}/seed_42"
PRED_FILE="${OUTDIR}/predictions.jsonl"

if [ -f "$PRED_FILE" ]; then
    echo "Predictions already exist for ${MODEL}, skipping"
    exit 0
fi

echo "=== Per-sample eval: ${MODEL} ==="
nvidia-smi
echo "Start: $(date)"

python -m clarimol evaluate \
    --model-path "$CHECKPOINT" \
    --data-dir data/test \
    --output-file "${OUTDIR}/results_persample.json" \
    --save-predictions "$PRED_FILE" \
    --batch-size 16

echo "Done: $(date)"
