#!/bin/bash
#SBATCH --job-name=cm_persample_g1
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/persample/persample_g1_%j.log
#SBATCH --error=output/logs/persample/persample_g1_%j.err
#SBATCH --requeue
#
# GPU 1: Per-sample eval for LLaMA, Mistral, OLMo
# Produces predictions.jsonl for bootstrap CIs, McNemar tests, error analysis
#
# Submit: sbatch scripts/eval_persample_gpu1.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/persample

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== Per-sample Eval (GPU 1): LLaMA, Mistral, OLMo ==="
nvidia-smi
echo "Start: $(date)"

for MODEL in llama-8b mistral-7b olmo-7b; do
    CHECKPOINT="output/multi_seed/${MODEL}/seed_42/final"
    OUTDIR="output/multi_seed/${MODEL}/seed_42"
    PRED_FILE="${OUTDIR}/predictions.jsonl"

    if [ -f "$PRED_FILE" ]; then
        echo "SKIP: predictions exist for ${MODEL}"
        continue
    fi

    if [ ! -d "$CHECKPOINT" ]; then
        echo "SKIP: checkpoint missing for ${MODEL}"
        continue
    fi

    echo ""
    echo "=== ${MODEL} ($(date)) ==="
    python -m clarimol evaluate \
        --model-path "$CHECKPOINT" \
        --data-dir data/test \
        --output-file "${OUTDIR}/results_persample.json" \
        --save-predictions "$PRED_FILE" \
        --no-unsloth \
        --batch-size 16

    echo "${MODEL} done: $(date)"
done

echo ""
echo "=== GPU 1 complete at $(date) ==="
