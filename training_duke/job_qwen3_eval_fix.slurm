#!/bin/bash
#SBATCH --job-name=cm_qwen3_fix
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/multi_seed/qwen3_eval_fix_%j.log
#SBATCH --error=output/logs/multi_seed/qwen3_eval_fix_%j.err
#SBATCH --requeue
#
# Re-evaluate Qwen3-8B seeds 42 and 137 (broken results from May 12).
# Adapters exist; only eval needed.
#
# Submit: sbatch training_duke/job_qwen3_eval_fix.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

echo "=== Qwen3-8B re-eval seeds 42, 137 ==="
nvidia-smi
echo "Start: $(date)"

for SEED in 42 137; do
    SEED_DIR="output/multi_seed/qwen3-8b/seed_${SEED}"

    if [ ! -d "$SEED_DIR/final" ]; then
        echo "ERROR: $SEED_DIR/final does not exist, skipping"
        continue
    fi

    echo "--- Evaluating seed_${SEED}: $(date) ---"
    python -m clarimol evaluate \
        --model-path "$SEED_DIR/final" \
        --data-dir data/test \
        --output-file "$SEED_DIR/results.json" \
        --no-unsloth \
        --batch-size 16

    echo "seed_${SEED} results:"
    python3 -c "
import json
d = json.load(open('$SEED_DIR/results.json'))
for t in ['functional_group','ring_counting','chain_length','canonicalization','fragment_assembly']:
    if t in d: print(f'  {t}: {d[t][\"accuracy\"]:.4f}')
"
done

echo "=== Complete at $(date) ==="
