#!/bin/bash
#SBATCH --job-name=cm_qwen3_eval
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/multi_seed/qwen3_eval_seed2024_%j.log
#SBATCH --error=output/logs/multi_seed/qwen3_eval_seed2024_%j.err
#SBATCH --requeue
#
# Qwen3-8B seed_2024: final/ exists but no results.json. Just run eval.
#
# Submit: sbatch training_duke/job_qwen3_eval_seed2024.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

SEED_DIR="output/multi_seed/qwen3-8b/seed_2024"

echo "=== Qwen3-8B seed_2024 eval ==="
nvidia-smi
echo "Start: $(date)"

if [ ! -d "$SEED_DIR/final" ]; then
    echo "ERROR: $SEED_DIR/final does not exist"
    exit 1
fi

python -m clarimol evaluate \
    --model-path "$SEED_DIR/final" \
    --data-dir data/test \
    --output-file "$SEED_DIR/results.json" \
    --no-unsloth \
    --batch-size 16

echo "Eval complete at $(date)"
echo "Results:"
python3 -c "
import json
d = json.load(open('$SEED_DIR/results.json'))
for t in ['functional_group','ring_counting','chain_length','canonicalization','fragment_assembly']:
    if t in d: print(f'  {t}: {d[t][\"accuracy\"]:.4f}')
"
