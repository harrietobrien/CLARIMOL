#!/bin/bash
#SBATCH --job-name=cm_olmo_s99
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/multi_seed/olmo_s99_%j.log
#SBATCH --error=output/logs/multi_seed/olmo_s99_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# OLMo-7B seed 99 only (seed 7 already complete).
#
# Submit: sbatch scripts/train_olmo_seed99.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/multi_seed

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== OLMo-7B Seed 99 ==="
nvidia-smi
echo "Start: $(date)"

MODEL_ID="allenai/OLMo-3-7B-Instruct"
MODEL_NAME="olmo-7b"
SEED=99
SEED_DIR="output/multi_seed/${MODEL_NAME}/seed_${SEED}"

if [ -f "$SEED_DIR/results.json" ]; then
    echo "SKIP: ${MODEL_NAME}/seed_${SEED} (results exist)"
    echo "=== Complete at $(date) ==="
    exit 0
fi

echo ""
echo "=== ${MODEL_NAME} seed_${SEED} ($(date)) ==="

# Train
if [ ! -d "$SEED_DIR/final" ]; then
    RESUME_FLAG=""
    if ls "$SEED_DIR"/checkpoint-* 1>/dev/null 2>&1; then
        RESUME_FLAG="--resume"
    fi
    python -m clarimol train \
        --model "$MODEL_ID" \
        --data-dir data/clarimol \
        --output-dir "$SEED_DIR" \
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
        --seed $SEED \
        $RESUME_FLAG
fi

# Evaluate
if [ -d "$SEED_DIR/final" ] && [ ! -f "$SEED_DIR/results.json" ]; then
    python -m clarimol evaluate \
        --model-path "$SEED_DIR/final" \
        --data-dir data/test \
        --output-file "$SEED_DIR/results.json" \
        --no-unsloth \
        --batch-size 16
fi

if [ -f "$SEED_DIR/results.json" ]; then
    echo "--- ${MODEL_NAME}/seed_${SEED} ---"
    python3 -c "
import json
d = json.load(open('$SEED_DIR/results.json'))
accs = {k: round(v['accuracy'], 4) for k, v in d.items() if 'accuracy' in v}
print(accs)
print(f'mean={sum(accs.values())/len(accs):.4f}')
"
fi

echo ""
echo "=== Complete at $(date) ==="
