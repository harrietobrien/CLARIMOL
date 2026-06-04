#!/bin/bash
#SBATCH --job-name=cm_smolm360
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/scaling/smolm360_%j.log
#SBATCH --error=output/logs/scaling/smolm360_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# Model scale sweep: SmolLM2-360M on SMILES parsing tasks.
# Adds 360M as a third scale point (360M, 1.7B, 7-8B).
#
# Submit: sbatch scripts/train_scale_smolm_360m.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/scaling

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== Scale Sweep: SmolLM2-360M ==="
nvidia-smi
echo "Start: $(date)"

OUT_DIR="output/scale_sweep/smolm2-360m"

# Train with 3 seeds
for SEED in 42 137 2024; do
    SEED_DIR="$OUT_DIR/seed_${SEED}"

    if [ -f "$SEED_DIR/results.json" ]; then
        echo "SKIP: seed_${SEED} (results exist)"
        continue
    fi

    echo ""
    echo "=== SmolLM2-360M seed_${SEED} ($(date)) ==="

    # Train
    if [ ! -d "$SEED_DIR/final" ]; then
        RESUME_FLAG=""
        if ls "$SEED_DIR"/checkpoint-* 1>/dev/null 2>&1; then
            RESUME_FLAG="--resume"
        fi
        python -m clarimol train \
            --model HuggingFaceTB/SmolLM2-360M-Instruct \
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
        echo "--- seed_${SEED} results ---"
        python3 -c "
import json
d = json.load(open('$SEED_DIR/results.json'))
accs = {k: round(v['accuracy'], 4) for k, v in d.items() if 'accuracy' in v}
print(accs)
print(f'mean={sum(accs.values())/len(accs):.4f}')
"
    fi
done

# Zero-shot eval
ZERO_DIR="$OUT_DIR/zero_shot"
if [ ! -f "$ZERO_DIR/results.json" ]; then
    echo ""
    echo "=== SmolLM2-360M zero-shot ($(date)) ==="
    mkdir -p "$ZERO_DIR"
    python -m clarimol evaluate \
        --model-path HuggingFaceTB/SmolLM2-360M-Instruct \
        --data-dir data/test \
        --output-file "$ZERO_DIR/results.json" \
        --no-unsloth \
        --max-samples 5000 \
        --batch-size 16
fi

echo ""
echo "=== Complete at $(date) ==="
