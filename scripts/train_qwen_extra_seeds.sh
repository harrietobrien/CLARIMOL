#!/bin/bash
#SBATCH --job-name=cm_qwen_seeds
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/multi_seed/qwen_seeds_%j.log
#SBATCH --error=output/logs/multi_seed/qwen_seeds_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# Extra seeds (7, 99) for Qwen2.5-7B and Qwen3-8B to bring all models to 5 seeds.
#
# Submit: sbatch scripts/train_qwen_extra_seeds.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/multi_seed

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== Qwen2.5 + Qwen3 Extra Seeds (7, 99) ==="
nvidia-smi
echo "Start: $(date)"

declare -A MODELS
MODELS["qwen-7b"]="Qwen/Qwen2.5-7B-Instruct"
MODELS["qwen3-8b"]="Qwen/Qwen3-8B"

SEEDS=(7 99)

for MODEL_NAME in "qwen-7b" "qwen3-8b"; do
    MODEL_ID="${MODELS[$MODEL_NAME]}"

    for SEED in "${SEEDS[@]}"; do
        SEED_DIR="output/multi_seed/${MODEL_NAME}/seed_${SEED}"

        if [ -f "$SEED_DIR/results.json" ]; then
            echo "SKIP: ${MODEL_NAME}/seed_${SEED} (results exist)"
            continue
        fi

        echo ""
        echo "=== ${MODEL_NAME} seed_${SEED} ($(date)) ==="

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
    done
done

echo ""
echo "=== Complete at $(date) ==="
