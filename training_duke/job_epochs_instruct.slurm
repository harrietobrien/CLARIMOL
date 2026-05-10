#!/bin/bash
#SBATCH --job-name=cm_epochs
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/epochs_instruct_%j.log
#SBATCH --error=output/epochs_instruct_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# Combined P3-20 + P3-21:
#   - Multi-epoch training (1, 2, 3 epochs) x (easy-hard, random curriculum)
#   - Instruct vs base model comparison (LLaMA instruct vs base)
#
# 8 training runs @ ~20-60min each = ~4 hours
#
# Pairs with job_dataset_scaling.slurm on the other H200.
#   sbatch training_duke/job_dataset_scaling.slurm
#   sbatch training_duke/job_epochs_instruct.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

SEED=42

echo "=== Multi-Epoch + Instruct vs Base ==="
nvidia-smi
echo "Start: $(date)"

# ─── Multi-epoch (curriculum x epochs) ───────────────────────────────────────
MODEL="meta-llama/Llama-3.1-8B-Instruct"
ORDERS=("easy-hard" "random")
EPOCHS=(1 2 3)
BASE_OUT="output/ablation_epochs"

for ORDER in "${ORDERS[@]}"; do
    DATA_DIR="data/curriculum_ablation/${ORDER}"

    if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        echo "Preparing data: curriculum=$ORDER"
        python -m clarimol prepare \
            --output-dir "$DATA_DIR" \
            --curriculum-order "$ORDER" \
            --seed "$SEED"
    fi

    for EP in "${EPOCHS[@]}"; do
        # Skip 1-epoch easy-hard — covered by multi-seed baseline
        if [ "$ORDER" == "easy-hard" ] && [ "$EP" == "1" ]; then
            echo "Skipping easy-hard ep=1 (covered by multi-seed baseline)"
            continue
        fi

        OUT_DIR="${BASE_OUT}/${ORDER}_ep${EP}"
        mkdir -p "$OUT_DIR"
        echo ""
        echo "=== curriculum=$ORDER, epochs=$EP ==="
        echo "Start: $(date)"

        python -m clarimol train \
            --model "$MODEL" \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUT_DIR" \
            --no-unsloth --max-length 512 \
            --batch-size 16 --grad-accum 1 \
            --lr 1e-4 --epochs "$EP" \
            --lora-r 64 --lora-alpha 16 \
            --bf16 --no-4bit --no-wandb \
            --save-steps 500 --seed "$SEED"

        python -m clarimol evaluate \
            --model-path "$OUT_DIR/final" \
            --data-dir data/test \
            --output-file "$OUT_DIR/results.json" \
            --no-unsloth --batch-size 16

        echo "Done: $(date)"
    done
done

# ─── Instruct vs Base model ──────────────────────────────────────────────────
BASE_OUT_IB="output/instruct_vs_base"

declare -A MODELS
MODELS=(
    ["llama-8b_instruct"]="meta-llama/Llama-3.1-8B-Instruct"
    ["llama-8b_base"]="meta-llama/Llama-3.1-8B"
)

for KEY in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$KEY]}"
    OUT_DIR="${BASE_OUT_IB}/${KEY}"
    mkdir -p "$OUT_DIR"
    echo ""
    echo "=== $KEY: $MODEL_NAME ==="
    echo "Start: $(date)"

    python -m clarimol train \
        --model "$MODEL_NAME" \
        --data-dir data/clarimol \
        --output-dir "$OUT_DIR" \
        --no-unsloth --max-length 512 \
        --batch-size 16 --grad-accum 1 \
        --lr 1e-4 --epochs 1 \
        --lora-r 64 --lora-alpha 16 \
        --bf16 --no-4bit --no-wandb \
        --save-steps 500 --seed "$SEED"

    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/results.json" \
        --no-unsloth --batch-size 16

    echo "Done: $(date)"
done

echo ""
echo "=== All experiments complete at $(date) ==="
echo ""
echo "Multi-epoch results:"
for ORDER in "${ORDERS[@]}"; do
    for EP in "${EPOCHS[@]}"; do
        DIR="${BASE_OUT}/${ORDER}_ep${EP}"
        echo -n "  ${ORDER} ep=${EP}: "
        python -c "
import json
with open('${DIR}/results.json') as f:
    d = json.load(f)
accs = [v['accuracy'] for v in d.values()]
print(f'mean={sum(accs)/len(accs):.4f}')
" 2>/dev/null || echo "no results"
    done
done
echo ""
echo "Instruct vs base:"
for KEY in "${!MODELS[@]}"; do
    DIR="${BASE_OUT_IB}/${KEY}"
    echo -n "  $KEY: "
    python -c "
import json
with open('${DIR}/results.json') as f:
    d = json.load(f)
accs = [v['accuracy'] for v in d.values()]
print(f'mean={sum(accs)/len(accs):.4f}')
" 2>/dev/null || echo "no results"
done
