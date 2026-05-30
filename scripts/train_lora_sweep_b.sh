#!/bin/bash
#SBATCH --job-name=cm_lora_sweep_b
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/lr_lora/lora_sweep_b_%j.log
#SBATCH --error=output/logs/lr_lora/lora_sweep_b_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# LoRA sweep batch B: r64 and r128 low-alpha configs
#
# Submit: sbatch scripts/train_lora_sweep_b.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/lr_lora

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== LoRA Sweep Batch B ==="
nvidia-smi
echo "Start: $(date)"

CONFIGS=(
    "64 32"
    "64 64"
    "64 128"
    "64 256"
    "128 16"
    "128 32"
    "128 64"
)

for CONFIG in "${CONFIGS[@]}"; do
    R=$(echo $CONFIG | cut -d' ' -f1)
    A=$(echo $CONFIG | cut -d' ' -f2)
    NAME="r${R}_a${A}"
    OUT_DIR="output/ablation_lora/${NAME}"

    if [ -f "$OUT_DIR/results.json" ]; then
        echo "SKIP: ${NAME} (results exist)"
        continue
    fi

    echo ""
    echo "=== ${NAME} (r=${R}, alpha=${A}) ==="
    echo "$(date)"

    if [ ! -d "$OUT_DIR/final" ]; then
        RESUME_FLAG=""
        if ls "$OUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
            RESUME_FLAG="--resume"
        fi
        python -m clarimol train \
            --model meta-llama/Llama-3.1-8B-Instruct \
            --data-dir data/clarimol \
            --output-dir "$OUT_DIR" \
            --no-unsloth \
            --max-length 512 \
            --batch-size 16 \
            --grad-accum 1 \
            --lr 1e-4 \
            --epochs 1 \
            --lora-r $R \
            --lora-alpha $A \
            --bf16 \
            --no-4bit \
            --no-wandb \
            --save-steps 500 \
            --seed 42 \
            $RESUME_FLAG
    fi

    if [ -d "$OUT_DIR/final" ] && [ ! -f "$OUT_DIR/results.json" ]; then
        python -m clarimol evaluate \
            --model-path "$OUT_DIR/final" \
            --data-dir data/test \
            --output-file "$OUT_DIR/results.json" \
            --no-unsloth \
            --batch-size 16
    fi

    if [ -f "$OUT_DIR/results.json" ]; then
        echo "--- ${NAME} results ---"
        python3 -c "
import json
d = json.load(open('$OUT_DIR/results.json'))
accs = {k: round(v['accuracy'], 4) for k, v in d.items() if 'accuracy' in v}
print(accs)
print(f'mean={sum(accs.values())/len(accs):.4f}')
"
    fi

    echo "${NAME} done: $(date)"
done

echo "=== Batch B complete at $(date) ==="
