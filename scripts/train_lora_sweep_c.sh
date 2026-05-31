#!/bin/bash
#SBATCH --job-name=cm_lora_c
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/lr_lora/lora_sweep_c_%j.log
#SBATCH --error=output/logs/lr_lora/lora_sweep_c_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# LoRA sweep batch C: remaining 8 configs to complete 4x5 grid
# r32/{a16,a128,a256}, r64/{a16,a256}, r128/{a16,a32,a64}
#
# Submit: sbatch scripts/train_lora_sweep_c.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/lr_lora

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== LoRA Sweep Batch C (remaining 8 configs) ==="
nvidia-smi
echo "Start: $(date)"

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

CONFIGS=(
    "32,16"
    "32,128"
    "32,256"
    "64,16"
    "64,256"
    "128,16"
    "128,32"
    "128,64"
)

for cfg in "${CONFIGS[@]}"; do
    IFS=',' read -r RANK ALPHA <<< "$cfg"
    OUT_DIR="output/ablation_lora/r${RANK}_a${ALPHA}"

    if [ -f "$OUT_DIR/results.json" ]; then
        echo "SKIP: r${RANK}/a${ALPHA} (results exist)"
        continue
    fi

    echo ""
    echo "=== r${RANK}/a${ALPHA} (eta_eff=$(python3 -c "print(f'{1e-4 * $ALPHA / $RANK:.1e}')")) ($(date)) ==="

    RESUME_FLAG=""
    if ls "$OUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
        RESUME_FLAG="--resume"
    fi

    python -m clarimol train \
        --model "$MODEL_ID" \
        --data-dir data/clarimol \
        --output-dir "$OUT_DIR" \
        --no-unsloth \
        --max-length 512 \
        --batch-size 16 \
        --grad-accum 1 \
        --lr 1e-4 \
        --epochs 1 \
        --lora-r $RANK \
        --lora-alpha $ALPHA \
        --bf16 \
        --no-4bit \
        --no-wandb \
        --save-steps 500 \
        --seed 42 \
        $RESUME_FLAG

    if [ -d "$OUT_DIR/final" ]; then
        python -m clarimol evaluate \
            --model-path "$OUT_DIR/final" \
            --data-dir data/test \
            --output-file "$OUT_DIR/results.json" \
            --no-unsloth \
            --batch-size 16

        echo "--- r${RANK}/a${ALPHA} ---"
        python3 -c "
import json
d = json.load(open('$OUT_DIR/results.json'))
accs = {k: round(v['accuracy'], 4) for k, v in d.items() if 'accuracy' in v}
print(accs)
print(f'mean={sum(accs.values())/len(accs):.4f}')
"
    fi
done

echo ""
echo "=== Complete at $(date) ==="
