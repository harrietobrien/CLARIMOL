#!/bin/bash
#SBATCH --job-name=cm_r128a256
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/lr_lora/r128a256_%j.log
#SBATCH --error=output/logs/lr_lora/r128a256_%j.err
#SBATCH --requeue
#
# LoRA r=128, alpha=256 — completes the heatmap at highest effective LR
# eta_eff = 1e-4 * 256/128 = 2e-4
#
# Submit: sbatch scripts/train_lora_r128a256.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/lr_lora

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== LoRA r=128, alpha=256 ==="
nvidia-smi
echo "Start: $(date)"

OUT_DIR="output/ablation_lora/r128_a256"

# Train
if [ ! -d "$OUT_DIR/final" ]; then
    echo "=== Training r128/a256 ==="
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
        --lora-r 128 \
        --lora-alpha 256 \
        --bf16 \
        --no-4bit \
        --no-wandb \
        --save-steps 500 \
        --seed 42 \
        $RESUME_FLAG
else
    echo "SKIP: training complete"
fi

# Evaluate
if [ ! -f "$OUT_DIR/results.json" ]; then
    echo "=== Evaluating r128/a256 ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/results.json" \
        --no-unsloth \
        --batch-size 16
fi

echo ""
echo "=== Results ==="
if [ -f "$OUT_DIR/results.json" ]; then
    python3 -c "
import json
d = json.load(open('$OUT_DIR/results.json'))
for t in sorted(d.keys()):
    if 'accuracy' in d[t]:
        print(f'  {t}: {d[t][\"accuracy\"]:.4f}')
accs = [d[t]['accuracy'] for t in d if 'accuracy' in d[t]]
if accs: print(f'  mean: {sum(accs)/len(accs):.4f}')
"
fi

echo "=== Complete at $(date) ==="
