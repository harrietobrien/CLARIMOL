#!/bin/bash
#SBATCH --job-name=cm_lora
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=output/logs/lr_lora/lora_sweep_%j.log
#SBATCH --error=output/logs/lr_lora/lora_sweep_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# LoRA rank/alpha ablation (6 configs)
# r in {16, 32, 128}, alpha in {r, 2r}, lr=1e-4
# (r=64/a=16 at lr=1e-4 is the multi-seed baseline — skip)
#

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/lr_lora

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="meta-llama/Llama-3.1-8B-Instruct"
BASE_OUT="output/ablation_lora"
SEED=42

echo "=== LoRA Rank/Alpha Sweep ==="
nvidia-smi
echo "Start: $(date)"

for RANK in 16 32 128; do
    for ALPHA_MULT in 1 2; do
        ALPHA=$((RANK * ALPHA_MULT))
        OUT_DIR="${BASE_OUT}/r${RANK}_a${ALPHA}"
        mkdir -p "$OUT_DIR"
        echo ""
        echo "=== LoRA r=$RANK, alpha=$ALPHA ==="
        echo "Start: $(date)"

        python -m clarimol train \
            --model "$MODEL" \
            --data-dir data/clarimol \
            --output-dir "$OUT_DIR" \
            --no-unsloth --max-length 512 \
            --batch-size 16 --grad-accum 1 \
            --lr 1e-4 --epochs 1 \
            --lora-r "$RANK" --lora-alpha "$ALPHA" \
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

echo ""
echo "=== LoRA sweep complete at $(date) ==="
echo ""
for RANK in 16 32 128; do
    for ALPHA_MULT in 1 2; do
        ALPHA=$((RANK * ALPHA_MULT))
        echo -n "  r=$RANK a=$ALPHA: "
        python -c "
import json
with open('${BASE_OUT}/r${RANK}_a${ALPHA}/results.json') as f:
    d = json.load(f)
accs = [v['accuracy'] for v in d.values()]
print(f'mean={sum(accs)/len(accs):.4f}')
" 2>/dev/null || echo "no results"
    done
done
