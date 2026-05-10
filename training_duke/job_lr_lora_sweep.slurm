#!/bin/bash
#SBATCH --job-name=cm_lr_lora
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=48:00:00
#SBATCH --output=output/lr_lora_%j.log
#SBATCH --error=output/lr_lora_%j.err
#SBATCH --requeue
#
# Combined P1-10 + P2-16: LR sweep + LoRA rank/alpha ablation
# LR:   {1e-4, 2e-4, 5e-4} with r=64, alpha=16 (3 runs)
# LoRA: r∈{16, 32, 128}, alpha∈{r, 2r} with lr=1e-4 (6 runs)
# (r=64/a=16 at lr=1e-4 is the multi-seed baseline — skip it here)
#
# 9 training runs @ ~20min each = ~3 hours
#
# Pairs with job_task_contribution.slurm on the other H200.
#   sbatch training_duke/job_task_contribution.slurm
#   sbatch training_duke/job_lr_lora_sweep.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="meta-llama/Llama-3.1-8B-Instruct"
BASE_OUT_LR="output/ablation_lr"
BASE_OUT_LORA="output/ablation_lora"
SEED=42

echo "=== LR + LoRA Ablation ==="
nvidia-smi
echo "Start: $(date)"

# ─── LR Sweep (r=64, alpha=16) ───────────────────────────────────────────────
for LR in "1e-4" "2e-4" "5e-4"; do
    OUT_DIR="${BASE_OUT_LR}/lr_${LR}"
    mkdir -p "$OUT_DIR"
    echo ""
    echo "=== LR=$LR ==="
    echo "Start: $(date)"

    python -m clarimol train \
        --model "$MODEL" \
        --data-dir data/clarimol \
        --output-dir "$OUT_DIR" \
        --no-unsloth --max-length 512 \
        --batch-size 16 --grad-accum 1 \
        --lr "$LR" --epochs 1 \
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

# ─── LoRA Rank/Alpha Sweep (lr=1e-4) ─────────────────────────────────────────
# Skip r=64/a=16 — covered in LR sweep above
for RANK in 16 32 128; do
    for ALPHA_MULT in 1 2; do
        ALPHA=$((RANK * ALPHA_MULT))
        OUT_DIR="${BASE_OUT_LORA}/r${RANK}_a${ALPHA}"
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
echo "=== LR + LoRA ablation complete at $(date) ==="
echo ""
echo "LR sweep:"
for LR in "1e-4" "2e-4" "5e-4"; do
    echo -n "  lr=$LR: "
    python -c "
import json
with open('${BASE_OUT_LR}/lr_${LR}/results.json') as f:
    d = json.load(f)
accs = [v['accuracy'] for v in d.values()]
print(f'mean={sum(accs)/len(accs):.4f}')
" 2>/dev/null || echo "no results"
done
echo ""
echo "LoRA sweep:"
for RANK in 16 32 128; do
    for ALPHA_MULT in 1 2; do
        ALPHA=$((RANK * ALPHA_MULT))
        echo -n "  r=$RANK a=$ALPHA: "
        python -c "
import json
with open('${BASE_OUT_LORA}/r${RANK}_a${ALPHA}/results.json') as f:
    d = json.load(f)
accs = [v['accuracy'] for v in d.values()]
print(f'mean={sum(accs)/len(accs):.4f}')
" 2>/dev/null || echo "no results"
    done
done
