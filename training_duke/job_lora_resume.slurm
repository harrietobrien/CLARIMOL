#!/bin/bash
#SBATCH --job-name=cm_lora_res
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/lora_resume_%j.log
#SBATCH --error=output/lora_resume_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# Resume LoRA ablation: only runs configs missing results.json
# Remaining: r32_a64, r128_a128, r128_a256 (~3 runs @ ~20min = ~1h)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="meta-llama/Llama-3.1-8B-Instruct"
BASE_OUT="output/ablation_lora"
SEED=42

echo "=== LoRA Ablation (resume) ==="
nvidia-smi
echo "Start: $(date)"

for RANK in 16 32 128; do
    for ALPHA_MULT in 1 2; do
        ALPHA=$((RANK * ALPHA_MULT))
        OUT_DIR="${BASE_OUT}/r${RANK}_a${ALPHA}"

        if [ -f "$OUT_DIR/results.json" ]; then
            echo "SKIP r=$RANK a=$ALPHA (results.json exists)"
            continue
        fi

        mkdir -p "$OUT_DIR"
        echo ""
        echo "=== LoRA r=$RANK, alpha=$ALPHA ==="
        echo "Start: $(date)"

        # Check for checkpoint to resume from
        RESUME_FLAG=""
        LATEST_CKPT=$(ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        if [ -n "$LATEST_CKPT" ]; then
            echo "Resuming from $LATEST_CKPT"
            RESUME_FLAG="--resume-from-checkpoint $LATEST_CKPT"
        fi

        python -m clarimol train \
            --model "$MODEL" \
            --data-dir data/clarimol \
            --output-dir "$OUT_DIR" \
            --no-unsloth --max-length 512 \
            --batch-size 16 --grad-accum 1 \
            --lr 1e-4 --epochs 1 \
            --lora-r "$RANK" --lora-alpha "$ALPHA" \
            --bf16 --no-4bit --no-wandb \
            --save-steps 500 --seed "$SEED" \
            $RESUME_FLAG

        python -m clarimol evaluate \
            --model-path "$OUT_DIR/final" \
            --data-dir data/test \
            --output-file "$OUT_DIR/results.json" \
            --no-unsloth --batch-size 16

        echo "Done: $(date)"
    done
done

echo ""
echo "=== LoRA ablation (resume) complete at $(date) ==="
echo ""
echo "LoRA sweep results:"
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
