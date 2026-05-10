#!/bin/bash
#SBATCH --job-name=clarimol_h200_2gpu
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=12:00:00
#SBATCH --output=output/h200_2gpu_%j.log
#SBATCH --error=output/h200_2gpu_%j.err
#SBATCH --requeue
#
# Dual H200: Qwen 2.5-7B on full 250K ZINC dataset across 2 GPUs
# accelerate handles data-parallel sharding via DDP
# ~2x throughput vs single GPU → halved wall time
#
# scavenger-h200 jobs may be preempted and requeued.
# Checkpoints saved every 500 steps; use --resume to continue.
#
# Submit:     sbatch training_duke/job_h200_250k_dual.slurm
#
# Docs: https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/h200/

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

OUTPUT_DIR=output/parsing_pretrain_h200
mkdir -p "$OUTPUT_DIR"

echo "=== CLARIMOL H200 dual-GPU (250K) ==="
nvidia-smi
echo "Start: $(date)"

# Auto-resume: if checkpoints exist in output dir, resume from last one
RESUME_FLAG=""
if ls "$OUTPUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
    echo "Checkpoint(s) detected — resuming training"
    RESUME_FLAG="--resume"
fi

accelerate launch \
    --num_processes 2 \
    --multi_gpu \
    -m clarimol train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data-dir data/clarimol \
    --output-dir "$OUTPUT_DIR" \
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
    --seed 42 \
    $RESUME_FLAG

echo "Training complete at $(date)"

# Evaluate on test set
bash scripts/evaluate_all.sh "$OUTPUT_DIR/final" --max-samples 2000
