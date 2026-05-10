#!/bin/bash
#SBATCH --job-name=cm_baselines
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/baselines_%j.log
#SBATCH --error=output/baselines_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# Baseline evaluations (P0-4):
#   1. Zero-shot eval of untrained LLaMA-8B and Qwen-7B on parsing tasks
#   2. Downstream-only training (no parsing pre-training) — proves pre-training helps
#   3. Data contamination check
#
# Can run on either H200. Pairs well with job_curriculum_ablation.slurm.
#   sbatch training_duke/job_baselines.slurm
#   sbatch training_duke/job_curriculum_ablation.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

BASE_OUT="output/baselines"
mkdir -p "$BASE_OUT"

echo "=== CLARIMOL Baselines ==="
nvidia-smi
echo "Start: $(date)"

# ─── 0. Data contamination check ─────────────────────────────────────────────
echo ""
echo "=== Data Contamination Check ==="
python -m clarimol check-contamination \
    --train-dir data/clarimol \
    --test-dir data/test \
    --output-file "$BASE_OUT/contamination.json"

# ─── 1. Zero-shot parsing evaluation ─────────────────────────────────────────
ZERO_SHOT_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

for MODEL in "${ZERO_SHOT_MODELS[@]}"; do
    SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    OUT="$BASE_OUT/zero_shot_${SHORT}"
    mkdir -p "$OUT"
    echo ""
    echo "=== Zero-shot: $MODEL ==="
    echo "Start: $(date)"

    python -m clarimol evaluate \
        --model-path "$MODEL" \
        --data-dir data/test \
        --output-file "$OUT/results.json" \
        --batch-size 8 \
        --no-unsloth \
        2>&1 | tee "$OUT/eval.log"

    echo "Done: $(date)"
done

# ─── 2. Downstream-only (no parsing pre-training) ────────────────────────────
# Train directly on downstream tasks from base LLaMA — ablation that parsing
# pre-training actually helps
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DS_TASKS=("retrosynthesis" "reagent_prediction" "forward_reaction_prediction")

for TASK in "${DS_TASKS[@]}"; do
    OUT="$BASE_OUT/downstream_only_${TASK}"
    mkdir -p "$OUT"
    echo ""
    echo "=== Downstream-only: $TASK (no parsing pre-training) ==="
    echo "Start: $(date)"

    python -m clarimol downstream-train \
        --model "$BASE_MODEL" \
        --task "$TASK" \
        --data-dir data/mol_instructions \
        --output-dir "$OUT" \
        --batch-size 16 \
        --grad-accum 1 \
        --lr 5e-4 \
        --epochs 1 \
        --no-unsloth \
        --no-wandb \
        --bf16 \
        --no-4bit \
        --seed 42

    python -m clarimol downstream-eval \
        --model-path "$OUT/final" \
        --data-dir data/mol_instructions \
        --tasks "$TASK" \
        --output-file "$OUT/results.json" \
        --no-unsloth

    echo "Done: $(date)"
done

echo ""
echo "=== All baselines complete at $(date) ==="
echo "Results in $BASE_OUT/"
