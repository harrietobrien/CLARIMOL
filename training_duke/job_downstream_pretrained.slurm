#!/bin/bash
#SBATCH --job-name=cm_ds_pre
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/ds_pretrained_%j.log
#SBATCH --error=output/ds_pretrained_%j.err
#SBATCH --requeue
#
# Downstream fine-tuning FROM a parsing-pretrained model
# This is the "with pre-training" arm of the ablation.
# Compare results against job_baselines.slurm (downstream-only, no pre-training).
#
# PREREQUISITE: run job_multi_seed_llama.slurm first (needs seed_42 model)
#
# Can pair with job_cod_domain_shift.slurm on the other H200.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

PRETRAINED="output/multi_seed/llama-8b/seed_42/final"
BASE_OUT="output/downstream_pretrained"
DS_TASKS=("retrosynthesis" "reagent_prediction" "forward_reaction_prediction")

echo "=== Downstream with Parsing Pre-training ==="
nvidia-smi
echo "Start: $(date)"

if [ ! -d "$PRETRAINED" ]; then
    echo "ERROR: Pre-trained model not found at $PRETRAINED"
    echo "Run job_multi_seed_llama.slurm first."
    exit 1
fi

for TASK in "${DS_TASKS[@]}"; do
    OUT_DIR="${BASE_OUT}/${TASK}"
    mkdir -p "$OUT_DIR"
    echo ""
    echo "=== $TASK (with parsing pre-training) ==="
    echo "Start: $(date)"

    python -m clarimol downstream-train \
        --model "$PRETRAINED" \
        --task "$TASK" \
        --data-dir data/mol_instructions \
        --output-dir "$OUT_DIR" \
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
        --model-path "$OUT_DIR/final" \
        --data-dir data/mol_instructions \
        --tasks "$TASK" \
        --output-file "$OUT_DIR/results.json" \
        --no-unsloth

    echo "Done: $(date)"
done

echo ""
echo "=== Downstream (pretrained) complete at $(date) ==="
echo ""
echo "Compare these with output/baselines/downstream_only_*/ to measure"
echo "the impact of parsing pre-training on downstream tasks."
echo ""
for TASK in "${DS_TASKS[@]}"; do
    echo -n "  WITH pre-training ($TASK): "
    python -c "
import json
with open('${BASE_OUT}/${TASK}/results.json') as f:
    d = json.load(f)
for k,v in d.items():
    print(f'exact={v[\"exact_match\"]:.3f}  bleu={v[\"bleu\"]:.3f}  validity={v[\"validity\"]:.3f}')
" 2>/dev/null || echo "no results"
done
