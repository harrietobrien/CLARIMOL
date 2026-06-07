#!/bin/bash
#SBATCH --job-name=cm_persample_g2
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/persample/persample_g2_%j.log
#SBATCH --error=output/logs/persample/persample_g2_%j.err
#SBATCH --requeue
#
# GPU 2: Per-sample eval for Qwen-7B, Qwen3-8B + ZINC-trained -> ChEMBL transfer
#
# Submit: sbatch scripts/eval_persample_gpu2.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/persample

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== Per-sample Eval (GPU 2): Qwen-7B, Qwen3-8B + ChEMBL ==="
nvidia-smi
echo "Start: $(date)"

# Part 1: Per-sample eval for Qwen models
for MODEL in qwen-7b qwen3-8b; do
    CHECKPOINT="output/multi_seed/${MODEL}/seed_42/final"
    OUTDIR="output/multi_seed/${MODEL}/seed_42"
    PRED_FILE="${OUTDIR}/predictions.jsonl"

    if [ -f "$PRED_FILE" ]; then
        echo "SKIP: predictions exist for ${MODEL}"
        continue
    fi

    if [ ! -d "$CHECKPOINT" ]; then
        echo "SKIP: checkpoint missing for ${MODEL}"
        continue
    fi

    echo ""
    echo "=== ${MODEL} ($(date)) ==="
    python -m clarimol evaluate \
        --model-path "$CHECKPOINT" \
        --data-dir data/test \
        --output-file "${OUTDIR}/results_persample.json" \
        --save-predictions "$PRED_FILE" \
        --no-unsloth \
        --batch-size 16

    echo "${MODEL} done: $(date)"
done

# Part 2: ZINC-trained LLaMA -> ChEMBL transfer evaluation
# chembl_250k/ has pre-prepared 5-task data (250K samples)
# Use a subsample for eval (same scale as other domain shift evals)
CHEMBL_DATA="data/chembl_250k"
ZINC_MODEL="output/multi_seed/llama-8b/seed_42/final"
CHEMBL_OUT="output/chembl_domain_shift/zinc_on_chembl.json"
CHEMBL_PRED="output/chembl_domain_shift/zinc_on_chembl_predictions.jsonl"

mkdir -p output/chembl_domain_shift

if [ -d "$CHEMBL_DATA" ] && [ -d "$ZINC_MODEL" ] && [ ! -f "$CHEMBL_OUT" ]; then
    echo ""
    echo "=== ZINC-trained LLaMA -> ChEMBL ($(date)) ==="
    python -m clarimol evaluate \
        --model-path "$ZINC_MODEL" \
        --data-dir "$CHEMBL_DATA" \
        --output-file "$CHEMBL_OUT" \
        --save-predictions "$CHEMBL_PRED" \
        --no-unsloth \
        --max-samples 10000 \
        --batch-size 16
    echo "ChEMBL eval done: $(date)"
else
    echo "SKIP: ChEMBL eval already exists or data/model missing"
    echo "  CHEMBL_DATA=$CHEMBL_DATA exists: $([ -d $CHEMBL_DATA ] && echo yes || echo no)"
    echo "  ZINC_MODEL=$ZINC_MODEL exists: $([ -d $ZINC_MODEL ] && echo yes || echo no)"
    echo "  CHEMBL_OUT=$CHEMBL_OUT exists: $([ -f $CHEMBL_OUT ] && echo yes || echo no)"
fi

echo ""
echo "=== GPU 2 complete at $(date) ==="
