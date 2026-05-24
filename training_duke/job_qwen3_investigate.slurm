#!/bin/bash
#SBATCH --job-name=cm_qwen3_investigate
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/multi_seed/qwen3_investigate_%j.log
#SBATCH --error=output/logs/multi_seed/qwen3_investigate_%j.err
#SBATCH --requeue
#
# Investigate Qwen3-8B Canon=0.000, Frag=0.000 on seeds 42 and 137.
# Qwen/Qwen3-8B is a BASE model (not instruct) which likely cannot follow
# chat-formatted instructions for generative tasks.
#
# This script:
# 1. Prints 5 sample outputs from canon and frag to diagnose the failure mode
# 2. If outputs are garbage/empty, the model needs the instruct variant
#
# Submit: sbatch training_duke/job_qwen3_investigate.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

echo "=== Qwen3-8B Investigation ==="
nvidia-smi
echo "Start: $(date)"

SEED_DIR="output/multi_seed/qwen3-8b/seed_42"

echo ""
echo "=== Step 1: Check adapter config ==="
cat "$SEED_DIR/final/adapter_config.json" | python3 -m json.tool | head -20

echo ""
echo "=== Step 2: Check what base model the adapter was trained on ==="
python3 -c "
import json
cfg = json.load(open('$SEED_DIR/final/adapter_config.json'))
print(f'Base model: {cfg.get(\"base_model_name_or_path\", \"UNKNOWN\")}')
print(f'Task type: {cfg.get(\"task_type\", \"UNKNOWN\")}')
"

echo ""
echo "=== Step 3: Run eval with --max-samples 20 and verbose output ==="
python -m clarimol evaluate \
    --model-path "$SEED_DIR/final" \
    --data-dir data/test \
    --output-file "$SEED_DIR/results_debug.json" \
    --no-unsloth \
    --batch-size 4 \
    --max-samples 20

echo ""
echo "=== Step 4: Check debug results ==="
python3 -c "
import json
d = json.load(open('$SEED_DIR/results_debug.json'))
for t in ['functional_group','ring_counting','chain_length','canonicalization','fragment_assembly']:
    if t in d:
        print(f'{t}: accuracy={d[t][\"accuracy\"]:.4f}, total={d[t][\"total\"]}')
"

echo ""
echo "=== Complete at $(date) ==="
