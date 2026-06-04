#!/bin/bash
#SBATCH --job-name=cm_attn_extract
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/probing/attention_%j.log
#SBATCH --error=output/logs/probing/attention_%j.err
#SBATCH --requeue

set -euo pipefail

cd ~/storage/CLARIMOL

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

mkdir -p output/logs/probing
mkdir -p output/probing

ADAPTER_PATH="output/multi_seed/llama-8b/seed_42/final"
DATA_DIR="data/test"
OUTPUT_FILE="output/probing/attention_patterns.npz"

echo "Attention pattern extraction"
echo "Adapter: ${ADAPTER_PATH}"
echo "Test data: ${DATA_DIR}"
echo "Output: ${OUTPUT_FILE}"

python scripts/extract_attention_patterns.py \
    --adapter-path "${ADAPTER_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output "${OUTPUT_FILE}" \
    --samples-per-task 200 \
    --seed 42

echo "Attention pattern extraction complete."
