#!/bin/bash
#SBATCH --job-name=cm_probing
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/probing/extract_%j.log
#SBATCH --error=output/logs/probing/extract_%j.err
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

MODEL_PATH="output/multi_seed/llama-8b/seed_42/final"
DATA_DIR="data/test"
OUTPUT_FILE="output/probing/llama_representations.npz"

echo "Extracting representations from ${MODEL_PATH}"
echo "Test data: ${DATA_DIR}"
echo "Output: ${OUTPUT_FILE}"

python scripts/extract_representations.py \
    --model-path "${MODEL_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output "${OUTPUT_FILE}" \
    --samples-per-task 1000 \
    --seed 42

echo "Representation extraction complete."
