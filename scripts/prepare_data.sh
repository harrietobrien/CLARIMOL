#!/usr/bin/env bash
: <<'COMMENT'
Prepare CLARIMOL dataset from ZINC250K.
    e.g., bash scripts/prepare_data.sh

To limit a run to n molecules, use the following option:
    e.g., bash scripts/prepare_data.sh --max-molecules n
COMMENT

set -euo pipefail
cd "$(dirname "$0")/.."

python -m clarimol prepare \
    --output-dir data/clarimol \
    --keep-n 50000 \
    --subsample middle \
    --trim-fraction 0.15 \
    --seed 42 \
    "$@"