#!/bin/bash
# Safely delete intermediate checkpoints where training is complete.
# A checkpoint is safe to delete when its parent directory contains
# both a final/ directory and a results.json file.
#
# Usage:
#   bash scripts/cleanup_checkpoints.sh          # dry run (default)
#   bash scripts/cleanup_checkpoints.sh --delete  # actually delete

set -euo pipefail
cd ~/storage/CLARIMOL

MODE="${1:-dry}"
TOTAL_BYTES=0
COUNT=0

echo "=== Checkpoint Cleanup ==="
echo "Scanning output/ for deletable checkpoints..."
echo ""

while IFS= read -r ckpt_dir; do
    parent=$(dirname "$ckpt_dir")
    if [ -d "$parent/final" ] && [ -f "$parent/results.json" ]; then
        size=$(du -sh "$ckpt_dir" 2>/dev/null | cut -f1)
        size_bytes=$(du -sb "$ckpt_dir" 2>/dev/null | cut -f1)
        TOTAL_BYTES=$((TOTAL_BYTES + size_bytes))
        COUNT=$((COUNT + 1))
        if [ "$MODE" = "--delete" ]; then
            echo "DELETING: $ckpt_dir ($size)"
            rm -rf "$ckpt_dir"
        else
            echo "DELETABLE: $ckpt_dir ($size)"
        fi
    fi
done < <(find ~/storage/CLARIMOL/output -name "checkpoint-*" -type d 2>/dev/null | sort)

TOTAL_GB=$(echo "scale=1; $TOTAL_BYTES / 1073741824" | bc)

echo ""
echo "=== Summary ==="
echo "Deletable checkpoints: $COUNT"
echo "Total space: ${TOTAL_GB} GB"

if [ "$MODE" != "--delete" ]; then
    echo ""
    echo "This was a DRY RUN. To actually delete, run:"
    echo "  bash scripts/cleanup_checkpoints.sh --delete"
fi
