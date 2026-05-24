#!/bin/bash
# Pull experiment results from DCC to local machine for figure generation
# Usage: bash scripts/pull_results.sh
#
# Uses SSH connection multiplexing: authenticates ONCE, reuses for all transfers.

DCC_HOST="gc237@dcc-login.oit.duke.edu"
DCC_BASE="~/storage/CLARIMOL/output"
LOCAL="output"

# SSH multiplexing — one login for the whole script
SOCK="/tmp/ssh-clarimol-$$"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$SOCK -o ControlPersist=60"

cleanup() {
    ssh -o ControlPath="$SOCK" -O exit "$DCC_HOST" 2>/dev/null
    rm -f "$SOCK"
}
trap cleanup EXIT

echo "=== Connecting to DCC (authenticate once) ==="
ssh $SSH_OPTS -o ControlMaster=yes -fN "$DCC_HOST" || { echo "SSH connection failed"; exit 1; }
echo "Connected."

# Helper: pull a single file, report OK/MISSING
pull() {
    local remote="$1" local_dir="$2" label="$3"
    mkdir -p "$local_dir"
    scp -o ControlPath="$SOCK" "$DCC_HOST:$remote" "$local_dir/" 2>/dev/null \
        && echo "  $label: OK" || echo "  $label: MISSING"
}

echo ""
echo "=== Pulling results from DCC ==="

# Multi-seed summaries
echo "Multi-seed summaries..."
for model in chemllm llama-8b mistral olmo qwen3 qwen; do
    pull "$DCC_BASE/multi_seed/$model/summary.json" "$LOCAL/multi_seed/$model" "$model"
done

# LR ablation
echo "LR ablation..."
for lr in 1e-4 2e-4 5e-4; do
    pull "$DCC_BASE/ablation_lr/lr_$lr/results.json" "$LOCAL/ablation_lr/lr_$lr" "lr=$lr"
done

# Curriculum ablation
echo "Curriculum ablation..."
for order in easy-hard hard-easy random none; do
    pull "$DCC_BASE/ablation_curriculum/$order/results.json" "$LOCAL/ablation_curriculum/$order" "$order"
done

# Dataset size ablation
echo "Dataset size ablation..."
for size in 10000 25000 50000 100000 250000; do
    found=false
    for pattern in "size_$size" "${size}k" "keep_$size"; do
        mkdir -p "$LOCAL/ablation_dataset_size/$pattern"
        if scp -o ControlPath="$SOCK" "$DCC_HOST:$DCC_BASE/ablation_dataset_size/$pattern/results.json" "$LOCAL/ablation_dataset_size/$pattern/" 2>/dev/null; then
            echo "  $pattern: OK"
            found=true
            break
        fi
    done
    $found || echo "  size=$size: MISSING"
done

# Task contribution
echo "Task contribution..."
for task in functional_group ring_counting chain_length canonicalization fragment_assembly; do
    pull "$DCC_BASE/ablation_tasks/single_$task/results.json" "$LOCAL/ablation_tasks/single_$task" "single_$task"
done
pull "$DCC_BASE/ablation_tasks/leave_out_functional_group/results.json" "$LOCAL/ablation_tasks/leave_out_functional_group" "leave_out_fg"

# LoRA sweep
echo "LoRA sweep..."
for r in 16 32 128; do
    for am in 1 2; do
        a=$((r * am))
        pull "$DCC_BASE/ablation_lora/r${r}_a${a}/results.json" "$LOCAL/ablation_lora/r${r}_a${a}" "r=$r a=$a"
    done
done

# COD domain shift
echo "COD domain shift..."
for f in zero_shot_cod zinc_on_cod cod_on_cod cod_on_zinc; do
    pull "$DCC_BASE/cod_domain_shift/$f.json" "$LOCAL/cod_domain_shift" "$f"
done

# COD 82K domain shift
echo "COD 82K domain shift..."
for f in zero_shot_cod zinc_on_cod cod_on_cod cod_on_zinc; do
    pull "$DCC_BASE/cod_domain_shift_82k/$f.json" "$LOCAL/cod_domain_shift_82k" "$f"
done

# QM9 domain shift
echo "QM9 domain shift..."
for f in zero_shot_qm9 zinc_on_qm9 qm9_on_qm9 qm9_on_zinc; do
    pull "$DCC_BASE/qm9_domain_shift/$f.json" "$LOCAL/qm9_domain_shift" "$f"
done

# ChEMBL domain shift
echo "ChEMBL domain shift..."
for f in zero_shot_chembl zinc_on_chembl chembl_on_chembl chembl_on_zinc; do
    pull "$DCC_BASE/chembl_domain_shift/$f.json" "$LOCAL/chembl_domain_shift" "$f"
done

# Downstream
echo "Downstream transfer..."
for task in retrosynthesis forward_reaction_prediction reagent_prediction; do
    pull "$DCC_BASE/downstream_pretrained/$task/results.json" "$LOCAL/downstream_pretrained/$task" "$task"
done

echo ""
echo "=== Done ==="
