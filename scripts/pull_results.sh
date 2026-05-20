#!/bin/bash
# Pull experiment results from DCC to local machine for figure generation
# Usage: bash scripts/pull_results.sh

DCC="gc237@dcc-login-03:~/storage/CLARIMOL/output"
LOCAL="output"

mkdir -p "$LOCAL"

echo "=== Pulling results from DCC ==="

# Multi-seed summaries
echo "Multi-seed summaries..."
for model in chemllm llama-8b mistral olmo qwen3 qwen; do
    mkdir -p "$LOCAL/multi_seed/$model"
    scp "$DCC/multi_seed/$model/summary.json" "$LOCAL/multi_seed/$model/" 2>/dev/null && echo "  $model: OK" || echo "  $model: MISSING"
done

# LR ablation
echo "LR ablation..."
for lr in 1e-4 2e-4 5e-4; do
    mkdir -p "$LOCAL/ablation_lr/lr_$lr"
    scp "$DCC/ablation_lr/lr_$lr/results.json" "$LOCAL/ablation_lr/lr_$lr/" 2>/dev/null && echo "  lr=$lr: OK" || echo "  lr=$lr: MISSING"
done

# Curriculum ablation
echo "Curriculum ablation..."
for order in easy-hard hard-easy random none; do
    mkdir -p "$LOCAL/ablation_curriculum/$order"
    scp "$DCC/ablation_curriculum/$order/results.json" "$LOCAL/ablation_curriculum/$order/" 2>/dev/null && echo "  $order: OK" || echo "  $order: MISSING"
done

# Dataset size ablation
echo "Dataset size ablation..."
for size in 10000 25000 50000 100000 250000; do
    for pattern in "size_$size" "${size}k" "keep_$size"; do
        mkdir -p "$LOCAL/ablation_dataset_size/$pattern"
        scp "$DCC/ablation_dataset_size/$pattern/results.json" "$LOCAL/ablation_dataset_size/$pattern/" 2>/dev/null && echo "  $pattern: OK" && break || true
    done
done

# Task contribution
echo "Task contribution..."
for task in functional_group ring_counting chain_length canonicalization fragment_assembly; do
    mkdir -p "$LOCAL/ablation_tasks/single_$task"
    scp "$DCC/ablation_tasks/single_$task/results.json" "$LOCAL/ablation_tasks/single_$task/" 2>/dev/null && echo "  single_$task: OK" || echo "  single_$task: MISSING"
done
mkdir -p "$LOCAL/ablation_tasks/leave_out_functional_group"
scp "$DCC/ablation_tasks/leave_out_functional_group/results.json" "$LOCAL/ablation_tasks/leave_out_functional_group/" 2>/dev/null && echo "  leave_out_fg: OK" || echo "  leave_out_fg: MISSING"

# LoRA sweep
echo "LoRA sweep..."
for r in 16 32 128; do
    for am in 1 2; do
        a=$((r * am))
        mkdir -p "$LOCAL/ablation_lora/r${r}_a${a}"
        scp "$DCC/ablation_lora/r${r}_a${a}/results.json" "$LOCAL/ablation_lora/r${r}_a${a}/" 2>/dev/null && echo "  r=$r a=$a: OK" || echo "  r=$r a=$a: MISSING"
    done
done

# COD domain shift
echo "COD domain shift..."
mkdir -p "$LOCAL/cod_domain_shift"
for f in zero_shot_cod zinc_on_cod cod_on_cod cod_on_zinc; do
    scp "$DCC/cod_domain_shift/$f.json" "$LOCAL/cod_domain_shift/" 2>/dev/null && echo "  $f: OK" || echo "  $f: MISSING"
done

# COD 82K domain shift
echo "COD 82K domain shift..."
mkdir -p "$LOCAL/cod_domain_shift_82k"
for f in zero_shot_cod zinc_on_cod cod_on_cod cod_on_zinc; do
    scp "$DCC/cod_domain_shift_82k/$f.json" "$LOCAL/cod_domain_shift_82k/" 2>/dev/null && echo "  $f: OK" || echo "  $f: MISSING"
done

# QM9 domain shift
echo "QM9 domain shift..."
mkdir -p "$LOCAL/qm9_domain_shift"
for f in zero_shot_qm9 zinc_on_qm9 qm9_on_qm9 qm9_on_zinc; do
    scp "$DCC/qm9_domain_shift/$f.json" "$LOCAL/qm9_domain_shift/" 2>/dev/null && echo "  $f: OK" || echo "  $f: MISSING"
done

# ChEMBL domain shift
echo "ChEMBL domain shift..."
mkdir -p "$LOCAL/chembl_domain_shift"
for f in zero_shot_chembl zinc_on_chembl chembl_on_chembl chembl_on_zinc; do
    scp "$DCC/chembl_domain_shift/$f.json" "$LOCAL/chembl_domain_shift/" 2>/dev/null && echo "  $f: OK" || echo "  $f: MISSING"
done

# Downstream
echo "Downstream transfer..."
for task in retrosynthesis forward_reaction_prediction reagent_prediction; do
    mkdir -p "$LOCAL/downstream_pretrained/$task"
    scp "$DCC/downstream_pretrained/$task/results.json" "$LOCAL/downstream_pretrained/$task/" 2>/dev/null && echo "  $task: OK" || echo "  $task: MISSING"
done

echo ""
echo "=== Done ==="
