#!/bin/bash
#SBATCH --job-name=cm_smiles_aug
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=output/logs/augmentation/augmentation_%j.log
#SBATCH --error=output/logs/augmentation/augmentation_%j.err
#
# SMILES Augmentation Experiment
#
# Tests whether training with randomized (non-canonical) SMILES fixes the
# robustness collapse found in the paper: FA drops -87.8pp and CA drops
# -55.5pp under SMILES randomization, while FG is robust at -1.8pp.
#
# Four training conditions x 3 seeds = 12 runs on Mistral-7B:
#   augmented_canonical  -- control (original data unchanged)
#   augmented_50_50      -- 50% canonical, 50% randomized
#   augmented_random     -- 100% randomized SMILES
#   augmented_curriculum -- first half canonical, second half randomized
#
# Each run is followed by evaluation on canonical test data and robustness
# evaluation on randomized test data (5 random SMILES per molecule, 200
# molecules per task).
#
# Submit: sbatch scripts/train_smiles_augmentation.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/augmentation

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null || hf auth token 2>/dev/null || true)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== SMILES Augmentation Experiment ==="
nvidia-smi
echo "Start: $(date)"

MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_KEY="mistral-7b"
CONDITIONS=("augmented_canonical" "augmented_50_50" "augmented_random" "augmented_curriculum")
SEEDS=(42 137 2024)
BASE_OUT="output/augmentation"
TEST_DIR="data/test"
ROBUSTNESS_DIR="output/augmentation/robustness_test_data"
N_RANDOM=5
MAX_MOLECULES=200

# Step 0: Generate randomized test data for robustness evaluation (shared
# across all conditions). Only generate once.
if [ ! -f "$ROBUSTNESS_DIR/generation_complete.flag" ]; then
    echo "=== Generating randomized test data for robustness evaluation ==="
    mkdir -p "$ROBUSTNESS_DIR"
    python3 << 'PYEOF'
import json
import random
import sys
from pathlib import Path
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
sys.path.insert(0, ".")

TEST_DIR = "data/test"
OUT_DIR = "output/augmentation/robustness_test_data"
N_RANDOM = 5
MAX_MOLECULES = 200
TASKS = ["functional_group", "ring_counting", "chain_length", "canonicalization", "fragment_assembly"]

rng = random.Random(42)

for task in TASKS:
    task_file = Path(TEST_DIR) / f"{task}.json"
    if not task_file.exists():
        print(f"  {task}: test file not found, skipping")
        continue

    with open(task_file) as f:
        all_samples = json.load(f)

    # Deterministic subsample
    rng_sub = random.Random(42)
    rng_sub.shuffle(all_samples)
    samples = all_samples[:MAX_MOLECULES]

    canonical_out = []
    randomized_out = []
    skipped = 0

    for sample in samples:
        smiles = sample.get("smiles", "")
        if not smiles:
            skipped += 1
            continue

        is_fragment = " . " in smiles

        if is_fragment:
            parts = smiles.split(" . ")
            mols = []
            valid = True
            for part in parts:
                mol = Chem.MolFromSmiles(part)
                if mol is None:
                    valid = False
                    break
                mols.append(mol)
            if not valid:
                skipped += 1
                continue
            canon_smi = " . ".join(Chem.MolToSmiles(m, canonical=True) for m in mols)
            rand_smiles = []
            for _ in range(N_RANDOM * 10):
                if len(rand_smiles) >= N_RANDOM:
                    break
                try:
                    rp = [Chem.MolToSmiles(m, doRandom=True, canonical=False) for m in mols]
                    combined = " . ".join(rp)
                    if combined not in rand_smiles:
                        rand_smiles.append(combined)
                except Exception:
                    continue
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                skipped += 1
                continue
            canon_smi = Chem.MolToSmiles(mol, canonical=True)
            rand_smiles = []
            for _ in range(N_RANDOM * 10):
                if len(rand_smiles) >= N_RANDOM:
                    break
                try:
                    r = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                    if r and r not in rand_smiles:
                        rand_smiles.append(r)
                except Exception:
                    continue

        if len(rand_smiles) < 1:
            skipped += 1
            continue

        mol_id = len(canonical_out)

        canonical_out.append({
            "smiles": canon_smi,
            "task": sample.get("task", task),
            "question": sample.get("question", ""),
            "answer": sample.get("answer", ""),
            "metadata": sample.get("metadata", {}),
            "difficulty": sample.get("difficulty", 0),
            "molecule_id": mol_id,
        })

        for j, rsmi in enumerate(rand_smiles):
            q = sample.get("question", "")
            if smiles in q:
                q = q.replace(smiles, rsmi)
            randomized_out.append({
                "smiles": rsmi,
                "task": sample.get("task", task),
                "question": q,
                "answer": sample.get("answer", ""),
                "metadata": sample.get("metadata", {}),
                "difficulty": sample.get("difficulty", 0),
                "molecule_id": mol_id,
                "variant_id": j,
            })

    with open(f"{OUT_DIR}/canonical_{task}.json", "w") as f:
        json.dump(canonical_out, f)
    with open(f"{OUT_DIR}/randomized_{task}.json", "w") as f:
        json.dump(randomized_out, f)
    print(f"  {task}: {len(canonical_out)} canonical, {len(randomized_out)} randomized ({skipped} skipped)")

Path(f"{OUT_DIR}/generation_complete.flag").touch()
print("Randomized test data generation complete.")
PYEOF
    echo "=== Robustness test data ready ==="
fi

# Step 1: Train and evaluate each condition x seed
for CONDITION in "${CONDITIONS[@]}"; do
    DATA_DIR="data/${CONDITION}"

    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: augmented data not found at $DATA_DIR"
        echo "       Run 'python scripts/prepare_augmented_data.py' first."
        exit 1
    fi

    for SEED in "${SEEDS[@]}"; do
        SEED_DIR="${BASE_OUT}/${CONDITION}/seed_${SEED}"
        mkdir -p "$SEED_DIR"

        # Skip if both results already exist
        if [ -f "$SEED_DIR/results.json" ] && [ -f "$SEED_DIR/robustness_results.json" ]; then
            echo "SKIP: ${CONDITION}/seed_${SEED} (results + robustness exist)"
            continue
        fi

        echo ""
        echo "=== ${CONDITION} / seed_${SEED} ($(date)) ==="

        # Train (skip if final model already exists)
        if [ ! -d "$SEED_DIR/final" ]; then
            RESUME_FLAG=""
            if ls "$SEED_DIR"/checkpoint-* 1>/dev/null 2>&1; then
                RESUME_FLAG="--resume"
            fi
            python -m clarimol train \
                --model "$MODEL_ID" \
                --data-dir "$DATA_DIR" \
                --output-dir "$SEED_DIR" \
                --no-unsloth \
                --max-length 512 \
                --batch-size 16 \
                --grad-accum 1 \
                --lr 1e-4 \
                --epochs 1 \
                --lora-r 64 \
                --lora-alpha 16 \
                --bf16 \
                --no-4bit \
                --no-wandb \
                --save-steps 500 \
                --seed $SEED \
                $RESUME_FLAG
        fi

        # Evaluate on canonical test data
        if [ -d "$SEED_DIR/final" ] && [ ! -f "$SEED_DIR/results.json" ]; then
            echo "--- Evaluating on canonical test data ---"
            python -m clarimol evaluate \
                --model-path "$SEED_DIR/final" \
                --data-dir "$TEST_DIR" \
                --output-file "$SEED_DIR/results.json" \
                --no-unsloth \
                --batch-size 16
        fi

        # Robustness evaluation on randomized test data
        if [ -d "$SEED_DIR/final" ] && [ ! -f "$SEED_DIR/robustness_results.json" ]; then
            echo "--- Robustness evaluation (randomized SMILES) ---"
            python3 << PYEOF
import json
import random
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, ".")
from clarimol.data.sample import Sample
from clarimol.tasks.prompts import build_messages
from clarimol.eval.metrics import evaluate_parsing

TASKS = ["functional_group", "ring_counting", "chain_length", "canonicalization", "fragment_assembly"]
ROBUSTNESS_DIR = "output/augmentation/robustness_test_data"
SEED_DIR = "$SEED_DIR"
N_RANDOM = $N_RANDOM
BATCH_SIZE = 16

adapter_path = f"{SEED_DIR}/final"
adapter_config = json.load(open(f"{adapter_path}/adapter_config.json"))
base_name = adapter_config["base_model_name_or_path"]

print(f"Loading model from {adapter_path}...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path, padding_side="left", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(base_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()


def run_inference(model, tokenizer, samples, batch_size=BATCH_SIZE):
    rng = random.Random(0)
    predictions = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        prompts = []
        for s in batch:
            msgs = build_messages(s, rng=rng, use_system_prompt=True)
            msgs_no_answer = [m for m in msgs if m["role"] != "assistant"]
            try:
                prompt = tokenizer.apply_chat_template(msgs_no_answer, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in msgs_no_answer) + "\n<|assistant|>\n"
            prompts.append(prompt)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1920).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            text = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            predictions.append(text)
        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"    processed {min(i + batch_size, len(samples))}/{len(samples)}")
    return predictions


all_results = {}
for task in TASKS:
    canon_path = f"{ROBUSTNESS_DIR}/canonical_{task}.json"
    rand_path = f"{ROBUSTNESS_DIR}/randomized_{task}.json"
    if not Path(canon_path).exists():
        print(f"  {task}: data files missing, skipping")
        continue

    with open(canon_path) as f:
        canon_data = json.load(f)
    with open(rand_path) as f:
        rand_data = json.load(f)

    canon_samples = [
        Sample(smiles=d["smiles"], task=task, question=d["question"],
               answer=d["answer"], metadata=d.get("metadata", {}))
        for d in canon_data
    ]
    rand_samples = [
        Sample(smiles=d["smiles"], task=task, question=d["question"],
               answer=d["answer"], metadata=d.get("metadata", {}))
        for d in rand_data
    ]

    print(f"  {task}: {len(canon_samples)} canonical, {len(rand_samples)} randomized")

    canon_preds = run_inference(model, tokenizer, canon_samples)
    canon_result = evaluate_parsing(
        canon_preds, [s.answer for s in canon_samples], task,
        [s.metadata for s in canon_samples],
    )

    rand_preds = run_inference(model, tokenizer, rand_samples)
    rand_result = evaluate_parsing(
        rand_preds, [s.answer for s in rand_samples], task,
        [s.metadata for s in rand_samples],
    )

    # Consistency: for each molecule, check if all N_RANDOM variants are correct
    n_molecules = len(canon_data)
    consistent = 0
    for mol_id in range(n_molecules):
        mol_preds = rand_preds[mol_id * N_RANDOM:(mol_id + 1) * N_RANDOM]
        mol_refs = [rand_samples[mol_id * N_RANDOM + k].answer for k in range(min(N_RANDOM, len(rand_samples) - mol_id * N_RANDOM))]
        if len(mol_preds) == 0:
            continue
        mol_result = evaluate_parsing(mol_preds, mol_refs, task)
        if mol_result.correct == len(mol_preds):
            consistent += 1

    all_results[task] = {
        "canonical_accuracy": round(canon_result.accuracy, 4),
        "randomized_accuracy": round(rand_result.accuracy, 4),
        "drop_pp": round((rand_result.accuracy - canon_result.accuracy) * 100, 1),
        "consistency": round(consistent / n_molecules, 4) if n_molecules > 0 else 0.0,
        "n_molecules": n_molecules,
        "n_canonical": len(canon_samples),
        "n_randomized": len(rand_samples),
    }
    print(f"    canon={canon_result.accuracy:.4f}  rand={rand_result.accuracy:.4f}  "
          f"drop={all_results[task]['drop_pp']:.1f}pp  "
          f"consistency={all_results[task]['consistency']:.1%}")

with open(f"{SEED_DIR}/robustness_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Robustness results saved to {SEED_DIR}/robustness_results.json")

# Free GPU memory before next run
del model, base_model
torch.cuda.empty_cache()
PYEOF
        fi

        # Print results summary for this seed
        if [ -f "$SEED_DIR/results.json" ]; then
            echo "--- ${CONDITION}/seed_${SEED} canonical results ---"
            python3 -c "
import json
d = json.load(open('$SEED_DIR/results.json'))
accs = {k: round(v['accuracy'], 4) for k, v in d.items() if 'accuracy' in v}
print(accs)
print(f'mean={sum(accs.values())/len(accs):.4f}')
"
        fi
        if [ -f "$SEED_DIR/robustness_results.json" ]; then
            echo "--- ${CONDITION}/seed_${SEED} robustness results ---"
            python3 -c "
import json
d = json.load(open('$SEED_DIR/robustness_results.json'))
for task, v in d.items():
    print(f'  {task}: canon={v[\"canonical_accuracy\"]:.4f} rand={v[\"randomized_accuracy\"]:.4f} drop={v[\"drop_pp\"]:.1f}pp consistency={v[\"consistency\"]:.1%}')
"
        fi
    done
done

# Step 2: Aggregate results across seeds and conditions
echo ""
echo "=== Aggregating results ==="
python3 << 'PYEOF'
import json
import statistics
from pathlib import Path

BASE_OUT = Path("output/augmentation")
CONDITIONS = ["augmented_canonical", "augmented_50_50", "augmented_random", "augmented_curriculum"]
SEEDS = [42, 137, 2024]
TASKS = ["functional_group", "ring_counting", "chain_length", "canonicalization", "fragment_assembly"]

summary = {}

for condition in CONDITIONS:
    condition_data = {"canonical": {}, "robustness": {}}

    for seed in SEEDS:
        seed_dir = BASE_OUT / condition / f"seed_{seed}"

        # Canonical results
        results_path = seed_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            for task, vals in results.items():
                condition_data["canonical"].setdefault(task, []).append(vals.get("accuracy", 0))

        # Robustness results
        rob_path = seed_dir / "robustness_results.json"
        if rob_path.exists():
            with open(rob_path) as f:
                rob = json.load(f)
            for task, vals in rob.items():
                for metric in ["canonical_accuracy", "randomized_accuracy", "drop_pp", "consistency"]:
                    key = f"{task}_{metric}"
                    condition_data["robustness"].setdefault(key, []).append(vals.get(metric, 0))

    summary[condition] = condition_data

# Print canonical accuracy table
print()
print(f"{'Condition':<25s}", end="")
for task in TASKS:
    print(f"  {task[:8]:>10s}", end="")
print(f"  {'mean':>10s}")
print("-" * (25 + 11 * (len(TASKS) + 1)))

for condition in CONDITIONS:
    accs = summary[condition]["canonical"]
    print(f"{condition:<25s}", end="")
    task_means = []
    for task in TASKS:
        vals = accs.get(task, [])
        if vals:
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f"  {m:.3f}±{s:.3f}", end="")
            task_means.append(m)
        else:
            print(f"  {'N/A':>10s}", end="")
    if task_means:
        print(f"  {statistics.mean(task_means):.3f}")
    else:
        print()

# Print robustness drop table
print()
print("Robustness: accuracy drop (pp) under SMILES randomization")
print(f"{'Condition':<25s}", end="")
for task in TASKS:
    print(f"  {task[:8]:>10s}", end="")
print()
print("-" * (25 + 11 * len(TASKS)))

for condition in CONDITIONS:
    rob = summary[condition]["robustness"]
    print(f"{condition:<25s}", end="")
    for task in TASKS:
        key = f"{task}_drop_pp"
        vals = rob.get(key, [])
        if vals:
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f"  {m:+.1f}±{s:.1f}", end="")
        else:
            print(f"  {'N/A':>10s}", end="")
    print()

# Print consistency table
print()
print("Consistency: fraction of molecules where all 5 random SMILES variants are correct")
print(f"{'Condition':<25s}", end="")
for task in TASKS:
    print(f"  {task[:8]:>10s}", end="")
print()
print("-" * (25 + 11 * len(TASKS)))

for condition in CONDITIONS:
    rob = summary[condition]["robustness"]
    print(f"{condition:<25s}", end="")
    for task in TASKS:
        key = f"{task}_consistency"
        vals = rob.get(key, [])
        if vals:
            m = statistics.mean(vals)
            print(f"  {m:.3f}", end="")
        else:
            print(f"  {'N/A':>10s}", end="")
    print()

# Save full summary
summary_path = BASE_OUT / "augmentation_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, default=list)
print(f"\nFull summary saved to {summary_path}")
PYEOF

echo ""
echo "=== SMILES Augmentation Experiment Complete ==="
echo "End: $(date)"
