#!/bin/bash
#SBATCH --job-name=cm_robust_mis
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=output/logs/robustness/robustness_mistral_%j.log
#SBATCH --error=output/logs/robustness/robustness_mistral_%j.err
#
# SMILES robustness evaluation for Mistral-7B.
# Mirrors the LLaMA-8B robustness evaluation to test whether the
# three-regime robustness pattern generalizes across architectures.
#
# Submit: sbatch scripts/eval_robustness_mistral.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/robustness

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

# Reuse the existing robustness script with Mistral paths by
# creating a temporary copy with substituted model paths.
# The randomized test data is model-independent and can be reused
# if already generated from the LLaMA run.

MISTRAL_MODEL="output/multi_seed/mistral-7b/seed_42/final"
MISTRAL_BASE="mistralai/Mistral-7B-Instruct-v0.3"
OUT_DIR="output/robustness_mistral"
TEST_DIR="data/test"
N_RANDOM=5
MAX_MOLECULES=2000

mkdir -p "$OUT_DIR"

if [ ! -d "$MISTRAL_MODEL" ]; then
    echo "ERROR: Mistral model not found at $MISTRAL_MODEL"
    exit 1
fi

# Always generate fresh data in OUT_DIR to avoid path confusion.
# The randomized SMILES are model-independent but storing them per-model
# prevents cross-directory reference bugs.
RANDOM_DATA_DIR="$OUT_DIR"

echo "=== SMILES Robustness Evaluation: Mistral-7B ==="
nvidia-smi
echo "Start: $(date)"

# Step 1: Generate randomized SMILES test data
if [ ! -f "$OUT_DIR/randomized_data.flag" ]; then
    echo "=== Generating randomized SMILES variants ==="
    python3 << 'PYEOF'
import json
import random
import sys
from pathlib import Path
from rdkit import Chem

sys.path.insert(0, ".")
from clarimol.data.dataset import load_dataset_from_disk
from clarimol.data.sample import Sample

TEST_DIR = "data/test"
OUT_DIR = "output/robustness_mistral"
N_RANDOM = 5
MAX_MOLECULES = 2000
TASKS = ["functional_group", "ring_counting", "chain_length", "canonicalization", "fragment_assembly"]

rng = random.Random(42)
task_samples = load_dataset_from_disk(TEST_DIR)

for task in TASKS:
    samples = task_samples.get(task, [])
    rng_copy = random.Random(42)
    rng_copy.shuffle(samples)
    samples = samples[:MAX_MOLECULES]

    canonical_out = []
    randomized_out = []
    skipped = 0

    for sample in samples:
        smiles = sample.smiles

        # Handle fragment assembly (two fragments separated by ' . ')
        if " . " in smiles:
            parts = smiles.split(" . ")
            canon_parts = []
            rand_parts_list = [[] for _ in range(N_RANDOM)]
            valid = True
            for part in parts:
                mol = Chem.MolFromSmiles(part)
                if mol is None:
                    valid = False
                    break
                canon_parts.append(Chem.MolToSmiles(mol, canonical=True))
                for j in range(N_RANDOM):
                    rand_parts_list[j].append(Chem.MolToSmiles(mol, doRandom=True))
            if not valid:
                skipped += 1
                continue
            canon_smi = " . ".join(canon_parts)
            rand_smiles = [" . ".join(rp) for rp in rand_parts_list]
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                skipped += 1
                continue
            canon_smi = Chem.MolToSmiles(mol, canonical=True)
            rand_smiles = [Chem.MolToSmiles(mol, doRandom=True) for _ in range(N_RANDOM)]

        canonical_out.append({
            "smiles": canon_smi,
            "task": task,
            "question": sample.question,
            "answer": sample.answer,
            "metadata": sample.metadata,
            "molecule_id": len(canonical_out),
        })
        for j, rsmi in enumerate(rand_smiles):
            randomized_out.append({
                "smiles": rsmi,
                "task": task,
                "question": sample.question.replace(smiles, rsmi) if smiles in sample.question else sample.question,
                "answer": sample.answer,
                "metadata": sample.metadata,
                "molecule_id": len(canonical_out) - 1,
                "variant_id": j,
            })

    with open(f"{OUT_DIR}/canonical_{task}.json", "w") as f:
        json.dump(canonical_out, f)
    with open(f"{OUT_DIR}/randomized_{task}.json", "w") as f:
        json.dump(randomized_out, f)
    print(f"  {task}: {len(canonical_out)} canonical, {len(randomized_out)} randomized ({skipped} skipped)")

print("Randomized data generation complete.")
Path(f"{OUT_DIR}/randomized_data.flag").touch()
PYEOF
    echo "=== Data generation done ==="
fi

# Step 2: Evaluate Mistral-trained model on both canonical and randomized
echo "=== Evaluating Mistral-trained model ==="
python3 << 'PYEOF'
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
OUT_DIR = "output/robustness_mistral"
ADAPTER = "output/multi_seed/mistral-7b/seed_42/final"
BATCH_SIZE = 16

print("Loading Mistral-trained model...")
adapter_config = json.load(open(f"{ADAPTER}/adapter_config.json"))
base_name = adapter_config["base_model_name_or_path"]
tokenizer = AutoTokenizer.from_pretrained(ADAPTER, padding_side="left", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(base_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

def run_inference(model, tokenizer, samples, batch_size=16):
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
    return predictions

all_results = {}
for task in TASKS:
    canon_path = f"{OUT_DIR}/canonical_{task}.json"
    rand_path = f"{OUT_DIR}/randomized_{task}.json"
    if not Path(canon_path).exists():
        print(f"Skipping {task}: no data")
        continue

    with open(canon_path) as f:
        canon_data = json.load(f)
    with open(rand_path) as f:
        rand_data = json.load(f)

    canon_samples = [Sample(smiles=d["smiles"], task=task, question=d["question"], answer=d["answer"], metadata=d.get("metadata", {})) for d in canon_data]
    rand_samples = [Sample(smiles=d["smiles"], task=task, question=d["question"], answer=d["answer"], metadata=d.get("metadata", {})) for d in rand_data]

    print(f"  {task}: {len(canon_samples)} canonical, {len(rand_samples)} randomized")

    canon_preds = run_inference(model, tokenizer, canon_samples)
    canon_result = evaluate_parsing(canon_preds, [s.answer for s in canon_samples], task, [s.metadata for s in canon_samples])

    rand_preds = run_inference(model, tokenizer, rand_samples)
    rand_result = evaluate_parsing(rand_preds, [s.answer for s in rand_samples], task, [s.metadata for s in rand_samples])

    # Consistency: for each molecule, check if all N_RANDOM variants are correct
    n_molecules = len(canon_data)
    n_random = 5
    consistent = 0
    for mol_id in range(n_molecules):
        mol_preds = rand_preds[mol_id * n_random:(mol_id + 1) * n_random]
        mol_refs = [rand_samples[mol_id * n_random + k].answer for k in range(n_random)]
        mol_result = evaluate_parsing(mol_preds, mol_refs, task)
        if mol_result.correct == n_random:
            consistent += 1

    all_results[task] = {
        "canonical_accuracy": round(canon_result.accuracy, 4),
        "randomized_accuracy": round(rand_result.accuracy, 4),
        "drop_pp": round((rand_result.accuracy - canon_result.accuracy) * 100, 1),
        "consistency": round(consistent / n_molecules, 4) if n_molecules > 0 else 0.0,
        "n_molecules": n_molecules,
    }
    print(f"    canon={canon_result.accuracy:.4f}, rand={rand_result.accuracy:.4f}, drop={all_results[task]['drop_pp']:.1f}pp, consistency={all_results[task]['consistency']:.1%}")

with open(f"{OUT_DIR}/robustness_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Results saved to {OUT_DIR}/robustness_results.json")
PYEOF

echo "=== Robustness evaluation complete ==="
echo "End: $(date)"
