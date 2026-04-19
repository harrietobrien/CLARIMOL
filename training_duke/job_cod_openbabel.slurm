#!/bin/bash
#SBATCH --job-name=clarimol_codob
#SBATCH --partition=scavenger-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --time=1-00:00:00
#SBATCH --output=output/duke_codob_%j.log
#SBATCH --error=output/duke_codob_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

mkdir -p output data/cod_cif

echo "CLARIMOL: COD via OpenBabel CIF->SMILES pipeline"
nvidia-smi
echo "Start: $(date)"

# Step 1: Download CIF files from COD and convert to SMILES using OpenBabel
echo "=== Fetching COD CIF files and converting to SMILES ==="
python -c "
import requests
import json
import subprocess
import tempfile
import os
from pathlib import Path
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

# Query COD for entries (no element restrictions for broader coverage)
print('Querying COD API...')
params = {
    'format': 'json',
    'strictmax': '8',  # up to 8 element types
    'el1': 'C',        # must have carbon
}
resp = requests.get('https://www.crystallography.net/cod/result', params=params, timeout=120)
entries = resp.json()
print(f'COD returned {len(entries)} entries')

# Download CIF and convert to SMILES via OpenBabel
smiles_set = set()
smiles_list = []
errors = 0
target = 5000

for i, entry in enumerate(entries):
    if len(smiles_list) >= target:
        break
    cod_id = entry.get('file')
    if not cod_id:
        continue

    cif_url = f'https://www.crystallography.net/cod/{cod_id}.cif'
    try:
        cif_resp = requests.get(cif_url, timeout=10)
        if cif_resp.status_code != 200:
            continue

        # Write CIF to temp file
        with tempfile.NamedTemporaryFile(suffix='.cif', mode='w', delete=False) as f:
            f.write(cif_resp.text)
            cif_path = f.name

        # Convert CIF to SMILES using OpenBabel
        result = subprocess.run(
            ['obabel', cif_path, '-osmi', '--gen3d'],
            capture_output=True, text=True, timeout=30
        )
        os.unlink(cif_path)

        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                smi = line.split()[0] if line.split() else ''
                if not smi or '.' in smi:  # skip multi-component
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol and mol.GetNumAtoms() >= 4:
                    canonical = Chem.MolToSmiles(mol, canonical=True)
                    if canonical not in smiles_set:
                        smiles_set.add(canonical)
                        smiles_list.append(canonical)
    except Exception as e:
        errors += 1

    if (i + 1) % 100 == 0:
        print(f'  Processed {i+1}/{len(entries)}, {len(smiles_list)} unique SMILES, {errors} errors')

print(f'Final: {len(smiles_list)} unique SMILES from {i+1} COD entries')

# Save
with open('data/cod_cif/smiles.json', 'w') as f:
    json.dump(smiles_list, f)
print(f'Saved to data/cod_cif/smiles.json')
"

# Step 2: Generate parsing tasks from COD molecules
echo "=== Generating parsing tasks from COD ==="
python -m clarimol prepare \
    --source file \
    --smiles-file data/cod_cif/smiles.json \
    --output-dir data/cod_5k

# Step 3: Evaluate pre-trained model on COD data
echo "=== Evaluating pre-trained model on COD ==="
python -m clarimol evaluate \
    --model-path output/parsing_pretrain_ada_2/final \
    --data-dir data/cod_5k \
    --output-file output/results_cod5k_trained.json \
    --max-samples 2000 \
    --no-unsloth

# Step 4: Evaluate baseline on COD
echo "=== Evaluating baseline on COD ==="
python -m clarimol evaluate \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --data-dir data/cod_5k \
    --output-file output/results_cod5k_baseline.json \
    --max-samples 2000 \
    --no-unsloth

echo "COD OpenBabel pipeline complete at $(date) :)"
