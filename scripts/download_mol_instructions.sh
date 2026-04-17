#!/usr/bin/env bash
: <<'COMMENT'
Download Mol-Instructions (Fang et al., 2024) from HuggingFace
and extract the three molecular generation task JSONs.

bash scripts/download_mol_instructions.sh

Requires: huggingface_hub (pip install huggingface_hub)
COMMENT

set -euo pipefail
cd "$(dirname "$0")/.."

DEST="data/mol_instructions"
REPO="zjunlp/Mol-Instructions"
ZIP_PATH="data/Molecule-oriented_Instructions.zip"
TASKS=("retrosynthesis" "reagent_prediction" "forward_reaction_prediction")

# Skip if all three files already exist
all_present=true
for task in "${TASKS[@]}"; do
    if [[ ! -f "$DEST/$task.json" ]]; then
        all_present=false
        break
    fi
done
if $all_present; then
    echo "All task files already present in $DEST/, skipping download."
    exit 0
fi

echo "Downloading Mol-Instructions from $REPO ..."
mkdir -p "$DEST"

# Download the zip via huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='$REPO',
    filename='$ZIP_PATH',
    repo_type='dataset',
)
print(path)
" > /tmp/mol_instructions_zip_path.txt

ZIP_LOCAL=$(tail -1 /tmp/mol_instructions_zip_path.txt)
echo "Extracting task JSONs from $ZIP_LOCAL ..."

# Extract only the three task files we need
python3 - "$ZIP_LOCAL" "$DEST" <<'PYEOF'
import json, sys, zipfile
from pathlib import Path

zip_path, dest = sys.argv[1], Path(sys.argv[2])
tasks = ["retrosynthesis", "reagent_prediction", "forward_reaction_prediction"]

with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    for task in tasks:
        # Find the matching file inside the zip
        matches = [n for n in names if task in n and n.endswith(".json")]
        if not matches:
            print(f"WARNING: {task}.json not found in zip", file=sys.stderr)
            continue
        src = matches[0]
        print(f"  {src} -> {dest / f'{task}.json'}")
        raw = json.loads(zf.read(src))
        # Ensure metadata has task and split fields
        for entry in raw:
            md = entry.setdefault("metadata", {})
            md.setdefault("task", task)
            md.setdefault("split", "train")
        with open(dest / f"{task}.json", "w") as f:
            json.dump(raw, f)
PYEOF

rm -f /tmp/mol_instructions_zip_path.txt

echo ""
echo "Done. Files in $DEST/:"
ls -lh "$DEST"/*.json
