#!/usr/bin/env bash
# Download all molecular database sources for CLARIMOL.
# Each script skips the download if its output file already exists.
# RDKit is required for SMILES validation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-data/sources}"

echo "==> Downloading molecular databases to ${OUTPUT_DIR}"
echo ""

run_downloader() {
    local name="$1"
    local script="$2"
    echo "--- ${name} ---"
    if python "${SCRIPT_DIR}/${script}" --output-dir "${OUTPUT_DIR}"; then
        echo "${name}: done."
    else
        echo "${name}: FAILED (continuing with remaining sources)."
    fi
    echo ""
}

run_downloader "COCONUT (natural products)"     download_coconut.py
run_downloader "PubChem 1M subsample"            download_pubchem_sample.py
run_downloader "HMDB (metabolites)"              download_hmdb.py
run_downloader "GDB-13 1M subsample"             download_gdb13_sample.py
run_downloader "PI1M (polymers)"                 download_pi1m.py
run_downloader "NCI DTP (screening compounds)"   download_nci.py

echo "==> All downloads complete. Output directory contents:"
ls -lh "${OUTPUT_DIR}"/*.smi 2>/dev/null || echo "(no .smi files found)"
