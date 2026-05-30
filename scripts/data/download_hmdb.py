"""Download HMDB (Human Metabolome Database) structures and convert to SMILES."""

from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile

logger = logging.getLogger(__name__)

# HMDB bulk SDF download. Contains ~220K metabolite structures.
# https://hmdb.ca/downloads — "All Metabolite Structures (SDF)"
HMDB_SDF_URL = "https://hmdb.ca/system/downloads/current/structures.zip"

OUTPUT_FILENAME = "hmdb.smi"


def extract_smiles_from_sdf(sdf_dir: str) -> list[str]:
    """Extract SMILES from SDF files in a directory using RDKit.

    HMDB structures.zip typically contains a single large SDF file
    with all metabolite structures.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    smiles_list = []
    sdf_files = [f for f in os.listdir(sdf_dir) if f.endswith(".sdf")]

    if not sdf_files:
        logger.error("No .sdf files found in extracted archive.")
        return []

    for sdf_file in sdf_files:
        sdf_path = os.path.join(sdf_dir, sdf_file)
        logger.info("Processing SDF file: %s", sdf_file)

        suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=True)
        count = 0
        for mol in suppl:
            if mol is not None:
                smi = Chem.MolToSmiles(mol)
                if smi:
                    smiles_list.append(smi)
            count += 1
            if count % 50_000 == 0:
                logger.info("Processed %d entries, %d valid SMILES so far", count, len(smiles_list))

        logger.info(
            "Finished %s: %d entries processed, %d valid SMILES",
            sdf_file,
            count,
            len(smiles_list),
        )

    return smiles_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download HMDB metabolite structures and convert to SMILES."
    )
    parser.add_argument(
        "--output-dir",
        default="data/sources",
        help="Directory for output .smi file (default: data/sources)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, OUTPUT_FILENAME)

    if os.path.exists(output_path):
        logger.info("Output file already exists: %s — skipping download.", output_path)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "structures.zip")

        logger.info("Downloading HMDB structures from %s", HMDB_SDF_URL)
        logger.info("This file is ~500 MB; download may take several minutes.")
        try:
            req = urllib.request.Request(HMDB_SDF_URL, headers={"User-Agent": "CLARIMOL/1.0"})
            with urllib.request.urlopen(req, timeout=1800) as resp, open(zip_path, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
        except Exception as e:
            logger.error(
                "HMDB download failed: %s. Manually download from %s",
                e,
                HMDB_SDF_URL,
            )
            raise SystemExit(1)

        logger.info("Download complete: %d MB", os.path.getsize(zip_path) // (1 << 20))

        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        logger.info("Extracting ZIP archive...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            logger.error("Downloaded file is not a valid ZIP. The URL may have changed.")
            raise SystemExit(1)

        logger.info("Converting SDF to SMILES with RDKit...")
        smiles_list = extract_smiles_from_sdf(extract_dir)

    if not smiles_list:
        logger.error("No valid SMILES extracted from HMDB.")
        raise SystemExit(1)

    # Deduplicate while preserving order
    seen = set()
    unique_smiles = []
    for smi in smiles_list:
        if smi not in seen:
            seen.add(smi)
            unique_smiles.append(smi)

    with open(output_path, "w") as f:
        for smi in unique_smiles:
            f.write(smi + "\n")

    logger.info(
        "HMDB download complete: %d unique SMILES written to %s (from %d total)",
        len(unique_smiles),
        output_path,
        len(smiles_list),
    )


if __name__ == "__main__":
    main()
