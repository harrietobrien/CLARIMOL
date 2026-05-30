"""Download PI1M polymer informatics dataset (~1M polymer SMILES)."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile

logger = logging.getLogger(__name__)

# PI1M: ~1M hypothetical polymer SMILES from Ma et al.
# Repository: https://github.com/RUIMINMA1996/PI1M
# The dataset is distributed as a CSV within the repository.
PI1M_URLS = [
    # Direct raw file from GitHub
    "https://raw.githubusercontent.com/RUIMINMA1996/PI1M/main/PI1M.csv",
    "https://raw.githubusercontent.com/RUIMINMA1996/PI1M/master/PI1M.csv",
    # GitHub archive fallback
    "https://github.com/RUIMINMA1996/PI1M/archive/refs/heads/main.zip",
]

OUTPUT_FILENAME = "pi1m.smi"


def download_file(url: str, dest: str) -> bool:
    """Download a file from url to dest. Returns True on success."""
    try:
        logger.info("Attempting download from %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "CLARIMOL/1.0"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        return os.path.getsize(dest) > 1000
    except Exception as e:
        logger.warning("Download failed from %s: %s", url, e)
        return False


def extract_smiles_from_csv(csv_path: str) -> list[str]:
    """Extract SMILES column from PI1M CSV file.

    The CSV typically has columns including 'SMILES' or 'smiles'.
    """
    smiles_list = []

    with open(csv_path, "r", errors="replace") as f:
        # Detect delimiter and SMILES column
        sample = f.read(4096)
        f.seek(0)

        dialect = csv.Sniffer().sniff(sample)
        reader = csv.DictReader(f, dialect=dialect)

        # Find the SMILES column (case-insensitive)
        smi_col = None
        for col in reader.fieldnames or []:
            if col.lower().strip() in ("smiles", "smi", "canonical_smiles"):
                smi_col = col
                break

        if smi_col is None:
            # If no header match, try first column
            logger.warning("No SMILES column header found. Trying first column.")
            f.seek(0)
            for line in f:
                smi = line.strip().split(",")[0].split("\t")[0]
                if smi:
                    smiles_list.append(smi)
            return smiles_list

        for row in reader:
            smi = row.get(smi_col, "").strip()
            if smi:
                smiles_list.append(smi)
            if len(smiles_list) % 200_000 == 0 and len(smiles_list) > 0:
                logger.info("Read %d SMILES from CSV so far", len(smiles_list))

    return smiles_list


def validate_smiles(smiles_list: list[str]) -> list[str]:
    """Filter to valid SMILES using RDKit."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    valid = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid.append(smi)
        if (i + 1) % 200_000 == 0:
            logger.info("Validated %d / %d SMILES", i + 1, len(smiles_list))

    logger.info(
        "RDKit validation: %d / %d valid (%.1f%%)",
        len(valid),
        len(smiles_list),
        100.0 * len(valid) / max(len(smiles_list), 1),
    )
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PI1M polymer informatics SMILES dataset."
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
        raw_smiles = None

        # Try direct CSV downloads first
        for url in PI1M_URLS:
            if url.endswith(".zip"):
                continue
            csv_path = os.path.join(tmpdir, "PI1M.csv")
            if download_file(url, csv_path):
                logger.info("Extracting SMILES from CSV...")
                raw_smiles = extract_smiles_from_csv(csv_path)
                if raw_smiles:
                    break

        # Fallback: download full repo ZIP
        if not raw_smiles:
            for url in PI1M_URLS:
                if not url.endswith(".zip"):
                    continue
                zip_path = os.path.join(tmpdir, "pi1m.zip")
                if download_file(url, zip_path):
                    extract_dir = os.path.join(tmpdir, "extracted")
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(extract_dir)
                    # Find CSV files in the extracted contents
                    for root, _dirs, files in os.walk(extract_dir):
                        for fname in files:
                            if fname.lower().endswith(".csv"):
                                csv_path = os.path.join(root, fname)
                                logger.info("Found CSV: %s", csv_path)
                                raw_smiles = extract_smiles_from_csv(csv_path)
                                if raw_smiles:
                                    break
                        if raw_smiles:
                            break

        if not raw_smiles:
            logger.error(
                "PI1M download failed. Manually download from "
                "https://github.com/RUIMINMA1996/PI1M and extract the CSV."
            )
            raise SystemExit(1)

    logger.info("Validating %d SMILES with RDKit...", len(raw_smiles))
    valid = validate_smiles(raw_smiles)

    if not valid:
        logger.error("No valid SMILES extracted from PI1M.")
        raise SystemExit(1)

    with open(output_path, "w") as f:
        for smi in valid:
            f.write(smi + "\n")

    logger.info("PI1M download complete: %d SMILES written to %s", len(valid), output_path)


if __name__ == "__main__":
    main()
