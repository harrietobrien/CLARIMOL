"""Download COCONUT natural products database (~695K molecules) as SMILES."""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile

logger = logging.getLogger(__name__)

# COCONUT bulk SMILES download endpoint.
# Primary: the official download page at coconut.naturalproducts.net
# Fallback: Zenodo archive of COCONUT SMILES dumps.
COCONUT_URLS = [
    # COCONUT 2.0 CSV lite contains SMILES column. Updated monthly at coconut.s3.uni-jena.de.
    "https://coconut.s3.uni-jena.de/prod/downloads/2026-05/coconut_csv_lite-05-2026.zip",
    "https://coconut.s3.uni-jena.de/prod/downloads/2026-04/coconut_csv_lite-04-2026.zip",
    "https://coconut.s3.uni-jena.de/prod/downloads/2026-03/coconut_csv_lite-03-2026.zip",
    # Older direct SMILES endpoint (may no longer work)
    "https://coconut.naturalproducts.net/download/smiles",
]

OUTPUT_FILENAME = "coconut.smi"


def download_file(url: str, dest: str) -> bool:
    """Download a file from url to dest. Returns True on success."""
    try:
        logger.info("Attempting download from %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "CLARIMOL/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        return True
    except Exception as e:
        logger.warning("Download failed from %s: %s", url, e)
        return False


def extract_smiles_from_csv_zip(zip_path: str, output_path: str) -> int:
    """Extract SMILES from a COCONUT CSV ZIP archive, validate with RDKit.

    The COCONUT CSV lite files contain a 'canonical_smiles' or 'smiles' column.
    Returns count of valid SMILES written.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    valid_count = 0
    total = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            logger.error("No CSV files found in ZIP archive.")
            return 0

        with open(output_path, "w") as out:
            for csv_name in csv_files:
                logger.info("Processing %s from archive.", csv_name)
                with zf.open(csv_name) as raw:
                    reader = csv.DictReader(
                        (line.decode("utf-8", errors="replace") for line in raw)
                    )
                    # Find the SMILES column
                    smiles_col = None
                    for candidate in ["canonical_smiles", "smiles", "SMILES", "Canonical_SMILES"]:
                        if candidate in (reader.fieldnames or []):
                            smiles_col = candidate
                            break
                    if smiles_col is None:
                        logger.warning(
                            "No SMILES column found in %s. Columns: %s",
                            csv_name,
                            reader.fieldnames,
                        )
                        continue

                    for row in reader:
                        smi = (row.get(smiles_col) or "").strip()
                        if not smi:
                            continue
                        total += 1
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            out.write(smi + "\n")
                            valid_count += 1
                        if total % 100_000 == 0:
                            logger.info("Processed %d SMILES, %d valid so far", total, valid_count)

    logger.info(
        "Validation complete: %d / %d SMILES valid (%.1f%%)",
        valid_count,
        total,
        100.0 * valid_count / max(total, 1),
    )
    return valid_count


def validate_and_write(input_path: str, output_path: str) -> int:
    """Read SMILES from a plain text file, validate with RDKit, write valid ones.

    Returns count of valid SMILES written.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    open_fn = gzip.open if input_path.endswith(".gz") else open
    valid_count = 0
    total = 0

    with open(output_path, "w") as out:
        with open_fn(input_path, "rt", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                smi = line.split()[0]
                total += 1
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    out.write(smi + "\n")
                    valid_count += 1
                if total % 100_000 == 0:
                    logger.info("Processed %d SMILES, %d valid so far", total, valid_count)

    logger.info(
        "Validation complete: %d / %d SMILES valid (%.1f%%)",
        valid_count,
        total,
        100.0 * valid_count / max(total, 1),
    )
    return valid_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download COCONUT natural products SMILES database."
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
        raw_path = None
        for url in COCONUT_URLS:
            if url.endswith(".zip"):
                suffix = ".zip"
            elif url.endswith(".gz"):
                suffix = ".gz"
            else:
                suffix = ".smi"
            tmp_file = os.path.join(tmpdir, "coconut_raw" + suffix)
            if download_file(url, tmp_file):
                if os.path.getsize(tmp_file) > 1000:
                    raw_path = tmp_file
                    break
                else:
                    logger.warning("Downloaded file too small (%d bytes), trying next URL.", os.path.getsize(tmp_file))

        if raw_path is None:
            logger.error(
                "All download URLs failed. Manually download COCONUT SMILES from "
                "https://coconut.naturalproducts.net/ and place at %s",
                output_path,
            )
            raise SystemExit(1)

        if raw_path.endswith(".zip"):
            count = extract_smiles_from_csv_zip(raw_path, output_path)
        else:
            count = validate_and_write(raw_path, output_path)

        if count == 0:
            if os.path.exists(output_path):
                os.remove(output_path)
            logger.error("No valid SMILES extracted. The download format may have changed.")
            raise SystemExit(1)

    logger.info("COCONUT download complete: %d SMILES written to %s", count, output_path)


if __name__ == "__main__":
    main()
