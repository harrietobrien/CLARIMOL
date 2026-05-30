"""Download NCI DTP (Developmental Therapeutics Program) open compound database."""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import tempfile
import urllib.request

logger = logging.getLogger(__name__)

# NCI/DTP open database SMILES download.
# The NCI CACTUS server provides bulk downloads of the ~250K compound screening set.
# https://cactus.nci.nih.gov/download/nci/
NCI_URLS = [
    # Tab-separated: NSC_number<tab>SMILES
    "https://cactus.nci.nih.gov/download/nci/NCI-Open_2012-05-01.smi.gz",
    "https://cactus.nci.nih.gov/download/nci/NCI-Open_2012-05-01.smi",
    # Older format
    "https://cactus.nci.nih.gov/download/nci/nci.smi.gz",
]

OUTPUT_FILENAME = "nci.smi"


def download_file(url: str, dest: str) -> bool:
    """Download a file from url to dest. Returns True on success."""
    try:
        logger.info("Attempting download from %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "CLARIMOL/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        return os.path.getsize(dest) > 1000
    except Exception as e:
        logger.warning("Download failed from %s: %s", url, e)
        return False


def extract_and_validate(input_path: str, output_path: str) -> int:
    """Read SMILES from NCI file, validate with RDKit, write valid ones.

    NCI files typically have format: SMILES<tab>NSC_ID or NSC_ID<tab>SMILES.
    Returns count of valid SMILES written.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    is_gz = input_path.endswith(".gz")
    open_fn = gzip.open if is_gz else open

    valid_count = 0
    total = 0

    with open(output_path, "w") as out:
        with open_fn(input_path, "rt", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) < 1:
                    continue

                # The SMILES could be in the first or second column.
                # Try the first token; if it looks numeric, use the second.
                smi_candidate = parts[0].strip()
                if smi_candidate.isdigit() and len(parts) > 1:
                    smi_candidate = parts[1].strip()

                # Skip if the candidate looks like an ID (pure digits)
                if smi_candidate.isdigit():
                    continue

                total += 1
                mol = Chem.MolFromSmiles(smi_candidate)
                if mol is not None:
                    canonical = Chem.MolToSmiles(mol)
                    out.write(canonical + "\n")
                    valid_count += 1

                if total % 50_000 == 0:
                    logger.info("Processed %d entries, %d valid so far", total, valid_count)

    logger.info(
        "Validation complete: %d / %d valid (%.1f%%)",
        valid_count,
        total,
        100.0 * valid_count / max(total, 1),
    )
    return valid_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NCI DTP open compound database SMILES."
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
        for url in NCI_URLS:
            suffix = ".gz" if url.endswith(".gz") else ".smi"
            tmp_file = os.path.join(tmpdir, "nci_raw" + suffix)
            if download_file(url, tmp_file):
                raw_path = tmp_file
                break

        if raw_path is None:
            logger.error(
                "All NCI download URLs failed. Manually download from "
                "https://cactus.nci.nih.gov/download/nci/ and place at %s",
                output_path,
            )
            raise SystemExit(1)

        logger.info("Validating and converting NCI SMILES...")
        count = extract_and_validate(raw_path, output_path)

    if count == 0:
        os.remove(output_path)
        logger.error("No valid SMILES extracted. The download format may have changed.")
        raise SystemExit(1)

    logger.info("NCI download complete: %d SMILES written to %s", count, output_path)


if __name__ == "__main__":
    main()
