"""Download COCONUT natural products database (~695K molecules) as SMILES."""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import tempfile
import urllib.request

logger = logging.getLogger(__name__)

# COCONUT bulk SMILES download endpoint.
# Primary: the official download page at coconut.naturalproducts.net
# Fallback: Zenodo archive of COCONUT SMILES dumps.
COCONUT_URLS = [
    "https://coconut.naturalproducts.net/download/smiles",
    "https://zenodo.org/records/13904699/files/coconut_smiles.smi.gz",
    "https://zenodo.org/records/13904699/files/COCONUT_DB.smi.gz",
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


def validate_and_write(input_path: str, output_path: str) -> int:
    """Read SMILES from input_path, validate with RDKit, write valid ones to output_path.

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
                # SMILES may be the first whitespace-delimited token
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
            suffix = ".gz" if url.endswith(".gz") else ".smi"
            tmp_file = os.path.join(tmpdir, "coconut_raw" + suffix)
            if download_file(url, tmp_file):
                # Check the downloaded file has content
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

        count = validate_and_write(raw_path, output_path)
        if count == 0:
            os.remove(output_path)
            logger.error("No valid SMILES extracted. The download format may have changed.")
            raise SystemExit(1)

    logger.info("COCONUT download complete: %d SMILES written to %s", count, output_path)


if __name__ == "__main__":
    main()
