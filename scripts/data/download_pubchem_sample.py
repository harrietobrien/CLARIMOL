"""Download a 1M random subsample of PubChem Compound SMILES via FTP."""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import random
import shutil
import tempfile
import urllib.request

logger = logging.getLogger(__name__)

# PubChem canonical SMILES dump (gzipped, ~6 GB compressed).
# Contains CID<tab>SMILES lines for all ~115M compounds.
PUBCHEM_URL = "ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
# HTTPS mirror (more reliable through firewalls)
PUBCHEM_URL_HTTPS = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"

OUTPUT_FILENAME = "pubchem_1m.smi"
DEFAULT_SAMPLE_SIZE = 1_000_000
SEED = 42


def reservoir_sample(filepath: str, k: int, seed: int) -> list[str]:
    """Reservoir-sample k SMILES from a gzipped CID-SMILES file.

    Uses Algorithm R (Vitter 1985) for memory-efficient uniform sampling
    without loading the full file into memory.
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    n = 0

    with gzip.open(filepath, "rt", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            smi = parts[1].strip()
            if not smi:
                continue

            n += 1
            if n <= k:
                reservoir.append(smi)
            else:
                j = rng.randint(0, n - 1)
                if j < k:
                    reservoir[j] = smi

            if n % 10_000_000 == 0:
                logger.info("Scanned %dM entries, reservoir full=%s", n // 1_000_000, len(reservoir) >= k)

    logger.info("Total entries scanned: %d, reservoir size: %d", n, len(reservoir))
    return reservoir


def validate_smiles(smiles_list: list[str]) -> list[str]:
    """Filter to valid SMILES using RDKit."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid.append(smi)

    logger.info(
        "RDKit validation: %d / %d valid (%.1f%%)",
        len(valid),
        len(smiles_list),
        100.0 * len(valid) / max(len(smiles_list), 1),
    )
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a random 1M subsample from PubChem Compound SMILES."
    )
    parser.add_argument(
        "--output-dir",
        default="data/sources",
        help="Directory for output .smi file (default: data/sources)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of molecules to sample (default: 1000000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducible sampling (default: 42)",
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
        gz_path = os.path.join(tmpdir, "CID-SMILES.gz")

        # Try HTTPS first (works through most firewalls), fall back to FTP
        for url in [PUBCHEM_URL_HTTPS, PUBCHEM_URL]:
            try:
                logger.info("Downloading PubChem CID-SMILES from %s", url)
                logger.info("This file is ~6 GB compressed; download may take 30+ minutes.")
                req = urllib.request.Request(url, headers={"User-Agent": "CLARIMOL/1.0"})
                with urllib.request.urlopen(req, timeout=3600) as resp, open(gz_path, "wb") as f:
                    shutil.copyfileobj(resp, f, length=1 << 20)
                logger.info("Download complete: %d MB", os.path.getsize(gz_path) // (1 << 20))
                break
            except Exception as e:
                logger.warning("Download failed from %s: %s", url, e)
        else:
            logger.error(
                "PubChem download failed. Manually download CID-SMILES.gz from "
                "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/ and re-run."
            )
            raise SystemExit(1)

        logger.info("Reservoir-sampling %d molecules (seed=%d)...", args.sample_size, args.seed)
        sampled = reservoir_sample(gz_path, args.sample_size, args.seed)

    logger.info("Validating %d sampled SMILES with RDKit...", len(sampled))
    valid = validate_smiles(sampled)

    with open(output_path, "w") as f:
        for smi in valid:
            f.write(smi + "\n")

    logger.info("PubChem subsample complete: %d SMILES written to %s", len(valid), output_path)


if __name__ == "__main__":
    main()
