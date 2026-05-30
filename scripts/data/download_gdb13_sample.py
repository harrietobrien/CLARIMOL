"""Download a 1M random subsample from GDB-13 (970M small organic molecules)."""

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

# GDB-13 is distributed as multiple gzipped SMILES files from the Reymond group.
# Total database: ~970M molecules with up to 13 heavy atoms (C, N, O, S, Cl).
# Download page: https://gdb.unibe.ch/downloads/
# The files are named GDB13.xaa.gz through GDB13.xam.gz (13 parts).
GDB13_BASE_URL = "https://gdb.unibe.ch/downloads/"
GDB13_PARTS = [f"GDB13.xa{c}.gz" for c in "abcdefghijklm"]

# Alternative: GDB-13 subset on Zenodo or HuggingFace
GDB13_ZENODO_URL = "https://zenodo.org/records/3588367/files/gdb13.smi.gz"

OUTPUT_FILENAME = "gdb13_1m.smi"
DEFAULT_SAMPLE_SIZE = 1_000_000
SEED = 42


def reservoir_sample_gzip(filepath: str, k: int, rng: random.Random, offset: int = 0) -> tuple[list[str], int]:
    """Reservoir-sample from a gzipped SMILES file.

    Args:
        filepath: Path to .gz file containing one SMILES per line.
        k: Reservoir capacity.
        rng: Random instance for reproducibility.
        offset: Current total count (for multi-file sampling).

    Returns:
        Tuple of (reservoir list, total count after this file).
    """
    reservoir: list[str] = []
    n = offset

    with gzip.open(filepath, "rt", errors="replace") as f:
        for line in f:
            smi = line.strip()
            if not smi or smi.startswith("#"):
                continue
            # GDB files may have SMILES<tab>ID format
            smi = smi.split()[0]

            n += 1
            if n <= k:
                reservoir.append(smi)
            else:
                j = rng.randint(0, n - 1)
                if j < k:
                    reservoir[j] = smi

            if n % 50_000_000 == 0:
                logger.info("Scanned %dM entries so far", n // 1_000_000)

    return reservoir, n


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


def try_zenodo_download(tmpdir: str) -> str | None:
    """Attempt download from Zenodo mirror. Returns path on success, None on failure."""
    gz_path = os.path.join(tmpdir, "gdb13.smi.gz")
    try:
        logger.info("Trying Zenodo mirror: %s", GDB13_ZENODO_URL)
        req = urllib.request.Request(GDB13_ZENODO_URL, headers={"User-Agent": "CLARIMOL/1.0"})
        with urllib.request.urlopen(req, timeout=3600) as resp, open(gz_path, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        if os.path.getsize(gz_path) > 10_000:
            return gz_path
    except Exception as e:
        logger.warning("Zenodo download failed: %s", e)
    return None


def try_reymond_download(tmpdir: str) -> list[str]:
    """Attempt download of GDB-13 parts from gdb.unibe.ch. Returns list of downloaded paths."""
    downloaded = []
    for part_name in GDB13_PARTS:
        url = GDB13_BASE_URL + part_name
        part_path = os.path.join(tmpdir, part_name)
        try:
            logger.info("Downloading %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "CLARIMOL/1.0"})
            with urllib.request.urlopen(req, timeout=3600) as resp, open(part_path, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
            if os.path.getsize(part_path) > 1000:
                downloaded.append(part_path)
                logger.info("Downloaded %s (%d MB)", part_name, os.path.getsize(part_path) // (1 << 20))
        except Exception as e:
            logger.warning("Failed to download %s: %s", part_name, e)
            # Continue with whatever parts were obtained
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a 1M random subsample from GDB-13."
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

    rng = random.Random(args.seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Strategy 1: try Zenodo single-file mirror
        zenodo_path = try_zenodo_download(tmpdir)
        if zenodo_path:
            logger.info("Sampling from Zenodo download...")
            reservoir, total = reservoir_sample_gzip(zenodo_path, args.sample_size, rng)
        else:
            # Strategy 2: download parts from gdb.unibe.ch
            logger.info("Zenodo unavailable. Trying gdb.unibe.ch part files...")
            part_paths = try_reymond_download(tmpdir)
            if not part_paths:
                logger.error(
                    "All GDB-13 download sources failed. Manually download from "
                    "https://gdb.unibe.ch/downloads/ and place .gz files in a temp directory."
                )
                raise SystemExit(1)

            logger.info("Sampling across %d part files...", len(part_paths))
            reservoir: list[str] = []
            total = 0
            for part_path in sorted(part_paths):
                reservoir, total = reservoir_sample_gzip(part_path, args.sample_size, rng, offset=total)
                # The reservoir from each file carries forward
                logger.info("After %s: scanned %dM total", os.path.basename(part_path), total // 1_000_000)

    logger.info("Validating %d sampled SMILES with RDKit...", len(reservoir))
    valid = validate_smiles(reservoir)

    with open(output_path, "w") as f:
        for smi in valid:
            f.write(smi + "\n")

    logger.info("GDB-13 subsample complete: %d SMILES written to %s", len(valid), output_path)


if __name__ == "__main__":
    main()
