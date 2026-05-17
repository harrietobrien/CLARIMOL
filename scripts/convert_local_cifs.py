"""
Batch convert local COD CIF files to SMILES using OpenBabel.
Runs in parallel, filters for organic molecules, deduplicates.

Usage:
    python scripts/convert_local_cifs.py \
        --cif-dir /home/harrie/6840DM/PolyMine/data/raw/cif \
        --output data/cod_cache/cod_bulk_smiles.json \
        --workers 10
"""

from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ORGANIC_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}


def cif_to_smiles(cif_path: str) -> tuple[str, str | None]:
    """Convert a single CIF to SMILES via obabel. Returns (cod_id, smiles_or_None)."""
    try:
        result = subprocess.run(
            ["obabel", cif_path, "-osmi", "-e"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().splitlines()[0]
            smiles = line.split("\t")[0].split()[0]
            return (Path(cif_path).stem, smiles)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return (Path(cif_path).stem, None)


def is_valid_organic(smiles: str, min_atoms: int = 4, max_atoms: int = 100) -> str | None:
    """Validate and canonicalize. Returns canonical SMILES or None."""
    if "." in smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atoms = {a.GetAtomicNum() for a in mol.GetAtoms()}
    if 6 not in atoms:
        return None
    if not atoms.issubset(ORGANIC_ELEMENTS):
        return None
    n = mol.GetNumHeavyAtoms()
    if n < min_atoms or n > max_atoms:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def main():
    parser = argparse.ArgumentParser(description="Batch convert COD CIFs to SMILES")
    parser.add_argument("--cif-dir", required=True, help="Directory with CIF files")
    parser.add_argument("--output", default="data/cod_cache/cod_bulk_smiles.json")
    parser.add_argument("--metadata", default=None, help="Metadata JSON output (default: alongside --output)")
    parser.add_argument("--max-molecules", type=int, default=250000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-report", type=int, default=10000, help="Log progress every N files")
    args = parser.parse_args()

    cif_dir = Path(args.cif_dir)
    cif_files = sorted(cif_dir.rglob("*.cif"))
    logger.info("Found %d CIF files in %s", len(cif_files), cif_dir)

    if not cif_files:
        logger.error("No CIF files found")
        sys.exit(1)

    seen: set[str] = set()
    results: list[str] = []
    metadata: list[dict] = []
    converted = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(cif_to_smiles, str(f)): f for f in cif_files}
        for i, future in enumerate(as_completed(futures), 1):
            cod_id, raw_smiles = future.result()

            if raw_smiles is None:
                failed += 1
            else:
                converted += 1
                canonical = is_valid_organic(raw_smiles)
                if canonical and canonical not in seen:
                    seen.add(canonical)
                    results.append(canonical)
                    metadata.append({
                        "cod_id": cod_id,
                        "canonical": canonical,
                        "raw": raw_smiles,
                    })

            if i % args.batch_report == 0:
                logger.info(
                    "Progress: %d/%d processed, %d converted, %d valid unique, %d failed",
                    i, len(cif_files), converted, len(results), failed,
                )

            if len(results) >= args.max_molecules:
                logger.info("Reached target of %d molecules, stopping", args.max_molecules)
                executor.shutdown(wait=False, cancel_futures=True)
                break

    logger.info(
        "Final: %d/%d processed, %d converted, %d valid unique organic SMILES",
        converted + failed, len(cif_files), converted, len(results),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results[:args.max_molecules], f)
    logger.info("Saved %d SMILES to %s", len(results), out_path)

    meta_path = Path(args.metadata) if args.metadata else out_path.with_name("cod_bulk_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata[:args.max_molecules], f, indent=2)
    logger.info("Saved metadata to %s", meta_path)


if __name__ == "__main__":
    main()
