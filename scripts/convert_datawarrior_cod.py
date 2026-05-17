"""
Extract organic SMILES from the DataWarrior COD snapshot (.dwar file).
Uses OpenChemLib (via JPype) to decode IDCode → SMILES, then filters with RDKit.

Usage:
    python scripts/convert_datawarrior_cod.py \
        --dwar-zip data/cod_cache/COD_datawarrior.zip \
        --output data/cod_cache/cod_bulk_smiles.json \
        --max-molecules 250000
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ORGANIC_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}


def is_valid_organic(smiles: str, min_atoms: int = 4, max_atoms: int = 100) -> str | None:
    """Validate and canonicalize with RDKit. Returns canonical SMILES or None."""
    from rdkit import Chem
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
    parser = argparse.ArgumentParser(description="Extract organic SMILES from DataWarrior COD")
    parser.add_argument("--dwar-zip", required=True, help="Path to COD_datawarrior.zip")
    parser.add_argument("--output", default="data/cod_cache/cod_bulk_smiles.json")
    parser.add_argument("--ocl-jar", default="/tmp/openchemlib.jar",
                        help="Path to OpenChemLib JAR")
    parser.add_argument("--max-molecules", type=int, default=250000)
    args = parser.parse_args()

    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)

    # Start JVM with OpenChemLib
    import jpype
    import jpype.imports
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=[args.ocl_jar])
    from com.actelion.research.chem import IDCodeParser, SmilesCreator, StereoMolecule

    ocl_parser = IDCodeParser()
    smiles_creator = SmilesCreator()

    # Read the .dwar file from ZIP
    zip_path = Path(args.dwar_zip)
    logger.info("Reading %s", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        dwar_name = [n for n in zf.namelist() if n.endswith(".dwar")][0]
        text = zf.read(dwar_name).decode("utf-8", errors="replace")

    lines = text.splitlines()
    logger.info("Total lines: %d", len(lines))

    # Find header line
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("idcoordinates3D\t"):
            header_idx = i
            break

    if header_idx is None:
        logger.error("Could not find header line")
        sys.exit(1)

    headers = lines[header_idx].split("\t")
    col = {h: i for i, h in enumerate(headers)}
    logger.info("Header at line %d, %d columns", header_idx, len(headers))
    logger.info("Key columns: Structure=%s, Type=%s, COD Number=%s",
                col.get("Structure"), col.get("Type"), col.get("COD Number"))

    struct_idx = col["Structure"]
    type_idx = col["Type"]
    cod_idx = col.get("COD Number")

    seen: set[str] = set()
    results: list[str] = []
    metadata: list[dict] = []
    processed = 0
    organic_count = 0
    decode_fail = 0
    filter_fail = 0

    for line in lines[header_idx + 1:]:
        if not line.strip() or line.startswith("<"):
            continue
        parts = line.split("\t")
        if len(parts) <= max(struct_idx, type_idx):
            continue

        entry_type = parts[type_idx].strip()
        if entry_type != "organic":
            continue

        organic_count += 1
        idcode = parts[struct_idx].strip()
        cod_id = parts[cod_idx].strip() if cod_idx is not None and len(parts) > cod_idx else ""

        # Decode IDCode → SMILES
        try:
            mol = StereoMolecule()
            ocl_parser.parse(mol, idcode)
            raw_smiles = str(smiles_creator.generateSmiles(mol))
        except Exception:
            decode_fail += 1
            continue

        if not raw_smiles:
            decode_fail += 1
            continue

        # Validate and canonicalize
        canonical = is_valid_organic(raw_smiles)
        if canonical is None:
            filter_fail += 1
            continue

        if canonical in seen:
            continue

        seen.add(canonical)
        results.append(canonical)
        metadata.append({"cod_id": cod_id, "canonical": canonical, "raw": raw_smiles})

        processed += 1
        if processed % 10000 == 0:
            logger.info("Progress: %d valid unique / %d organic scanned (%d decode fail, %d filter fail)",
                        len(results), organic_count, decode_fail, filter_fail)

        if len(results) >= args.max_molecules:
            logger.info("Reached target of %d molecules", args.max_molecules)
            break

    logger.info("Final: %d organic entries, %d decode failures, %d filter failures, %d valid unique",
                organic_count, decode_fail, filter_fail, len(results))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    logger.info("Saved %d SMILES to %s", len(results), out_path)

    meta_path = out_path.with_name("cod_bulk_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    logger.info("Saved metadata to %s", meta_path)


if __name__ == "__main__":
    main()
