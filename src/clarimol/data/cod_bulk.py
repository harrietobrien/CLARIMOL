"""
Bulk COD (Crystallography Open Database) SMILES pipeline.

Three acquisition strategies, tried in order of speed and reliability:
    1. COD pre-computed SMILES via rsync (fastest, ~255K structures)
    2. COD REST API with SMILES format (no CIF parsing needed)
    3. OpenBabel CIF→SMILES conversion (fallback for entries without SMILES)

All strategies share the same filtering and deduplication logic via
_is_organic_smiles() from the original cod.py module.
"""

from __future__ import annotations
import csv
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import requests
from rdkit import Chem, RDLogger

from clarimol.data.cod import _is_organic_smiles

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

# COD SMILES index URL (individual .smi files organized by COD ID)
COD_SMI_RSYNC = "rsync://www.crystallography.net/cod/smi/"
COD_SMI_SVN = "svn://www.crystallography.net/cod/smi"

# COD REST API
COD_RESULT_URL = "https://www.crystallography.net/cod/result"
COD_CIF_URL = "https://www.crystallography.net/cod/{cod_id}.cif"

# DataWarrior COD snapshot (pre-built SMILES for 257K structures)
DATAWARRIOR_URL = "https://openmolecules.org/datawarrior/data/COD_13jan2025.zip"


# ─── Strategy 1: COD pre-computed SMILES via rsync ──────────────────────────

def fetch_cod_smiles_rsync(
    output_dir: str | Path,
    timeout: int = 3600,
) -> Path:
    """
    Download COD's pre-computed SMILES files via rsync.
    Returns path to the local SMILES directory.
    """
    out = Path(output_dir) / "cod_smi"
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading COD SMILES via rsync to %s", out)
    try:
        subprocess.run(
            ["rsync", "-av", "--timeout=300", "--compress", COD_SMI_RSYNC, str(out)],
            timeout=timeout,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("rsync complete: %s", out)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("rsync failed: %s", e)
    return out


def parse_smi_directory(smi_dir: str | Path) -> Iterator[tuple[str, str]]:
    """
    Parse COD .smi files from the rsync'd directory.
    Yields (cod_id, smiles) tuples.
    """
    smi_path = Path(smi_dir)
    count = 0
    for smi_file in smi_path.rglob("*.smi"):
        try:
            content = smi_file.read_text().strip()
            if not content:
                continue
            # Format: "SMILES COD_ID" or just "SMILES"
            parts = content.split()
            if parts:
                smiles = parts[0]
                cod_id = smi_file.stem
                yield cod_id, smiles
                count += 1
        except Exception as e:
            logger.debug("Failed to parse %s: %s", smi_file, e)
    logger.info("Parsed %d .smi files from %s", count, smi_dir)


# ─── Strategy 2: COD REST API with SMILES ───────────────────────────────────

def fetch_cod_smiles_api(
    max_entries: int = 50000,
    required_elements: list[str] | None = None,
    max_elements: int = 8,
    timeout: float = 300,
) -> list[tuple[str, str]]:
    """
    Query COD REST API for organic entries and retrieve their SMILES.
    Returns list of (cod_id, smiles) tuples.
    """
    if required_elements is None:
        required_elements = ["C", "H"]

    params: dict[str, str] = {
        "format": "json",
        "strictmax": str(max_elements),
    }
    for i, el in enumerate(required_elements[:8], 1):
        params[f"el{i}"] = el
    # Exclude common metals
    for i, el in enumerate(["Fe", "Cu", "Zn", "Ni"], 1):
        params[f"nel{i}"] = el

    logger.info("Querying COD API for organic entries (max_elements=%d)", max_elements)
    try:
        resp = requests.get(COD_RESULT_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        entries = resp.json()
    except Exception as e:
        logger.error("COD API query failed: %s", e)
        return []

    logger.info("COD API returned %d entries", len(entries))
    results: list[tuple[str, str]] = []
    for entry in entries:
        if len(results) >= max_entries:
            break
        cod_id = str(entry.get("file", ""))
        smiles = (entry.get("smiles") or "").strip()
        if smiles and cod_id:
            results.append((cod_id, smiles))

    logger.info("COD API: %d entries with SMILES out of %d total", len(results), len(entries))
    return results


# ─── Strategy 3: OpenBabel CIF→SMILES conversion ────────────────────────────

def _obabel_available() -> bool:
    """Check if OpenBabel CLI is installed."""
    try:
        result = subprocess.run(
            ["obabel", "-V"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_cif(cod_id: str, output_dir: str | Path, timeout: float = 30) -> Path | None:
    """Download a single CIF file from COD."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cif_path = out / f"{cod_id}.cif"
    if cif_path.exists():
        return cif_path
    url = COD_CIF_URL.format(cod_id=cod_id)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        cif_path.write_text(resp.text)
        return cif_path
    except requests.RequestException as e:
        logger.debug("Failed to download CIF %s: %s", cod_id, e)
        return None


def cif_to_smiles_obabel(cif_path: str | Path) -> str | None:
    """Convert a CIF file to SMILES using OpenBabel."""
    try:
        result = subprocess.run(
            ["obabel", str(cif_path), "-osmi", "-e"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            # OpenBabel outputs "SMILES\tname" format
            line = result.stdout.strip().splitlines()[0]
            smiles = line.split("\t")[0].split()[0]
            return smiles
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug("OpenBabel conversion failed for %s: %s", cif_path, e)
    return None


def cif_to_smiles_batch(
    cif_dir: str | Path,
    max_files: int | None = None,
) -> list[tuple[str, str]]:
    """
    Batch convert CIF files to SMILES using OpenBabel.
    Returns list of (cod_id, smiles) tuples.
    """
    cif_path = Path(cif_dir)
    cif_files = sorted(cif_path.glob("*.cif"))
    if max_files:
        cif_files = cif_files[:max_files]

    if not cif_files:
        logger.warning("No CIF files found in %s", cif_dir)
        return []

    if not _obabel_available():
        logger.error("OpenBabel (obabel) not found. Install with: conda install -c conda-forge openbabel")
        return []

    logger.info("Converting %d CIF files via OpenBabel", len(cif_files))
    results: list[tuple[str, str]] = []
    failed = 0

    for i, cif_file in enumerate(cif_files):
        smiles = cif_to_smiles_obabel(cif_file)
        if smiles:
            results.append((cif_file.stem, smiles))
        else:
            failed += 1
        if (i + 1) % 1000 == 0:
            logger.info("OpenBabel progress: %d/%d converted, %d failed",
                        len(results), i + 1, failed)

    logger.info("OpenBabel: %d/%d converted successfully (%d failed)",
                len(results), len(cif_files), failed)
    return results


# ─── Filtering & Deduplication ───────────────────────────────────────────────

# Optional element whitelist for strict organic-only filtering
ORGANIC_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}  # H,B,C,N,O,F,Si,P,S,Cl,Br,I


def filter_smiles(
    entries: list[tuple[str, str]],
    require_carbon: bool = False,
    reject_multicomponent: bool = True,
    min_atoms: int | None = None,
    max_atoms: int | None = None,
    allowed_elements: set[int] | None = None,
) -> list[tuple[str, str, str]]:
    """
    Filter and canonicalize SMILES entries.

    By default, only rejects unparseable SMILES, multi-component entries,
    and molecules without carbon. No element whitelist or size filter
    is applied unless explicitly requested.

    :param entries: List of (cod_id, raw_smiles) tuples
    :param require_carbon: Require at least one carbon atom
    :param reject_multicomponent: Reject SMILES with "." (salts, solvates)
    :param min_atoms: Minimum heavy atom count (None = no minimum)
    :param max_atoms: Maximum heavy atom count (None = no maximum)
    :param allowed_elements: Set of allowed atomic numbers (None = allow all)
    :return: List of (cod_id, canonical_smiles, raw_smiles) tuples, deduplicated
    """
    seen: set[str] = set()
    results: list[tuple[str, str, str]] = []
    rejected = {"parse": 0, "multicomponent": 0, "elements": 0, "size": 0,
                "no_carbon": 0, "duplicate": 0}

    for cod_id, raw_smiles in entries:
        raw_smiles = raw_smiles.strip()
        if not raw_smiles:
            continue

        # Reject multi-component (salts, solvates)
        if reject_multicomponent and "." in raw_smiles:
            rejected["multicomponent"] += 1
            continue

        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            rejected["parse"] += 1
            continue

        atom_nums = {a.GetAtomicNum() for a in mol.GetAtoms()}

        # Element filter (only if whitelist provided)
        if allowed_elements is not None and not atom_nums.issubset(allowed_elements):
            rejected["elements"] += 1
            continue

        # Carbon requirement
        if require_carbon and 6 not in atom_nums:
            rejected["no_carbon"] += 1
            continue

        # Size filter (only if bounds provided)
        if min_atoms is not None or max_atoms is not None:
            n_atoms = mol.GetNumHeavyAtoms()
            if min_atoms is not None and n_atoms < min_atoms:
                rejected["size"] += 1
                continue
            if max_atoms is not None and n_atoms > max_atoms:
                rejected["size"] += 1
                continue

        canonical = Chem.MolToSmiles(mol, canonical=True)

        # Deduplicate
        if canonical in seen:
            rejected["duplicate"] += 1
            continue

        seen.add(canonical)
        results.append((cod_id, canonical, raw_smiles))

    logger.info(
        "Filtered %d → %d unique SMILES (rejected: %s)",
        len(entries), len(results), rejected,
    )
    return results


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def fetch_cod_bulk(
    max_molecules: int = 250000,
    cache_dir: str | Path = "data/cod_cache",
    strategies: list[str] | None = None,
    strict_organic: bool = True,
    min_atoms: int = 4,
    max_atoms: int = 100,
) -> list[str]:
    """
    Fetch organic molecular crystal SMILES from COD using multiple strategies.

    Tries strategies in order until enough molecules are collected:
        1. "rsync" — COD pre-computed SMILES (comprehensive, ~255K structures)
        2. "api" — COD REST API (moderate yield)
        3. "obabel" — CIF download + OpenBabel conversion (slow, fallback)

    :param max_molecules: Target number of unique SMILES
    :param cache_dir: Directory for caching downloaded data
    :param strategies: Which strategies to try (default: ["rsync", "api"])
    :param strict_organic: Apply organic element whitelist and require carbon
    :param min_atoms: Minimum heavy atom count
    :param max_atoms: Maximum heavy atom count
    :return: List of unique, valid, canonical SMILES strings
    """
    if strategies is None:
        strategies = ["rsync", "api"]

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # Check for cached results first
    cache_file = cache / "cod_bulk_smiles.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        if cached:
            logger.info("Loaded %d cached COD SMILES from %s", len(cached), cache_file)
            return cached[:max_molecules]

    all_entries: list[tuple[str, str]] = []

    for strategy in strategies:
        if len(all_entries) >= max_molecules * 2:  # fetch extra for filtering headroom
            break

        if strategy == "api":
            logger.info("Strategy: COD REST API")
            api_results = fetch_cod_smiles_api(
                max_entries=max_molecules * 3,
                max_elements=8,
            )
            all_entries.extend(api_results)
            logger.info("After API: %d total raw entries", len(all_entries))

        elif strategy == "rsync":
            logger.info("Strategy: COD rsync SMILES")
            smi_dir = cache / "cod_smi"
            if not any(smi_dir.rglob("*.smi")) if smi_dir.exists() else True:
                fetch_cod_smiles_rsync(cache)
            for cod_id, smiles in parse_smi_directory(smi_dir):
                all_entries.append((cod_id, smiles))
            logger.info("After rsync: %d total raw entries", len(all_entries))

        elif strategy == "obabel":
            logger.info("Strategy: OpenBabel CIF conversion")
            cif_dir = cache / "cif"
            if cif_dir.exists() and any(cif_dir.glob("*.cif")):
                obabel_results = cif_to_smiles_batch(cif_dir)
                all_entries.extend(obabel_results)
                logger.info("After OpenBabel: %d total raw entries", len(all_entries))
            else:
                logger.warning("No CIF files in %s — skipping OpenBabel strategy", cif_dir)

    # Filter and deduplicate
    filter_kwargs: dict = {
        "reject_multicomponent": True,
    }
    if strict_organic:
        filter_kwargs["require_carbon"] = True
        filter_kwargs["allowed_elements"] = ORGANIC_ELEMENTS
    if min_atoms is not None:
        filter_kwargs["min_atoms"] = min_atoms
    if max_atoms is not None:
        filter_kwargs["max_atoms"] = max_atoms
    filtered = filter_smiles(all_entries, **filter_kwargs)
    smiles_list = [canonical for _, canonical, _ in filtered]

    if len(smiles_list) > max_molecules:
        smiles_list = smiles_list[:max_molecules]

    # Cache results
    with open(cache_file, "w") as f:
        json.dump(smiles_list, f)
    logger.info("Saved %d COD SMILES to cache at %s", len(smiles_list), cache_file)

    # Also save full metadata for provenance
    meta_file = cache / "cod_bulk_metadata.json"
    metadata = [
        {"cod_id": cod_id, "canonical": canonical, "raw": raw}
        for cod_id, canonical, raw in filtered[:len(smiles_list)]
    ]
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", meta_file)

    return smiles_list
