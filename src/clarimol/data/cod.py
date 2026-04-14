"""
Crystallography Open Database (COD) Client (additional experiment):
    Fetches organic molecular crystal structures and extracts SMILES strings
    for use as an alternative/supplementary molecule source alongside ZINC250K.
        1. Query COD REST API for organic molecular crystals (element filters)
        2. Use chemical names (chemname) to look up SMILES via PubChem
        3. Filter for valid, single-component organic molecules
        4. Deduplicate and return list suitable for build_dataset()
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
import requests
from rdkit import Chem, RDLogger
import pubchempy as pcp

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

# PubChem 404s are expected (many crystal names aren't in PubChem)
logging.getLogger("pubchempy").setLevel(logging.WARNING)

COD_RESULT_URL = "https://www.crystallography.net/cod/result"

# Common metals to exclude when filtering for organic molecular crystals
EXCLUDE_METALS = [
    "Fe", "Cu", "Zn", "Ni", "Co", "Mn", "Cr", "Ti", "V", "Mo",
    "W", "Pd", "Pt", "Ru", "Rh", "Ir", "Os", "Ag", "Au", "Hg",
    "Cd", "Pb", "Sn", "Bi", "Sb", "As", "Se", "Te", "La", "Ce",
    "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Pr", "U", "Th", "Zr", "Hf", "Nb", "Ta", "Re", "Al",
    "Ga", "In", "Tl", "Ge",
]


def _is_organic_smiles(smiles: str) -> bool:
    """Check if a SMILES represents a single-component organic molecule"""
    # Reject multi-component (salts, solvates)
    if "." in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    # Must have carbon
    has_carbon = any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
    if not has_carbon:
        return False
    # Reject molecules with metals
    metal_nums = {
        26, 29, 30, 28, 27, 25, 24, 22, 23, 42, 74, 46, 78, 44, 45,
        77, 76, 47, 79, 80, 48, 82, 50, 83, 51, 33, 34, 52, 57, 58,
        60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 59, 92, 90, 40,
        72, 41, 73, 75, 13, 31, 49, 81, 32, 4, 11, 12, 19, 20, 37,
        38, 55, 56, 3,
    }
    if any(a.GetAtomicNum() in metal_nums for a in mol.GetAtoms()):
        return False
    # Minimum size
    if mol.GetNumAtoms() < 4:
        return False
    return True


def _name_to_smiles(name: str) -> str | None:
    """Look up a chemical name via PubChem and return validated SMILES"""
    try:
        results = pcp.get_compounds(name, "name")
        if results is not None:
            smiles = getattr(results[0], "connectivity_smiles", None) \
                     or results[0].canonical_smiles
            if smiles and _is_organic_smiles(smiles):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        logger.debug("PubChem lookup failed for '%s': %s", name, e)
    return None


def query_cod_entries(
    required_elements: list[str] | None = None,
    excluded_elements: list[str] | None = None,
    max_elements: int = 5,
    timeout: float = 120,
) -> list[dict]:
    """
    Query COD for organic molecular crystal entries with chemical names
    :return: list of dicts w/ keys: cod_id, chemname, formula, space_group, etc.
    """
    if required_elements is None:
        required_elements = ["C", "H"]
    if excluded_elements is None:
        excluded_elements = EXCLUDE_METALS[:4]
    params: dict[str, str] = {
        "format": "json",
        "strictmax": str(max_elements),
    }
    for i, el in enumerate(required_elements[:8], 1):
        params[f"el{i}"] = el
    for i, el in enumerate(excluded_elements[:4], 1):
        params[f"nel{i}"] = el
    logger.info("Querying COD: required=%s, max_elements=%d", required_elements, max_elements)
    try:
        resp = requests.get(COD_RESULT_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        entries = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("COD query failed: %s", e)
        return []
    # Filter to entries with chemical names (for PubChem lookup)
    named = list()
    for entry in entries:
        chemname = (entry.get("chemname") or "").strip()
        if chemname and len(chemname) > 2:
            named.append({
                "cod_id": entry.get("file"),
                "chemname": chemname,
                "formula": (entry.get("formula") or "").strip("- "),
                "space_group": entry.get("sg"),
                "sg_number": entry.get("sgNumber"),
                "a": entry.get("a"),
                "b": entry.get("b"),
                "c": entry.get("c"),
                "volume": entry.get("vol"),
                "Z": entry.get("Z"),
                "year": entry.get("year"),
            })
    logger.info("COD: %d total entries, %d with chemical names", len(entries), len(named))
    return named


def fetch_cod_smiles(
    max_entries: int = 1000,
    required_elements: list[str] | None = None,
    delay: float = 0.3,
    cache_dir: str | Path | None = None,
) -> list[str]:
    """
    Fetch organic molecular crystal SMILES from COD.
        Queries COD for organic structures, resolves SMILES via PubChem
        chemical name lookup, deduplicates, and returns valid SMILES.

    :param: max_entries: Target number of unique SMILES to collect
    :param: required_elements: Elements that must be present
    :param: delay: Seconds between PubChem lookups (polite rate limiting)
    :param: cache_dir: If set, cache resolved name→SMILES mappings

    :return: List of unique, valid SMILES strings
    """
    cache_path = None
    name_cache: dict[str, str | None] = {}
    if cache_dir:
        cache_p = Path(cache_dir)
        cache_p.mkdir(parents=True, exist_ok=True)
        cache_path = cache_p / "name_to_smiles.json"
        if cache_path.exists():
            with open(cache_path) as f:
                name_cache = json.load(f)
            logger.info("Loaded %d cached name→SMILES mappings", len(name_cache))
    entries = query_cod_entries(required_elements=required_elements)
    seen: set[str] = set()
    smiles_list: list[str] = []
    lookups = 0
    for i, entry in enumerate(entries):
        if len(smiles_list) >= max_entries:
            break
        name = entry["chemname"]
        # Check cache first
        if name in name_cache:
            smiles = name_cache[name]
        else:
            smiles = _name_to_smiles(name)
            name_cache[name] = smiles
            lookups += 1
            if delay > 0 and lookups % 5 == 0:
                time.sleep(delay)
        if smiles and smiles not in seen:
            seen.add(smiles)
            smiles_list.append(smiles)
        if (i + 1) % 200 == 0:
            logger.info(
                "COD progress: %d/%d checked, %d unique SMILES, %d PubChem lookups",
                i + 1, len(entries), len(smiles_list), lookups,
            )
    # Save cache
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(name_cache, f)
        logger.info("Saved %d name→SMILES mappings to cache", len(name_cache))
    logger.info(
        "COD: %d unique valid SMILES from %d entries (%d PubChem lookups)",
        len(smiles_list), len(entries), lookups,
    )
    return smiles_list