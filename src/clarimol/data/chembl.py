"""ChEMBL dataset loader — ~1.9M bioactive molecules from HuggingFace."""

from __future__ import annotations
import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)


def fetch_chembl(max_molecules: int | None = None) -> list[str]:
    """
    Load SMILES from ChEMBL (antoinebcx/smiles-molecules-chembl on HuggingFace).
    ~1.94M bioactive drug-like molecules, pre-cleaned.
    """
    logger.info("Loading ChEMBL from HuggingFace (antoinebcx/smiles-molecules-chembl)")
    ds = load_dataset("antoinebcx/smiles-molecules-chembl")
    # Combine all splits
    smiles: list[str] = []
    for split in ds:
        smiles.extend(s for s in ds[split]["smiles"] if s and s.strip())
    logger.info("ChEMBL: %d SMILES loaded", len(smiles))
    if max_molecules is not None:
        smiles = smiles[:max_molecules]
    return smiles
