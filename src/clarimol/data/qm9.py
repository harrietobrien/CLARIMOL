"""QM9 dataset loader — 134K small organic molecules from HuggingFace."""

from __future__ import annotations
import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)


def fetch_qm9(max_molecules: int | None = None) -> list[str]:
    """
    Load SMILES from the QM9 dataset (n0w0f/qm9-csv on HuggingFace).
    ~133,885 small organic molecules (up to 9 heavy atoms: C, H, O, N, F).
    """
    logger.info("Loading QM9 from HuggingFace (n0w0f/qm9-csv)")
    ds = load_dataset("n0w0f/qm9-csv", split="train")
    smiles = [s for s in ds["smiles"] if s and s.strip()]
    logger.info("QM9: %d SMILES loaded", len(smiles))
    if max_molecules is not None:
        smiles = smiles[:max_molecules]
    return smiles
