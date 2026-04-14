"""
Evaluation Metrics for CLARIMOL
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Levenshtein import distance

RDLogger.logger().setLevel(RDLogger.ERROR)

logger = logging.getLogger(__name__)


# Result Containers

@dataclass
class ParsingResult:
    """Results for a single parsing task evaluation"""
    task: str
    accuracy: float
    total: int
    correct: int
    validity: float = 0.0  # for SMILES-output tasks
    details: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Results for molecular generation evaluation"""
    exact_match: float = 0.0
    bleu: float = 0.0
    levenshtein: float = 0.0
    maccs_fps: float = 0.0
    rdk_fps: float = 0.0
    morgan_fps: float = 0.0
    validity: float = 0.0


# Answer Extraction Helpers
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_INTEGER_RE = re.compile(r"\b(\d+)\b")


def _extract_yes_no(text: str) -> str | None:
    m = _YES_NO_RE.search(text)
    return m.group(1).capitalize() if m else None


def _extract_integer(text: str) -> str | None:
    m = _INTEGER_RE.search(text)
    return m.group(1) if m else None


def _extract_smiles(text: str) -> str:
    """Extraction attempt of a SMILES string from model output"""
    text = text.strip()
    # If output contains multiple lines
    for line in text.splitlines():
        line = line.strip()
        # take the first non-empty one
        if line:
            return line
    return text



# Parsing Task Evaluation

def evaluate_parsing(
    predictions: list[str],
    references: list[str],
    task: str,
) -> ParsingResult:
    """
    Evaluate parsing task predictions against ground truth.

    :param: predictions: Raw model outputs
    :param: Ground truth answers
    :param: Task name (determines extraction strategy)
    """
    correct = 0
    valid_smiles = 0
    total = len(predictions)
    is_smiles_task = task in ("canonicalization", "fragment_assembly")
    for pred_raw, ref in zip(predictions, references):
        # Extract answer based on task type
        if task == "functional_group":
            pred = _extract_yes_no(pred_raw)
        elif task in ("ring_counting", "chain_length"):
            pred = _extract_integer(pred_raw)
        else:
            pred = _extract_smiles(pred_raw)
        if pred is None:
            continue
        # Check correctness
        if is_smiles_task:
            # Canonicalize both for fair comparison
            pred_mol = Chem.MolFromSmiles(pred)
            ref_mol = Chem.MolFromSmiles(ref)
            if pred_mol is not None:
                valid_smiles += 1
                pred_can = Chem.MolToSmiles(pred_mol, canonical=True)
                ref_can = Chem.MolToSmiles(ref_mol, canonical=True) if ref_mol else ref
                if pred_can == ref_can:
                    correct += 1
        else:
            if pred.strip().lower() == ref.strip().lower():
                correct += 1
    accuracy = correct / total if total > 0 else 0.0
    validity = valid_smiles / total if total > 0 and is_smiles_task else 0.0
    return ParsingResult(
        task=task,
        accuracy=accuracy,
        total=total,
        correct=correct,
        validity=validity,
    )



# Molecular Generation Evaluation

def _smiles_validity(smiles_list: list[str]) -> float:
    """Fraction of valid SMILES"""
    valid = sum(1 for s in smiles_list if Chem.MolFromSmiles(s) is not None)
    return valid / len(smiles_list) if smiles_list else 0.0


def _tanimoto_similarity(mol1: Chem.Mol, mol2: Chem.Mol, fp_type: str) -> float:
    """Compute Tanimoto similarity for a given fingerprint type"""
    if fp_type == "maccs":
        fp1 = MACCSkeys.GenMACCSKeys(mol1)
        fp2 = MACCSkeys.GenMACCSKeys(mol2)
    elif fp_type == "rdk":
        fp1 = RDKFingerprint(mol1)
        fp2 = RDKFingerprint(mol2)
    elif fp_type == "morgan":
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def _compute_bleu(pred: str, ref: str) -> float:
    """Character-level BLEU for SMILES strings"""
    # Character-level tokenization
    pred_chars = list(pred)
    ref_chars = list(ref)
    if not pred_chars or not ref_chars:
        return 0.0
    return sentence_bleu(
        [ref_chars],
        pred_chars,
        smoothing_function=SmoothingFunction().method1,
    )


def _compute_levenshtein(pred: str, ref: str) -> int:
    """Levenshtein edit distance"""
    return distance(pred, ref)


def evaluate_generation(
    predictions: list[str],
    references: list[str],
) -> GenerationResult:
    """
    Evaluation for molecular generation tasks.

    :param: predictions: Generated SMILES strings
    :param: references: Ground truth SMILES strings
    """
    n = len(predictions)
    if n == 0:
        return GenerationResult()
    exact = 0
    bleu_scores: list[float] = list()
    lev_distances: list[int] = list()
    maccs_sims: list[float] = list()
    rdk_sims: list[float] = list()
    morgan_sims: list[float] = list()
    for pred_raw, ref in zip(predictions, references):
        pred = _extract_smiles(pred_raw)
        pred_mol = Chem.MolFromSmiles(pred)
        ref_mol = Chem.MolFromSmiles(ref)
        # Canonicalize for exact match
        if pred_mol and ref_mol:
            pred_can = Chem.MolToSmiles(pred_mol, canonical=True)
            ref_can = Chem.MolToSmiles(ref_mol, canonical=True)
            if pred_can == ref_can:
                exact += 1
        else:
            pred_can = pred
            ref_can = ref
        bleu_scores.append(_compute_bleu(pred_can, ref_can))
        lev_distances.append(_compute_levenshtein(pred_can, ref_can))
        # Fingerprint similarities (only if both are valid)
        if pred_mol and ref_mol:
            maccs_sims.append(_tanimoto_similarity(pred_mol, ref_mol, "maccs"))
            rdk_sims.append(_tanimoto_similarity(pred_mol, ref_mol, "rdk"))
            morgan_sims.append(_tanimoto_similarity(pred_mol, ref_mol, "morgan"))
    valid_preds = [_extract_smiles(p) for p in predictions]
    validity = _smiles_validity(valid_preds)
    return GenerationResult(
        exact_match=exact / n,
        bleu=float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        levenshtein=float(np.mean(lev_distances)) if lev_distances else 0.0,
        maccs_fps=float(np.mean(maccs_sims)) if maccs_sims else 0.0,
        rdk_fps=float(np.mean(rdk_sims)) if rdk_sims else 0.0,
        morgan_fps=float(np.mean(morgan_sims)) if morgan_sims else 0.0,
        validity=validity,
    )