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
try:
    import selfies as sf
    HAS_SELFIES = True
except ImportError:
    HAS_SELFIES = False

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
    extraction_failures: int = 0  # predictions where answer couldn't be extracted
    details: dict = field(default_factory=dict)
    breakdown: dict = field(default_factory=dict)  # per-category accuracy


@dataclass
class GenerationResult:
    """Results for molecular generation evaluation"""
    exact_match: float = 0.0
    bleu: float = 0.0
    levenshtein: float = 0.0      # raw mean edit distance
    levenshtein_norm: float = 0.0  # normalized (0-1, lower=better)
    maccs_fps: float = 0.0
    rdk_fps: float = 0.0
    morgan_fps: float = 0.0
    validity: float = 0.0


# Answer Extraction Helpers
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_INTEGER_RE = re.compile(r"\b(\d+)\b")
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_STEREOCENTER_RE = re.compile(r"([A-Z][a-z]?)(\d+)\s*:\s*([RSrs])")
_DEGREE_SEQ_RE = re.compile(r"\d+(?:\s*,\s*\d+)*")


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return _THINK_RE.sub("", text).strip()


def _extract_yes_no(text: str) -> str | None:
    text = _strip_thinking(text)
    m = _YES_NO_RE.search(text)
    return m.group(1).capitalize() if m else None


def _extract_integer(text: str) -> str | None:
    text = _strip_thinking(text)
    m = _INTEGER_RE.search(text)
    return m.group(1) if m else None


def _is_selfies(text: str) -> bool:
    """Check if a string looks like SELFIES (bracketed token format)."""
    return bool(re.match(r"^\[", text.strip()) and "][" in text)


def _selfies_to_smiles(text: str) -> str:
    """Convert SELFIES string to SMILES. Returns original if conversion fails."""
    if not HAS_SELFIES:
        return text
    try:
        smiles = sf.decoder(text.strip())
        if smiles:
            return smiles
    except Exception:
        pass
    return text


def _parse_stereocenters(text: str) -> set[tuple[str, int, str]]:
    """Parse stereocenter string into a set of (symbol, index, label) tuples.

    Accepts formats like "C2:R, C5:S" or "C2:R,C5:S" or "C 2 : R, C 5 : S".
    Returns empty set if no valid centers found.
    """
    return {
        (m.group(1), int(m.group(2)), m.group(3).upper())
        for m in _STEREOCENTER_RE.finditer(text)
    }


def _extract_degree_sequence(text: str) -> list[int] | None:
    """Extract a comma-separated integer sequence from model output.

    Returns sorted list of integers, or None if nothing parseable found.
    Tolerant of whitespace around commas.
    """
    text = _strip_thinking(text)
    m = _DEGREE_SEQ_RE.search(text)
    if not m:
        return None
    try:
        return sorted(int(x.strip()) for x in m.group(0).split(","))
    except ValueError:
        return None


def _extract_smiles(text: str) -> str:
    """Extraction attempt of a SMILES string from model output"""
    # Strip thinking tags (e.g., Qwen3 hybrid thinking mode)
    text = _strip_thinking(text)
    # If output contains multiple lines
    for line in text.splitlines():
        line = line.strip()
        # Skip XML-like tags and empty lines
        if not line or line.startswith("<"):
            continue
        # Convert SELFIES to SMILES if detected
        if _is_selfies(line):
            return _selfies_to_smiles(line)
        return line
    return text



# Parsing Task Evaluation

def evaluate_parsing(
    predictions: list[str],
    references: list[str],
    task: str,
    metadata: list[dict] | None = None,
) -> ParsingResult:
    """
    Evaluate parsing task predictions against ground truth.

    :param predictions: Raw model outputs
    :param references: Ground truth answers
    :param task: Task name (determines extraction strategy)
    :param metadata: Per-sample metadata dicts for per-category breakdowns
    """
    correct = 0
    valid_smiles = 0
    extraction_failures = 0
    total = len(predictions)
    is_smiles_task = task in ("canonicalization", "fragment_assembly")

    # Per-category tracking
    category_correct: dict[str, int] = {}
    category_total: dict[str, int] = {}

    # Stereocenter: track center-level precision/recall for F1
    sc_center_tp = 0
    sc_center_pred_total = 0
    sc_center_ref_total = 0
    sc_count_correct = 0  # correct number of centers (regardless of labels)

    # Atom degree: track element-wise accuracy as partial credit
    ad_element_correct = 0
    ad_element_total = 0

    for i, (pred_raw, ref) in enumerate(zip(predictions, references)):
        # Determine category for breakdown
        cat = _get_category(task, ref, metadata[i] if metadata else None)

        category_total[cat] = category_total.get(cat, 0) + 1

        # Extract answer based on task type
        if task == "functional_group":
            pred = _extract_yes_no(pred_raw)
        elif task in ("ring_counting", "chain_length", "topological_distance"):
            pred = _extract_integer(pred_raw)
        elif task == "stereocenter":
            pred = _strip_thinking(pred_raw).strip()
        elif task == "atom_degree":
            pred = _strip_thinking(pred_raw).strip()
        else:
            pred = _extract_smiles(pred_raw)
        if pred is None:
            extraction_failures += 1
            continue

        # Check correctness
        is_correct = False
        if is_smiles_task:
            pred_mol = Chem.MolFromSmiles(pred)
            ref_mol = Chem.MolFromSmiles(ref)
            if pred_mol is not None:
                valid_smiles += 1
                pred_can = Chem.MolToSmiles(pred_mol, canonical=True)
                ref_can = Chem.MolToSmiles(ref_mol, canonical=True) if ref_mol else ref
                is_correct = pred_can == ref_can
        elif task == "stereocenter":
            pred_centers = _parse_stereocenters(pred)
            ref_centers = _parse_stereocenters(ref)
            # Set-based exact match (order-independent)
            is_correct = pred_centers == ref_centers and len(ref_centers) > 0
            # Center-level stats for F1
            tp = len(pred_centers & ref_centers)
            sc_center_tp += tp
            sc_center_pred_total += len(pred_centers)
            sc_center_ref_total += len(ref_centers)
            if len(pred_centers) == len(ref_centers):
                sc_count_correct += 1
        elif task == "atom_degree":
            pred_seq = _extract_degree_sequence(pred)
            ref_seq = _extract_degree_sequence(ref)
            if pred_seq is not None and ref_seq is not None:
                is_correct = pred_seq == ref_seq
                # Element-wise partial credit (compare sorted sequences)
                min_len = min(len(pred_seq), len(ref_seq))
                max_len = max(len(pred_seq), len(ref_seq))
                if max_len > 0:
                    matches = sum(
                        1 for a, b in zip(pred_seq, ref_seq) if a == b
                    )
                    ad_element_correct += matches
                    ad_element_total += max_len
            elif ref_seq is not None:
                ad_element_total += len(ref_seq)
        else:
            is_correct = pred.strip().lower() == ref.strip().lower()

        if is_correct:
            correct += 1
            category_correct[cat] = category_correct.get(cat, 0) + 1

    accuracy = correct / total if total > 0 else 0.0
    validity = valid_smiles / total if total > 0 and is_smiles_task else 0.0

    # Build per-category breakdown
    breakdown = {}
    for cat in sorted(category_total):
        ct = category_total[cat]
        cc = category_correct.get(cat, 0)
        breakdown[cat] = {
            "accuracy": cc / ct if ct > 0 else 0.0,
            "correct": cc,
            "total": ct,
        }

    # Extra details for structured tasks
    details: dict = {}
    if task == "stereocenter":
        sc_precision = sc_center_tp / sc_center_pred_total if sc_center_pred_total > 0 else 0.0
        sc_recall = sc_center_tp / sc_center_ref_total if sc_center_ref_total > 0 else 0.0
        sc_f1 = (2 * sc_precision * sc_recall / (sc_precision + sc_recall)
                 if (sc_precision + sc_recall) > 0 else 0.0)
        details = {
            "center_precision": round(sc_precision, 4),
            "center_recall": round(sc_recall, 4),
            "center_f1": round(sc_f1, 4),
            "count_accuracy": round(sc_count_correct / total, 4) if total > 0 else 0.0,
        }
    elif task == "atom_degree":
        details = {
            "element_accuracy": round(ad_element_correct / ad_element_total, 4) if ad_element_total > 0 else 0.0,
        }

    return ParsingResult(
        task=task,
        accuracy=accuracy,
        total=total,
        correct=correct,
        validity=validity,
        extraction_failures=extraction_failures,
        details=details,
        breakdown=breakdown,
    )


def _get_category(task: str, reference: str, meta: dict | None) -> str:
    """Determine the breakdown category for a sample."""
    if task == "functional_group" and meta:
        # Group by functional group name
        return meta.get("fg_name", "unknown")
    elif task == "ring_counting":
        # Group by answer (ring count)
        return f"count={reference}"
    elif task == "chain_length":
        # Group by chain length bucket
        try:
            length = int(reference)
            if length <= 3:
                return "short (1-3)"
            elif length <= 7:
                return "medium (4-7)"
            else:
                return "long (8+)"
        except ValueError:
            return "unknown"
    elif task in ("canonicalization", "fragment_assembly") and meta:
        # Group by SMILES length quartile
        smi = meta.get("smiles", "") or ""
        slen = len(smi)
        if slen <= 20:
            return "short (<=20)"
        elif slen <= 40:
            return "medium (21-40)"
        elif slen <= 60:
            return "long (41-60)"
        else:
            return "very long (60+)"
    elif task == "stereocenter" and meta:
        n = meta.get("n_centers", 0)
        return f"centers={n}"
    elif task == "atom_degree" and meta:
        n = meta.get("n_atoms", 0)
        if n <= 10:
            return "small (<=10)"
        elif n <= 20:
            return "medium (11-20)"
        elif n <= 30:
            return "large (21-30)"
        else:
            return "very large (30+)"
    elif task == "topological_distance" and meta:
        d = meta.get("distance", 0)
        if d <= 2:
            return "short (1-2)"
        elif d <= 5:
            return "medium (3-5)"
        else:
            return "long (6+)"
    return "all"



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
    lev_normalized: list[float] = list()
    maccs_sims: list[float] = list()
    rdk_sims: list[float] = list()
    morgan_sims: list[float] = list()
    for pred_raw, ref_raw in zip(predictions, references):
        pred = _extract_smiles(pred_raw)
        ref = _selfies_to_smiles(ref_raw) if _is_selfies(ref_raw) else ref_raw
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
        raw_lev = _compute_levenshtein(pred_can, ref_can)
        lev_distances.append(raw_lev)
        max_len = max(len(pred_can), len(ref_can))
        lev_normalized.append(raw_lev / max_len if max_len > 0 else 0.0)
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
        levenshtein_norm=float(np.mean(lev_normalized)) if lev_normalized else 0.0,
        maccs_fps=float(np.mean(maccs_sims)) if maccs_sims else 0.0,
        rdk_fps=float(np.mean(rdk_sims)) if rdk_sims else 0.0,
        morgan_fps=float(np.mean(morgan_sims)) if morgan_sims else 0.0,
        validity=validity,
    )