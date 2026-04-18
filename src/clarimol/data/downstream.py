"""
Mol-Instructions downstream task data loading.

Supports three molecular generation tasks from Fang et al., 2024:
    - retrosynthesis
    - reagent_prediction
    - forward_reaction_prediction

Data uses SELFIES-style bracketed tokens as in the original Mol-Instructions.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DOWNSTREAM_TASKS = [
    "retrosynthesis",
    "reagent_prediction",
    "forward_reaction_prediction",
]


@dataclass
class DownstreamSample:
    """A single downstream task example."""
    task: str
    instruction: str
    input: str
    output: str
    split: str  # "train" or "test"


def load_mol_instructions(
    data_dir: str | Path,
    tasks: list[str] | None = None,
    split: str | None = None,
) -> dict[str, list[DownstreamSample]]:
    """
    Load Mol-Instructions JSON files.

    :param data_dir: Directory containing {task}.json files
    :param tasks: Which tasks to load (default: all three)
    :param split: Filter by split ("train" or "test"), or None for both
    :return: Dict mapping task name to list of samples
    """
    data_dir = Path(data_dir)
    if tasks is None:
        tasks = DOWNSTREAM_TASKS
    result: dict[str, list[DownstreamSample]] = {}
    for task in tasks:
        path = data_dir / f"{task}.json"
        if not path.exists():
            logger.warning("Missing %s, skipping", path)
            continue
        with open(path) as f:
            raw = json.load(f)
        samples = []
        for entry in raw:
            s = DownstreamSample(
                task=task,
                instruction=entry["instruction"],
                input=entry["input"],
                output=entry["output"],
                split=entry.get("metadata", {}).get("split", "train"),
            )
            if split is None or s.split == split:
                samples.append(s)
        result[task] = samples
        logger.info("Loaded %s: %d samples (split=%s)", task, len(samples), split or "all")
    return result


def build_downstream_messages(
    sample: DownstreamSample,
    use_system_prompt: bool = True,
) -> list[dict[str, str]]:
    """Format a downstream sample into chat messages."""
    messages: list[dict[str, str]] = []
    if use_system_prompt:
        messages.append({
            "role": "system",
            "content": "You are a helpful chemistry assistant. Answer with only "
                       "the molecular string without any other information.",
        })
    messages.append({
        "role": "user",
        "content": f"{sample.instruction}\n{sample.input}",
    })
    messages.append({
        "role": "assistant",
        "content": sample.output,
    })
    return messages
