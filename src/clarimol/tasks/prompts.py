"""
Prompt templates and instruction formatting for SMILES parsing tasks.

Each task type has:
  - A system prompt constraining the answer format (system_prompts.yaml)
  - Multiple instruction paraphrases (instructions.yaml)
  - A message builder that produces the chat-template dict list
"""

from __future__ import annotations
import random
from pathlib import Path
import yaml
from clarimol.data.sample import Sample


_TASKS_DIR = Path(__file__).parent


def _load_yaml(filename: str) -> dict:
    with open(_TASKS_DIR / filename) as f:
        return yaml.safe_load(f)


SYSTEM_PROMPTS: dict[str, str] = _load_yaml("system_prompts.yaml")
_INSTRUCTIONS: dict[str, list[str]] = _load_yaml("instructions.yaml")


# Formatting
def _format_instruction(sample: Sample, rng: random.Random) -> str:
    """Pick a random instruction template and fill in the placeholders"""
    templates = _INSTRUCTIONS.get(sample.task, [])
    if not templates:
        return sample.question
    template = rng.choice(templates)
    # Build the substitution dict from sample fields + metadata
    fmt: dict[str, str] = {"smiles": sample.smiles}
    if sample.task == "functional_group":
        fg_name = sample.metadata.get("fg_name", "")
        fg_smarts = sample.metadata.get("fg_smarts", "")
        fmt["fg_info"] = f"{fg_name.replace('_', ' ')} ({fg_smarts})"
    elif sample.task == "ring_counting":
        fmt["ring_size"] = str(sample.metadata.get("ring_size", ""))
    return template.format_map(fmt)


def format_sample(
    sample: Sample,
    rng: random.Random | None = None,
    use_system_prompt: bool = True,
) -> dict[str, str]:
    """
    Format a Sample into a flat dict w/ 'instruction' + 'output' keys (for SFT)
    """
    if rng is None:
        rng = random.Random()
    instruction = _format_instruction(sample, rng)
    sys_prompt = SYSTEM_PROMPTS.get(sample.task, "")
    if use_system_prompt and sys_prompt:
        instruction = f"{sys_prompt}\n\n{instruction}"
    return {
        "instruction": instruction,
        "output": sample.answer,
    }


def build_messages(
    sample: Sample,
    rng: random.Random | None = None,
    use_system_prompt: bool = True,
) -> list[dict[str, str]]:
    """
    Format a Sample into chat-style messages for a chat-template tokenizer.
    :return: list of {'role': ..., 'content': ...} dicts
    """
    if rng is None:
        rng = random.Random()
    instruction = _format_instruction(sample, rng)
    messages: list[dict[str, str]] = []
    sys_prompt = SYSTEM_PROMPTS.get(sample.task, "")
    if use_system_prompt and sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": sample.answer})
    return messages
