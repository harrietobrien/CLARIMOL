"""
Prompt templates and instruction formatting for SMILES parsing tasks.

Each task type has:
  - A system prompt constraining the answer format
  - Multiple instruction paraphrases (selected randomly during data construction)
  - A message builder that produces the chat-template dict list
"""

from __future__ import annotations
import random
from clarimol.data.sample import Sample



# System prompts — constrain answer format per task type
SYSTEM_PROMPTS: dict[str, str] = {
    "functional_group": "Answer only in 'Yes' or 'No' without any other information.",
    "ring_counting": "Answer only with the corresponding integer number without any other information.",
    "chain_length": "Answer only with the corresponding integer number without any other information.",
    "canonicalization": "Answer only with the corresponding SMILES string without any other information.",
    "fragment_assembly": "Answer only with the corresponding SMILES string without any other information.",
}


# Instruction paraphrases — reduces overfitting to specific phrasings
_INSTRUCTIONS: dict[str, list[str]] = {
    "functional_group": [
        (
            "Does the molecule represented by this SMILES contain "
            "the specified functional group? Respond with 'Yes' or 'No'.\n"
            "**SMILES:** {smiles}\n**FUNCTIONAL GROUP:** {fg_info}"
        ),
        (
            "Determine the inclusion of the indicated functional group "
            "in the given SMILES string. Answer 'Yes' or 'No'.\n"
            "**SMILES:** {smiles}\n**FUNCTIONAL GROUP:** {fg_info}"
        ),
        (
            "Given the SMILES below, determine whether the functional group "
            "{fg_info} is present. Respond 'Yes' or 'No'.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Check if the molecule encoded by this SMILES includes the "
            "functional group {fg_info}. Answer with 'Yes' or 'No' only.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Is the functional group {fg_info} contained in the molecule "
            "represented by the following SMILES? Answer 'Yes' or 'No'.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Examine the SMILES string below and indicate whether it contains "
            "the functional group {fg_info}. Respond with only 'Yes' or 'No'.\n"
            "**SMILES:** {smiles}"
        ),
    ],
    "ring_counting": [
        (
            "Assess the SMILES below and report how many rings consist of "
            "{ring_size} atoms. Give me the integer only.\n"
            "**SMILES:** {smiles}\n**SIZE OF RINGS:** {ring_size}"
        ),
        (
            "Calculate the count of {ring_size}-membered rings in the given "
            "SMILES string.\n"
            "**SMILES:** {smiles}\n**RING SIZE:** {ring_size}"
        ),
        (
            "How many {ring_size}-membered rings does the molecule encoded by "
            "this SMILES contain? Provide only the number.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Count the number of rings with exactly {ring_size} atoms in the "
            "molecule below. Answer with a single integer.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "For the molecule represented by the following SMILES, determine "
            "how many {ring_size}-membered rings are present. Integer only.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Tell me the number of {ring_size}-membered rings in the molecule "
            "described by this SMILES string. Answer with just the count.\n"
            "**SMILES:** {smiles}"
        ),
    ],
    "chain_length": [
        (
            "Report the size of the largest carbon-only chain not contained "
            "within a ring in the molecule represented by this SMILES. "
            "Answer with an integer only.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "What is the length of the longest acyclic carbon chain in the "
            "molecule encoded by this SMILES? Give the integer only.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Determine the longest straight carbon chain (excluding ring atoms) "
            "in the following molecule. Provide only the integer length.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "For the molecule below, find the longest chain composed exclusively "
            "of carbon atoms that are not part of any ring. Answer with an integer.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Measure the length of the longest non-ring carbon chain in the "
            "molecule represented by this SMILES. Respond with the count only.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Identify the longest acyclic carbon chain in this molecule and "
            "report its length as a single integer.\n"
            "**SMILES:** {smiles}"
        ),
    ],
    "canonicalization": [
        (
            "Give me a canonicalized SMILES string that represents the same "
            "molecule as the given one.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Convert the following SMILES to its canonical form.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Provide the canonical SMILES representation for the molecule "
            "encoded by this SMILES string.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Transform the SMILES below into its standard canonical form. "
            "Answer with only the canonical SMILES.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Produce the canonical SMILES equivalent of the given molecular "
            "string.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Rewrite the following SMILES in canonical form. Return only "
            "the resulting SMILES.\n"
            "**SMILES:** {smiles}"
        ),
    ],
    "fragment_assembly": [
        (
            "Connect the following two SMILES fragments into a unified "
            "structure at their reactive sites.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Combine the two molecular fragments below into a single valid "
            "molecule. Answer with the resulting SMILES only.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Assemble the following pair of SMILES fragments into one "
            "complete molecule by joining at the indicated attachment points.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Given two disconnected SMILES fragments, reconstruct the full "
            "molecule by connecting them at their reactive sites.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Join these two molecular fragments into one structure. Provide "
            "the complete SMILES string.\n"
            "**SMILES:** {smiles}"
        ),
        (
            "Merge the two SMILES fragments below into a single valid "
            "molecular structure.\n"
            "**SMILES:** {smiles}"
        ),
    ],
}



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