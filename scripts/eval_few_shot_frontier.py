"""
Few-shot evaluation of frontier LLMs on SMILES parsing tasks.

Evaluates frontier LLMs (GPT-4o, Claude Sonnet) on the five core CLARIMOL
parsing tasks using 0-shot, 3-shot, and 5-shot in-context learning. Few-shot
examples are drawn deterministically from the training set (seed=42). Results
are saved per model × shot-count to output/few_shot/{model_name}/{n_shot}/results.json.

API keys: OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables.
Run with --dry-run to estimate API costs without making calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

# Add src to path so clarimol imports work when run from scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from clarimol.data.sample import Sample
from clarimol.eval.metrics import evaluate_parsing
from clarimol.tasks.prompts import build_messages, SYSTEM_PROMPTS

logger = logging.getLogger(__name__)

TASKS = [
    "functional_group",
    "ring_counting",
    "chain_length",
    "canonicalization",
    "fragment_assembly",
]

SAMPLES_PER_TASK = 200
FEW_SHOT_COUNTS = [0, 1, 5]
CHECKPOINT_INTERVAL = 50
MAX_REQUESTS_PER_SECOND = 10
SEED = 42


# Cost estimates (USD per 1M tokens, as of mid-2026 — verify before use)
_COST_PER_1M_IN = {
    "claude": 3.0,
    "gpt4o": 2.5,
    "gpt4o-mini": 0.15,
    "gemini-flash": 0.10,
    "deepseek": 0.27,
}
_COST_PER_1M_OUT = {
    "claude": 15.0,
    "gpt4o": 10.0,
    "gpt4o-mini": 0.60,
    "gemini-flash": 0.40,
    "deepseek": 1.10,
}

# Approximate token counts per request
_AVG_INPUT_TOKENS = {
    0: 120,
    3: 420,
    5: 640,
}
_AVG_OUTPUT_TOKENS = 10


@dataclass
class EvalConfig:
    """Configuration for a single evaluation run."""

    model_key: str
    n_shot: int
    task_name: str
    test_data_dir: Path
    train_data_dir: Path
    output_dir: Path
    dry_run: bool = False
    randomize: bool = False


@dataclass
class SampleResult:
    """Per-sample prediction record."""

    smiles: str
    question: str
    reference: str
    prediction: str
    correct: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_json_samples(path: Path) -> list[Sample]:
    """Load a JSON array of samples from disk, returning Sample dataclass instances."""
    with open(path) as f:
        raw = json.load(f)
    return [
        Sample(
            smiles=entry["smiles"],
            task=entry["task"],
            question=entry["question"],
            answer=entry["answer"],
            difficulty=entry.get("difficulty", 0.0),
            metadata=entry.get("metadata", {}),
        )
        for entry in raw
    ]


def _randomize_smiles(smiles: str) -> str:
    """Produce a non-canonical SMILES string via RDKit random traversal.

    For fragment assembly (two fragments separated by ' . '), each
    fragment is randomized independently.
    """
    from rdkit import Chem

    if " . " in smiles:
        parts = smiles.split(" . ")
        randomized = []
        for part in parts:
            mol = Chem.MolFromSmiles(part)
            if mol is not None:
                randomized.append(Chem.MolToSmiles(mol, doRandom=True))
            else:
                randomized.append(part)
        return " . ".join(randomized)

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, doRandom=True)
    return smiles


def _subsample(samples: list[Sample], n: int, rng: random.Random) -> list[Sample]:
    """Return a deterministic random subsample of n items."""
    if len(samples) <= n:
        return list(samples)
    return rng.sample(samples, n)


def _select_few_shot_examples(
    train_samples: list[Sample],
    n: int,
    rng: random.Random,
) -> list[Sample]:
    """Pick n examples from training data deterministically."""
    return rng.sample(train_samples, min(n, len(train_samples)))


def _build_few_shot_messages(
    query_sample: Sample,
    examples: list[Sample],
    rng: random.Random,
) -> list[dict[str, str]]:
    """
    Build chat messages for a query sample with optional few-shot examples prepended.

    System prompt is added once at the start. Each example is formatted as a
    user/assistant exchange using the same build_messages() used in SFT training.
    The final user turn contains only the query (no assistant turn appended).
    """
    messages: list[dict[str, str]] = []

    # System prompt for the task
    sys_prompt = SYSTEM_PROMPTS.get(query_sample.task, "")
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})

    # Few-shot examples: each appears as user → assistant exchange
    for ex in examples:
        ex_msgs = build_messages(ex, rng=rng, use_system_prompt=False)
        for msg in ex_msgs:
            messages.append(msg)

    # Query: user turn only (no assistant answer)
    query_msgs = build_messages(query_sample, rng=rng, use_system_prompt=False)
    for msg in query_msgs:
        if msg["role"] != "assistant":
            messages.append(msg)

    return messages


class _RateLimiter:
    """Token-bucket rate limiter for API calls."""

    def __init__(self, max_per_second: float) -> None:
        self._min_interval = 1.0 / max_per_second
        self._last_call: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


def _exponential_backoff(
    fn: Callable[[], str],
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> str:
    """
    Call fn() with exponential backoff on rate-limit or transient errors.

    Raises the last exception if all retries are exhausted.
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            name = type(exc).__name__
            # Identify retryable errors by name to avoid hard-coding SDK types
            is_rate_limit = "RateLimit" in name or "rate_limit" in str(exc).lower()
            is_transient = "Timeout" in name or "Connection" in name or "Server" in name
            if attempt < max_retries - 1 and (is_rate_limit or is_transient):
                jitter = random.uniform(0, 0.3 * delay)
                wait_time = delay + jitter
                logger.warning(
                    "API error (%s) on attempt %d/%d, retrying in %.1fs",
                    name, attempt + 1, max_retries, wait_time,
                )
                time.sleep(wait_time)
                delay *= 2.0
            else:
                raise
    raise RuntimeError("Exhausted retries")  # unreachable but satisfies type checker


class ClaudeClient:
    """Thin wrapper around the Anthropic SDK for single-turn completions."""

    MODEL_ID = "claude-sonnet-4-20250514"
    DISPLAY_NAME = "claude-sonnet"

    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 128) -> str:
        """Send messages and return the assistant text response."""
        system_content = ""
        filtered: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                filtered.append(msg)

        def _call() -> str:
            kwargs: dict[str, Any] = {
                "model": self.MODEL_ID,
                "max_tokens": max_tokens,
                "messages": filtered,
            }
            if system_content:
                kwargs["system"] = system_content
            response = self._client.messages.create(**kwargs)
            return response.content[0].text.strip()

        return _exponential_backoff(_call)


class GPT4oClient:
    """Thin wrapper around the OpenAI SDK for single-turn completions."""

    MODEL_ID = "gpt-4o"
    DISPLAY_NAME = "gpt-4o"

    def __init__(self) -> None:
        import openai
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 128) -> str:
        """Send messages and return the assistant text response."""

        def _call() -> str:
            response = self._client.chat.completions.create(
                model=self.MODEL_ID,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        return _exponential_backoff(_call)


class GPT4oMiniClient(GPT4oClient):
    """GPT-4o-mini: cheaper, faster, good for cost-effective sweeps."""
    MODEL_ID = "gpt-4o-mini"
    DISPLAY_NAME = "gpt-4o-mini"


class GeminiFlashClient:
    """Google Gemini Flash via the google-genai SDK."""

    MODEL_ID = "gemini-2.0-flash"
    DISPLAY_NAME = "gemini-flash"

    def __init__(self) -> None:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self._model = genai.GenerativeModel(self.MODEL_ID)

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 128) -> str:
        import google.generativeai as genai

        # Convert chat messages to Gemini format
        parts = []
        for msg in messages:
            role = "user" if msg["role"] in ("user", "system") else "model"
            parts.append({"role": role, "parts": [msg["content"]]})

        def _call() -> str:
            response = self._model.generate_content(
                parts,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens, temperature=0.0
                ),
            )
            return response.text.strip()

        return _exponential_backoff(_call)


class DeepSeekClient:
    """DeepSeek via OpenAI-compatible API."""

    MODEL_ID = "deepseek-chat"
    DISPLAY_NAME = "deepseek"

    def __init__(self) -> None:
        import openai
        self._client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 128) -> str:
        def _call() -> str:
            response = self._client.chat.completions.create(
                model=self.MODEL_ID,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        return _exponential_backoff(_call)


def _get_client(model_key: str):
    """Instantiate the appropriate API client for a model key."""
    clients = {
        "claude": ClaudeClient,
        "gpt4o": GPT4oClient,
        "gpt4o-mini": GPT4oMiniClient,
        "gemini-flash": GeminiFlashClient,
        "deepseek": DeepSeekClient,
    }
    if model_key not in clients:
        raise ValueError(f"Unknown model key: {model_key!r}. Options: {list(clients)}")
    return clients[model_key]()


def _checkpoint_path(output_dir: Path, task: str) -> Path:
    return output_dir / f"_checkpoint_{task}.json"


def _load_checkpoint(output_dir: Path, task: str) -> list[SampleResult]:
    """Load partial results for a task from disk, or return empty list."""
    p = _checkpoint_path(output_dir, task)
    if p.exists():
        with open(p) as f:
            raw = json.load(f)
        return [SampleResult(**r) for r in raw]
    return []


def _save_checkpoint(output_dir: Path, task: str, results: list[SampleResult]) -> None:
    """Persist partial results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    p = _checkpoint_path(output_dir, task)
    with open(p, "w") as f:
        json.dump([asdict(r) for r in results], f)


def _remove_checkpoint(output_dir: Path, task: str) -> None:
    p = _checkpoint_path(output_dir, task)
    if p.exists():
        p.unlink()


def _estimate_cost(model_key: str, n_shot: int, n_tasks: int, samples_per_task: int) -> float:
    """Rough USD cost estimate for an evaluation run."""
    total_requests = n_tasks * samples_per_task
    in_tokens = total_requests * _AVG_INPUT_TOKENS.get(n_shot, 200)
    out_tokens = total_requests * _AVG_OUTPUT_TOKENS
    cost_in = in_tokens / 1_000_000 * _COST_PER_1M_IN[model_key]
    cost_out = out_tokens / 1_000_000 * _COST_PER_1M_OUT[model_key]
    return cost_in + cost_out


def evaluate_task(
    client: ClaudeClient | GPT4oClient,
    config: EvalConfig,
    limiter: _RateLimiter,
) -> dict[str, Any]:
    """
    Evaluate a single task for a given model and shot count.

    Returns a dict with keys: accuracy, correct, total.
    """
    task = config.task_name
    test_path = config.test_data_dir / f"{task}.json"
    train_path = config.train_data_dir / f"{task}.json"

    if not test_path.exists():
        logger.warning("Test data not found for task %s at %s, skipping.", task, test_path)
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    rng_subsample = random.Random(SEED)
    rng_fewshot = random.Random(SEED)
    rng_prompt = random.Random(SEED)

    test_samples = _load_json_samples(test_path)
    test_subset = _subsample(test_samples, SAMPLES_PER_TASK, rng_subsample)

    train_samples: list[Sample] = []
    if config.n_shot > 0 and train_path.exists():
        all_train = _load_json_samples(train_path)
        # Draw a pool to sample few-shot examples from — larger than n_shot so
        # per-query sampling has variety, but fixed by seed for reproducibility.
        pool_size = min(500, len(all_train))
        train_pool = rng_fewshot.sample(all_train, pool_size)
        train_samples = train_pool
    elif config.n_shot > 0:
        logger.warning(
            "Training data not found for task %s at %s; running 0-shot.", task, train_path
        )

    # Load checkpoint (allows resuming after interruption)
    task_output_dir = config.output_dir / task
    task_output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[SampleResult] = _load_checkpoint(task_output_dir, task)
    n_completed = len(completed)

    if n_completed > 0:
        logger.info(
            "Resuming task %s from checkpoint: %d/%d already done.",
            task, n_completed, len(test_subset),
        )

    predictions: list[str] = [r.prediction for r in completed]
    references: list[str] = [r.reference for r in completed]
    metadata_list: list[dict] = [r.metadata for r in completed]

    for idx, sample in enumerate(test_subset):
        if idx < n_completed:
            continue  # skip already-completed samples

        # Select few-shot examples: for each query, draw deterministically
        # by advancing a per-sample seed derived from the global seed and index.
        examples: list[Sample] = []
        if config.n_shot > 0 and train_samples:
            per_sample_rng = random.Random(SEED + idx)
            examples = _select_few_shot_examples(train_samples, config.n_shot, per_sample_rng)

        # Optionally randomize the test sample's SMILES
        if config.randomize:
            from dataclasses import replace as dc_replace
            rand_smi = _randomize_smiles(sample.smiles)
            sample = dc_replace(sample, smiles=rand_smi)

        messages = _build_few_shot_messages(sample, examples, rng=rng_prompt)

        if config.dry_run:
            prediction = "[DRY RUN]"
        else:
            limiter.wait()
            try:
                prediction = client.complete(messages, max_tokens=128)
            except Exception as exc:
                logger.error(
                    "API call failed for task %s sample %d: %s. Storing empty prediction.",
                    task, idx, exc,
                )
                prediction = ""

        predictions.append(prediction)
        references.append(sample.answer)
        metadata_list.append(sample.metadata)

        completed.append(
            SampleResult(
                smiles=sample.smiles,
                question=sample.question,
                reference=sample.answer,
                prediction=prediction,
                correct=False,  # filled after full evaluation below
                metadata=sample.metadata,
            )
        )

        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(task_output_dir, task, completed)
            logger.info("Checkpoint saved at sample %d/%d for task %s.", idx + 1, len(test_subset), task)

    # Evaluate
    result = evaluate_parsing(predictions, references, task, metadata=metadata_list)

    logger.info(
        "Task %s [%d-shot]: accuracy=%.4f (%d/%d), extraction_failures=%d",
        task, config.n_shot, result.accuracy, result.correct, result.total,
        result.extraction_failures,
    )

    # Save detailed per-sample results
    per_sample_path = task_output_dir / "per_sample.json"
    per_sample_path.write_text(json.dumps([asdict(r) for r in completed], indent=2))

    _remove_checkpoint(task_output_dir, task)

    return {
        "accuracy": round(result.accuracy, 6),
        "correct": result.correct,
        "total": result.total,
        "validity": round(result.validity, 6),
        "extraction_failures": result.extraction_failures,
    }


def run_evaluation(
    model_key: str,
    n_shot: int,
    test_data_dir: Path,
    train_data_dir: Path,
    base_output_dir: Path,
    dry_run: bool = False,
    randomize: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Run evaluation for one model × shot-count combination across all tasks.

    Returns dict mapping task_name → metrics dict.
    """
    display_names = {
        "claude": "claude-sonnet", "gpt4o": "gpt-4o", "gpt4o-mini": "gpt-4o-mini",
        "gemini-flash": "gemini-flash", "deepseek": "deepseek",
    }
    display_name = display_names.get(model_key, model_key)
    output_dir = base_output_dir / display_name / str(n_shot)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting evaluation: model=%s, n_shot=%d, output=%s",
        display_name, n_shot, output_dir,
    )

    if not dry_run:
        client = _get_client(model_key)
    else:
        client = None  # type: ignore[assignment]

    limiter = _RateLimiter(MAX_REQUESTS_PER_SECOND)
    all_results: dict[str, dict[str, Any]] = {}

    # Load existing results.json to support resuming at task granularity
    results_path = output_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        logger.info("Loaded %d existing task results from %s.", len(all_results), results_path)

    for task in TASKS:
        if task in all_results:
            logger.info("Skipping already-completed task: %s.", task)
            continue

        config = EvalConfig(
            model_key=model_key,
            n_shot=n_shot,
            task_name=task,
            test_data_dir=test_data_dir,
            train_data_dir=train_data_dir,
            output_dir=output_dir,
            dry_run=dry_run,
            randomize=randomize,
        )

        task_result = evaluate_task(client, config, limiter)
        all_results[task] = task_result

        # Persist after each task so partial results survive early exit
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Saved results.json after task %s.", task)

    logger.info(
        "Completed: model=%s, n_shot=%d. Results at %s",
        display_name, n_shot, results_path,
    )
    return all_results


def dry_run_report(model_keys: list[str], n_shots: list[int]) -> None:
    """Print cost estimates for the planned evaluation without making API calls."""
    print("\n=== DRY RUN: Cost Estimate ===\n")
    total_cost = 0.0
    header = f"{'Model':<20} {'N-Shot':<10} {'Requests':<12} {'Est. Cost (USD)':<18}"
    print(header)
    print("-" * len(header))
    for mkey in model_keys:
        _display_map = {
            "claude": "claude-sonnet", "gpt4o": "gpt-4o", "gpt4o-mini": "gpt-4o-mini",
            "gemini-flash": "gemini-flash", "deepseek": "deepseek",
        }
        display = _display_map.get(mkey, mkey)
        for n_shot in n_shots:
            n_requests = len(TASKS) * SAMPLES_PER_TASK
            cost = _estimate_cost(mkey, n_shot, len(TASKS), SAMPLES_PER_TASK)
            total_cost += cost
            print(f"{display:<20} {n_shot:<10} {n_requests:<12} ${cost:.4f}")
    print("-" * len(header))
    print(f"{'TOTAL':<20} {'':<10} {'':<12} ${total_cost:.4f}")
    print(
        "\nNote: cost estimates use approximate token counts and mid-2026 pricing."
        " Verify current rates before large runs.\n"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Few-shot evaluation of frontier LLMs on CLARIMOL SMILES parsing tasks."
    )
    parser.add_argument(
        "--models",
        default="claude,gpt4o",
        help="Comma-separated list of models to evaluate. Options: claude, gpt4o, gpt4o-mini, gemini-flash, deepseek.",
    )
    parser.add_argument(
        "--shots",
        default="0,3,5",
        help="Comma-separated list of few-shot counts to evaluate.",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=_REPO_ROOT / "data" / "test",
        help="Directory containing per-task test JSON files.",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=_REPO_ROOT / "data" / "clarimol",
        help="Directory containing per-task training JSON files (source of few-shot examples).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "output" / "few_shot",
        help="Root output directory. Results saved to {output_dir}/{model}/{n_shot}/results.json.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize SMILES before evaluation (tests robustness to non-canonical input).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cost estimate and exit without making API calls.",
    )
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    n_shots = [int(s.strip()) for s in args.shots.split(",") if s.strip()]

    # Validate model keys
    valid_keys = {"claude", "gpt4o", "gpt4o-mini", "gemini-flash", "deepseek"}
    invalid = set(model_keys) - valid_keys
    if invalid:
        parser.error(f"Unknown model keys: {invalid}. Valid options: {valid_keys}")

    if args.dry_run:
        dry_run_report(model_keys, n_shots)
        return

    # Check API key availability; drop models whose key is absent
    active_models: list[str] = []
    for mkey in model_keys:
        env_var = {"claude": "ANTHROPIC_API_KEY", "gpt4o": "OPENAI_API_KEY"}[mkey]
        if env_var not in os.environ:
            logger.warning(
                "Skipping model %r: environment variable %s is not set.", mkey, env_var
            )
        else:
            active_models.append(mkey)

    if not active_models:
        logger.error(
            "No API keys found. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY, or use --dry-run."
        )
        sys.exit(1)

    for mkey in active_models:
        for n_shot in n_shots:
            run_evaluation(
                model_key=mkey,
                n_shot=n_shot,
                test_data_dir=args.test_data,
                train_data_dir=args.train_data,
                base_output_dir=args.output_dir,
                dry_run=False,
                randomize=args.randomize,
            )

    logger.info("All evaluations complete.")


if __name__ == "__main__":
    main()
