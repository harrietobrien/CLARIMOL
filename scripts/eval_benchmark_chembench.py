"""
ChemBench evaluation (Mirza et al., Nature Chemistry 2025).

Evaluates base vs CLARIMOL-trained models on ChemBench: 2,700+ QA pairs
across chemistry topics. Uses ChemBench's custom model wrapper API.

Paper: https://www.nature.com/articles/s41557-025-01815-x
GitHub: https://github.com/lamalab-org/chembench
Docs: https://lamalab-org.github.io/chembench/

ChemBench expects a model wrapper class with a .generate() method that
accepts a list of prompts and returns a Generations object. The wrapper
is passed to PrompterBuilder.from_model_object() to create a prompter.

TODO (verify before first run):
  1. Confirm import path: "from chembench.types import Generation, Generations"
     -- some versions may use "from chembench.utils import ..."
     -- if chembench.types does not exist, check chembench source for the
        actual module containing Generation/Generations dataclasses.
  2. Confirm PrompterBuilder.from_model_object() accepts a custom object
     (not just a LiteLLM model string). The docs show both patterns.
  3. Confirm save_topic_reports vs save_topic_results -- both names
     appear in different documentation versions.
  4. Confirm ChemBenchmark.from_huggingface() is the correct loader
     -- may require "jablonkagroup/ChemBench" as an argument.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def build_model_wrapper(base_model_name: str, adapter_path: str | None = None):
    """
    Build a ChemBench-compatible model wrapper.

    ChemBench expects:
      - A .generate(prompts: List[str], **kwargs) -> Generations method
      - Generations wraps a list of lists of Generation objects
      - Each Generation has a .text attribute

    The wrapper loads the model once and reuses it for all prompts.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # Import ChemBench types.
    # TODO: If this import fails, try alternative paths:
    #   from chembench.utils import Generation, Generations
    #   from langchain_core.outputs import Generation, Generations
    # ChemBench may use langchain's Generation type internally.
    try:
        from chembench.types import Generation, Generations
        logger.info("Imported Generation/Generations from chembench.types")
    except ImportError:
        try:
            from langchain_core.outputs import Generation, Generations
            logger.info("Imported Generation/Generations from langchain_core.outputs")
        except ImportError:
            logger.error(
                "Could not import Generation/Generations from chembench.types "
                "or langchain_core.outputs. Install chembench or check the "
                "correct import path in the chembench source."
            )
            raise

    tok_source = adapter_path if adapter_path else base_model_name
    logger.info("Loading tokenizer from %s", tok_source)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_source, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading base model %s (4-bit quantized)", base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    if adapter_path:
        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    class ClarimolChemBenchWrapper:
        """
        Wrapper that adapts CLARIMOL's model loading to ChemBench's
        expected interface.

        ChemBench calls .generate(prompts, **kwargs) where prompts
        is a list of strings (already formatted questions).

        Returns Generations(generations=[[Generation(text=...)], ...])
        """

        def __init__(self, model, tokenizer):
            self._model = model
            self._tokenizer = tokenizer

        @torch.inference_mode()
        def generate(self, prompts: List[str], **kwargs) -> Generations:
            generations = []
            for prompt in prompts:
                # ChemBench may pass raw question text or chat-formatted text.
                # Wrap in chat template if not already formatted.
                if "<|" not in prompt and "[INST]" not in prompt:
                    messages = [{"role": "user", "content": prompt}]
                    try:
                        formatted = self._tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    except Exception:
                        formatted = prompt
                else:
                    formatted = prompt

                inputs = self._tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1536,
                )
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=(
                        self._tokenizer.pad_token_id
                        or self._tokenizer.eos_token_id
                    ),
                )

                response = self._tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                generations.append([Generation(text=response)])

            return Generations(generations=generations)

    wrapper = ClarimolChemBenchWrapper(model, tokenizer)
    return wrapper, model, tokenizer


def run_chembench(
    base_model_name: str,
    adapter_path: str | None,
    label: str,
    output_dir: str,
) -> dict:
    """Run ChemBench evaluation for a single model configuration."""
    from chembench.prompter import PrompterBuilder
    from chembench.evaluate import ChemBenchmark

    # Try to import save function (name varies across versions)
    save_fn = None
    for fn_name in ["save_topic_reports", "save_topic_results"]:
        try:
            save_fn = getattr(
                __import__("chembench.evaluate", fromlist=[fn_name]), fn_name
            )
            logger.info("Using %s for saving results", fn_name)
            break
        except AttributeError:
            continue

    results_file = os.path.join(output_dir, f"{label}_results.json")
    if os.path.exists(results_file):
        logger.info("SKIP: %s results already exist at %s", label, results_file)
        with open(results_file) as f:
            return json.load(f)

    logger.info("Building model wrapper for %s...", label)
    wrapper, model, tokenizer = build_model_wrapper(base_model_name, adapter_path)

    logger.info("Creating ChemBench prompter...")
    # TODO: Verify this call. PrompterBuilder.from_model_object may expect:
    #   - from_model_object(model=wrapper)          # keyword arg
    #   - from_model_object(wrapper)                # positional
    #   - from_model_object("custom", model=wrapper)  # with provider prefix
    prompter = PrompterBuilder.from_model_object(model=wrapper)

    logger.info("Loading ChemBench benchmark...")
    benchmark = ChemBenchmark.from_huggingface(verbose=True)

    logger.info("Running ChemBench evaluation for %s...", label)
    results = benchmark.bench(prompter)

    # Save topic-level reports
    model_results_dir = os.path.join(output_dir, label)
    os.makedirs(model_results_dir, exist_ok=True)

    if save_fn:
        try:
            save_fn(benchmark, results, model_results_dir)
            logger.info("Topic reports saved to %s", model_results_dir)
        except Exception as e:
            logger.warning("save_topic_reports/results failed: %s", e)

    # Extract summary metrics from results.
    # TODO: Verify the structure of ChemBench results object.
    # It may be a list of dicts, a DataFrame, or a custom object.
    # Adapt the extraction logic based on actual return type.
    summary = {}
    try:
        if isinstance(results, list):
            # Each result may have topic, score, etc.
            correct = sum(1 for r in results if r.get("correct", False))
            total = len(results)
            summary = {
                "overall_accuracy": correct / total if total > 0 else 0.0,
                "correct": correct,
                "total": total,
            }

            # Group by topic
            by_topic = {}
            for r in results:
                topic = r.get("topic", r.get("category", "unknown"))
                if topic not in by_topic:
                    by_topic[topic] = {"correct": 0, "total": 0}
                by_topic[topic]["total"] += 1
                if r.get("correct", False):
                    by_topic[topic]["correct"] += 1
            for t in by_topic:
                by_topic[t]["accuracy"] = (
                    by_topic[t]["correct"] / by_topic[t]["total"]
                    if by_topic[t]["total"] > 0
                    else 0.0
                )
            summary["by_topic"] = by_topic
        elif isinstance(results, dict):
            summary = results
        else:
            # Store raw string representation
            summary = {"raw_results": str(results)[:5000]}
    except Exception as e:
        logger.warning("Could not parse ChemBench results: %s", e)
        summary = {"error": str(e)}

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Results for %s saved to %s", label, results_file)

    # Free GPU memory
    del wrapper, model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return summary


def main():
    parser = argparse.ArgumentParser(description="ChemBench evaluation")
    parser.add_argument(
        "--output-dir",
        default="output/benchmarks/chembench",
        help="Directory for results",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Verify chembench is installed
    try:
        import chembench
        logger.info("ChemBench version: %s", getattr(chembench, "__version__", "unknown"))
    except ImportError:
        logger.error(
            "ChemBench not installed. Run: pip install chembench\n"
            "See: https://github.com/lamalab-org/chembench"
        )
        sys.exit(1)

    # Model configurations
    MODELS = [
        (
            "llama-8b-base",
            "meta-llama/Llama-3.1-8B-Instruct",
            None,
        ),
        (
            "llama-8b-clarimol",
            "meta-llama/Llama-3.1-8B-Instruct",
            "output/multi_seed/llama-8b/seed_42/final",
        ),
        (
            "mistral-7b-base",
            "mistralai/Mistral-7B-Instruct-v0.3",
            None,
        ),
        (
            "mistral-7b-clarimol",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "output/multi_seed/mistral-7b/seed_42/final",
        ),
    ]

    all_results = {}

    for label, base_model, adapter_path in MODELS:
        logger.info("=== %s ===", label)
        try:
            result = run_chembench(base_model, adapter_path, label, args.output_dir)
            all_results[label] = result
        except Exception as e:
            logger.error("FAILED for %s: %s", label, e, exc_info=True)
            all_results[label] = {"error": str(e)}

    # Print summary
    print("\n\nChemBench Results Summary")
    print(f"{'Model':30s}  {'Accuracy':>10s}  {'Total':>6s}")
    print("-" * 50)
    for label, res in all_results.items():
        if "error" in res:
            print(f"{label:30s}  {'ERROR':>10s}  {res['error'][:40]}")
        else:
            acc = res.get("overall_accuracy", 0.0)
            total = res.get("total", 0)
            print(f"{label:30s}  {acc:10.4f}  {total:6d}")

    # Save combined results
    combined_path = os.path.join(args.output_dir, "results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Combined results saved to %s", combined_path)


if __name__ == "__main__":
    main()
