"""
CLI Entrypoint
    Examples:
        python -m clarimol prepare
        python -m clarimol train
        python -m clarimol evaluate
"""

from __future__ import annotations
import argparse
import sys


def cmd_prepare(args: argparse.Namespace) -> None:
    """Prepare the CLARIMOL dataset from ZINC250K"""
    from clarimol.utils.io import setup_logging
    setup_logging(args.log_level)
    from clarimol.data.dataset import build_dataset, load_zinc250k, save_dataset
    from clarimol.data.pruning import PruningConfig
    import logging
    logger = logging.getLogger(__name__)
    # Load source molecules
    if args.source == "cod":
        from clarimol.data.cod import fetch_cod_smiles

        logger.info("Fetching SMILES from COD (max_entries=%s)", args.max_molecules or 1000)
        smiles = fetch_cod_smiles(
            max_entries=args.max_molecules or 1000,
            cache_dir=args.cod_cache_dir,
        )
    elif args.smiles_file:
        from clarimol.data.dataset import load_smiles_file
        logger.info("Loading SMILES from %s", args.smiles_file)
        smiles = load_smiles_file(args.smiles_file)
    else:
        logger.info("Loading ZINC250K (split=%s)", args.split)
        smiles = load_zinc250k(args.split)
    logger.info("Loaded %d molecules", len(smiles))
    # Resolve curriculum ordering: explicit --curriculum-order takes precedence
    if args.curriculum_order is not None:
        curriculum_order = args.curriculum_order
    elif args.no_curriculum:
        curriculum_order = "none"
    else:
        curriculum_order = "easy-hard"
    pruning = PruningConfig(
        keep_n=args.keep_n,
        subsample=args.subsample,
        trim_fraction=args.trim_fraction,
        curriculum_order=curriculum_order,
    )
    # Decontamination: load test SMILES and build exclusion set
    exclude_smiles = None
    if args.decontaminate:
        from clarimol.data.dataset import load_dataset_from_disk, _canonicalize_smiles
        logger.info("Loading test data from %s for decontamination", args.decontaminate)
        test_data = load_dataset_from_disk(args.decontaminate)
        exclude_smiles = set()
        for task_samples in test_data.values():
            for s in task_samples:
                can = _canonicalize_smiles(s.smiles)
                if can is not None:
                    exclude_smiles.add(can)
        logger.info("Decontamination: %d unique test molecules to exclude", len(exclude_smiles))
    task_names = args.tasks.split(",") if args.tasks else None
    data = build_dataset(
        smiles_list=smiles,
        task_names=task_names,
        pruning=pruning,
        seed=args.seed,
        max_molecules=args.max_molecules,
        exclude_smiles=exclude_smiles,
    )
    save_dataset(data, args.output_dir)
    total = sum(len(v) for v in data.values())
    logger.info("Done. %d total samples across %d tasks → %s", total, len(data), args.output_dir)


def cmd_train(args: argparse.Namespace) -> None:
    """Run the training pipeline"""
    from clarimol.utils.io import setup_logging
    setup_logging(args.log_level)
    from clarimol.train.config import TrainConfig
    from clarimol.train.trainer import run_training
    # bf16 overrides fp16 when both are requested
    use_bf16 = args.bf16
    use_fp16 = (not args.no_fp16) and (not use_bf16)
    config = TrainConfig(
        model_name=args.model,
        model_max_length=args.max_length,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bf16=use_bf16,
        fp16=use_fp16,
        load_in_4bit=not args.no_4bit,
        use_wandb=not args.no_wandb,
        seed=args.seed,
        gradient_checkpointing=not args.no_grad_ckpt,
        packing=args.packing,
        select_sample=args.select_sample,
        use_unsloth=not args.no_unsloth,
        resume_from_checkpoint=args.resume,
        save_steps=args.save_steps,
    )
    run_training(config)


def cmd_downstream_train(args: argparse.Namespace) -> None:
    """Fine-tune a model on a downstream molecular generation task"""
    from clarimol.utils.io import setup_logging
    setup_logging(args.log_level)
    from clarimol.train.downstream import DownstreamConfig, run_downstream_training
    use_bf16 = args.bf16
    use_fp16 = (not args.no_fp16) and (not use_bf16)
    config = DownstreamConfig(
        model_name=args.model,
        data_dir=args.data_dir,
        task=args.task,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_wandb=not args.no_wandb,
        seed=args.seed,
        gradient_checkpointing=not args.no_grad_ckpt,
        bf16=use_bf16,
        fp16=use_fp16,
        load_in_4bit=not args.no_4bit,
        max_samples=args.max_samples,
        use_unsloth=not args.no_unsloth,
    )
    run_downstream_training(config)


def cmd_downstream_eval(args: argparse.Namespace) -> None:
    """Evaluate a model on downstream molecular generation tasks"""
    from clarimol.utils.io import setup_logging
    setup_logging(args.log_level)
    import json
    import logging
    from clarimol.eval.downstream import evaluate_downstream
    logger = logging.getLogger(__name__)
    tasks = args.tasks.split(",") if args.tasks else None
    results = evaluate_downstream(
        model_path=args.model_path,
        data_dir=args.data_dir,
        tasks=tasks,
        use_unsloth=not args.no_unsloth,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
    print("\nDownstream Evaluation Results")
    for task_name, r in results.items():
        print(f"  {task_name}:")
        print(f"    exact={r.exact_match:.3f}  bleu={r.bleu:.3f}  lev={r.levenshtein:.2f}  lev_norm={r.levenshtein_norm:.3f}")
        print(f"    maccs={r.maccs_fps:.3f}  rdk={r.rdk_fps:.3f}  morgan={r.morgan_fps:.3f}")
        print(f"    validity={r.validity:.3f}")
    if args.output_file:
        out = {}
        for name, r in results.items():
            out[name] = {
                "exact_match": r.exact_match,
                "bleu": r.bleu,
                "levenshtein": r.levenshtein,
                "levenshtein_norm": r.levenshtein_norm,
                "maccs_fps": r.maccs_fps,
                "rdk_fps": r.rdk_fps,
                "morgan_fps": r.morgan_fps,
                "validity": r.validity,
            }
        with open(args.output_file, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Results saved to %s", args.output_file)


def cmd_check_contamination(args: argparse.Namespace) -> None:
    """Check for data contamination between train and test splits."""
    from clarimol.utils.io import setup_logging
    setup_logging(args.log_level)
    import json
    import logging
    from rdkit import Chem
    logger = logging.getLogger(__name__)

    def _load_smiles_from_json_dir(data_dir: str) -> set[str]:
        """Load and canonicalize all SMILES from a dataset directory."""
        from pathlib import Path
        canonical: set[str] = set()
        d = Path(data_dir)
        for path in d.glob("*.json"):
            with open(path) as f:
                records = json.load(f)
            for r in records:
                smi = r.get("smiles", "")
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    canonical.add(Chem.MolToSmiles(mol, canonical=True))
        return canonical

    logger.info("Loading training SMILES from %s", args.train_dir)
    train_smiles = _load_smiles_from_json_dir(args.train_dir)
    logger.info("Loading test SMILES from %s", args.test_dir)
    test_smiles = _load_smiles_from_json_dir(args.test_dir)

    overlap = train_smiles & test_smiles
    print(f"\nData Contamination Check")
    print(f"  Train molecules (unique canonical): {len(train_smiles)}")
    print(f"  Test molecules  (unique canonical): {len(test_smiles)}")
    print(f"  Overlap: {len(overlap)}")
    if test_smiles:
        rate = len(overlap) / len(test_smiles)
        print(f"  Contamination rate: {rate:.4f} ({rate*100:.2f}%)")
    if overlap:
        print(f"\n  First 10 overlapping SMILES:")
        for smi in sorted(overlap)[:10]:
            print(f"    {smi}")

    if args.output_file:
        result = {
            "train_unique": len(train_smiles),
            "test_unique": len(test_smiles),
            "overlap_count": len(overlap),
            "contamination_rate": len(overlap) / len(test_smiles) if test_smiles else 0.0,
            "overlap_smiles": sorted(overlap),
        }
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Results saved to %s", args.output_file)

    if overlap:
        logger.warning("DATA CONTAMINATION DETECTED: %d molecules appear in both train and test", len(overlap))


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained model on parsing tasks"""
    from clarimol.utils.io import setup_logging
    setup_logging(args.log_level)
    import json
    import logging
    from clarimol.eval.inference import evaluate_model
    logger = logging.getLogger(__name__)
    results = evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        use_unsloth=not args.no_unsloth,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
    # Print summary
    print("\nEvaluation Results")
    for task_name, r in results.items():
        print(f"  {task_name:20s}  acc={r.accuracy:.4f}  ({r.correct}/{r.total})")
        if r.validity > 0:
            print(f"  {'':20s}  validity={r.validity:.4f}")
        if r.extraction_failures > 0:
            print(f"  {'':20s}  extraction_failures={r.extraction_failures}")
        if r.breakdown and len(r.breakdown) > 1:
            print(f"  {'':20s}  breakdown:")
            for cat, stats in sorted(r.breakdown.items(), key=lambda x: -x[1]["accuracy"]):
                print(f"  {'':22s}{cat:25s}  {stats['accuracy']:.4f}  ({stats['correct']}/{stats['total']})")
    # Optionally save to JSON
    if args.output_file:
        out = {
            name: {
                "accuracy": r.accuracy,
                "correct": r.correct,
                "total": r.total,
                "validity": r.validity,
                "extraction_failures": r.extraction_failures,
                "breakdown": r.breakdown,
            }
            for name, r in results.items()
        }
        with open(args.output_file, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Results saved to %s", args.output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clarimol",
        description="CLARIMOL — SMILES parsing for molecular LLM pre-training",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    # Prepare
    p_prep = sub.add_parser("prepare", help="Build CLARIMOL dataset")
    p_prep.add_argument("--source", default="zinc", choices=["zinc", "cod", "file"],
                        help="Molecule source: zinc (ZINC250K), cod (COD API), file (custom)")
    p_prep.add_argument("--smiles-file", type=str, default=None, help="Path to SMILES file (--source file)")
    p_prep.add_argument("--split", default="train", help="ZINC250K split (default: train)")
    p_prep.add_argument("--cod-cache-dir", default="data/cod_cache", help="Cache dir for COD CIF files")
    p_prep.add_argument("--tasks", default=None, help="Comma-separated task names (default: all)")
    p_prep.add_argument("--output-dir", default="data/clarimol", help="Output directory")
    p_prep.add_argument("--keep-n", type=int, default=50_000, help="Samples to keep per task")
    p_prep.add_argument("--subsample", default="middle", choices=["middle", "top", "bottom", "random"])
    p_prep.add_argument("--trim-fraction", type=float, default=0.15, help="Tail trim fraction")
    p_prep.add_argument("--no-curriculum", action="store_true", help="Disable curriculum ordering (legacy; prefer --curriculum-order)")
    p_prep.add_argument("--curriculum-order", default=None,
                        choices=["easy-hard", "hard-easy", "random", "none"],
                        help="Curriculum ordering strategy (default: easy-hard)")
    p_prep.add_argument("--seed", type=int, default=42)
    p_prep.add_argument("--max-molecules", type=int, default=None, help="Limit molecules (for dev)")
    p_prep.add_argument("--decontaminate", type=str, default=None,
                        help="Path to test data dir; removes overlapping molecules from training set")
    # Train
    p_train = sub.add_parser("train", help="Train on CLARIMOL dataset")
    p_train.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    p_train.add_argument("--data-dir", default="data/clarimol")
    p_train.add_argument("--output-dir", default="output/clarimol")
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--grad-accum", type=int, default=2)
    p_train.add_argument("--lr", type=float, default=5e-4)
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--lora-r", type=int, default=64)
    p_train.add_argument("--lora-alpha", type=int, default=16)
    p_train.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization (full precision)")
    p_train.add_argument("--no-wandb", action="store_true")
    p_train.add_argument("--no-grad-ckpt", action="store_true")
    p_train.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    p_train.add_argument("--no-fp16", action="store_true", help="Disable fp16 AMP (use fp32)")
    p_train.add_argument("--bf16", action="store_true", help="Use bf16 (requires Ampere+; overrides fp16)")
    p_train.add_argument("--no-unsloth", action="store_true")
    p_train.add_argument("--packing", action="store_true", help="Pack sequences to reduce padding waste")
    p_train.add_argument("--select-sample", type=int, default=None, help="Max samples per task (default: all)")
    p_train.add_argument("--resume", action="store_true", help="Resume from last checkpoint in output-dir")
    p_train.add_argument("--save-steps", type=int, default=2000, help="Save checkpoint every N steps")
    p_train.add_argument("--seed", type=int, default=42)
    # Evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate a trained model")
    p_eval.add_argument("--model-path", required=True, help="Path to trained model")
    p_eval.add_argument("--data-dir", default="data/clarimol")
    p_eval.add_argument("--output-file", default=None, help="JSON results output")
    p_eval.add_argument("--batch-size", type=int, default=8)
    p_eval.add_argument("--max-samples", type=int, default=None, help="Cap samples per task")
    p_eval.add_argument("--no-unsloth", action="store_true")
    # Downstream train
    p_ds_train = sub.add_parser("downstream-train", help="Fine-tune on a downstream task")
    p_ds_train.add_argument("--model", required=True, help="Pre-trained model or base model path")
    p_ds_train.add_argument("--data-dir", default="data/mol_instructions")
    p_ds_train.add_argument("--task", required=True,
                            choices=["retrosynthesis", "reagent_prediction", "forward_reaction_prediction"])
    p_ds_train.add_argument("--output-dir", default="output/downstream")
    p_ds_train.add_argument("--batch-size", type=int, default=8)
    p_ds_train.add_argument("--grad-accum", type=int, default=2)
    p_ds_train.add_argument("--lr", type=float, default=5e-4)
    p_ds_train.add_argument("--epochs", type=int, default=1)
    p_ds_train.add_argument("--lora-r", type=int, default=64)
    p_ds_train.add_argument("--lora-alpha", type=int, default=16)
    p_ds_train.add_argument("--max-samples", type=int, default=None, help="Cap training samples")
    p_ds_train.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization (full precision)")
    p_ds_train.add_argument("--no-wandb", action="store_true")
    p_ds_train.add_argument("--no-grad-ckpt", action="store_true")
    p_ds_train.add_argument("--no-fp16", action="store_true", help="Disable fp16 AMP (use fp32)")
    p_ds_train.add_argument("--bf16", action="store_true", help="Use bf16 (requires Ampere+; overrides fp16)")
    p_ds_train.add_argument("--no-unsloth", action="store_true")
    p_ds_train.add_argument("--seed", type=int, default=42)
    # Downstream evaluate
    p_ds_eval = sub.add_parser("downstream-eval", help="Evaluate on downstream generation tasks")
    p_ds_eval.add_argument("--model-path", required=True, help="Path to fine-tuned model")
    p_ds_eval.add_argument("--data-dir", default="data/mol_instructions")
    p_ds_eval.add_argument("--tasks", default=None, help="Comma-separated task names (default: all)")
    p_ds_eval.add_argument("--output-file", default=None, help="JSON results output")
    p_ds_eval.add_argument("--batch-size", type=int, default=4)
    p_ds_eval.add_argument("--max-samples", type=int, default=None)
    p_ds_eval.add_argument("--no-unsloth", action="store_true")
    # Check contamination
    p_contam = sub.add_parser("check-contamination", help="Check for train/test data contamination")
    p_contam.add_argument("--train-dir", default="data/clarimol", help="Training data directory")
    p_contam.add_argument("--test-dir", default="data/test", help="Test data directory")
    p_contam.add_argument("--output-file", default=None, help="JSON output for contamination report")

    args = parser.parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "downstream-train":
        cmd_downstream_train(args)
    elif args.command == "downstream-eval":
        cmd_downstream_eval(args)
    elif args.command == "check-contamination":
        cmd_check_contamination(args)


if __name__ == "__main__":
    main()