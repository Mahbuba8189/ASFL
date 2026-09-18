# asfl/main.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
from typing import Dict, Any, List, Optional
import pandas as pd

from config import ExperimentConfig
from utils import set_seed, print_banner, print_device_info, save_json, ensure_dir
from data import (
    get_wesad_loaders,
    get_bidmc_loaders,
    get_mhealth_loaders,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the main entry point.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="ASFL - Split Federated Learning ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single experiment
  python main.py --dataset wesad --method asfl --distribution iid
  
  # Run all main experiments
  python main.py --run_all
  
  # Run ablation study (3 experiments on WESAD with ASFL, skipping full)
  python main.py --run_ablation
  
  # Run ablation on specific dataset
  python main.py --run_ablation --run_ablation_dataset bidmc
  
  # Custom settings
  python main.py --dataset bidmc --method ressfl --distribution noniid --epochs 50 --batch_size 32
        """
    )

    # Experiment Selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wesad", "bidmc", "mhealth"],
        help="Dataset to run"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["plain", "ressfl", "asfl"],
        help="Method to run"
    )
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["iid", "noniid"],
        help="Data distribution"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Dirichlet alpha for non-IID partitioning (e.g., 0.1, 0.5, 1.0)"
    )

    # Run Modes
    parser.add_argument(
        "--run_all",
        action="store_true",
        default=False,
        help="Run all  main experiments"
    )
    parser.add_argument(
        "--run_iid",
        action="store_true",
        default=False,
        help="Run all IID experiments"
    )
    parser.add_argument(
        "--run_noniid",
        action="store_true",
        default=False,
        help="Run all Non-IID experiments"
    )
    parser.add_argument(
        "--run_ablation",
        action="store_true",
        default=False,
        help="Run ablation study (3 experiments on specified dataset with ASFL)"
    )
    parser.add_argument(
        "--run_ablation_dataset",
        type=str,
        choices=["wesad", "bidmc", "mhealth"],
        default="wesad",
        help="Dataset for ablation study (default: wesad)"
    )
    
    # Training Settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay"
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        default=False,
        help="Enable class-weighted CrossEntropyLoss"
    )
    parser.add_argument(
        "--oversample_target",
        type=int,
        default=None,
        help="Target count for KMeans-SMOTE oversampling (BIDMC Class 0). Example: 800"
    )
    
    # Window Size Setting
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Window size in seconds for data segmentation"
    )
    
    # ASFL-specific Settings (only passed when explicitly set)
    parser.add_argument(
        "--lambda_1",
        type=float,
        default=None,
        help="Reconstruction loss weight (ASFL only)"
    )
    parser.add_argument(
        "--lambda_2",
        type=float,
        default=None,
        help="Adversarial loss weight (ASFL only)"
    )
    parser.add_argument(
        "--lambda_3",
        type=float,
        default=None,
        help="Privacy penalty weight (ASFL only)"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
        help="Warm-up epochs (ASFL only)"
    )
    
    # PGD Attack Settings
    parser.add_argument(
        "--pgd_epsilon",
        type=float,
        default=None,
        help="PGD attack epsilon"
    )
    parser.add_argument(
        "--pgd_steps",
        type=int,
        default=None,
        help="PGD attack steps"
    )
    parser.add_argument(
        "--pgd_eval_steps",
        type=int,
        default=None,
        help="PGD attack steps for evaluation"
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=None,
        help="PGD attack step size"
    )
    
    # ResSFL-specific Settings
    parser.add_argument(
        "--ressfl_pretrain_epochs",
        type=int,
        default=None,
        help="ResSFL expert pre-training epochs"
    )
    parser.add_argument(
        "--ressfl_warmup_epochs",
        type=int,
        default=None,
        help="ResSFL warm-up epochs"
    )
    parser.add_argument(
        "--ressfl_lambda_rec",
        type=float,
        default=None,
        help="ResSFL reconstruction loss weight"
    )
    parser.add_argument(
        "--ressfl_teacher_lambda",
        type=float,
        default=None,
        help="ResSFL teacher lambda"
    )
    parser.add_argument(
        "--ressfl_finetune_lr",
        type=float,
        default=None,
        help="ResSFL fine-tuning learning rate"
    )
    parser.add_argument(
        "--ressfl_update_frequency",
        type=int,
        default=None,
        help="ResSFL decoder update frequency"
    )
    parser.add_argument(
        "--ressfl_use_pretrain",
        action="store_true",
        default=False,
        help="Enable ResSFL expert pre-training"
    )
    parser.add_argument(
        "--ressfl_source_task",
        type=str,
        choices=["same", "wesad", "bidmc", "mhealth"],
        default="same",
        help="ResSFL source task for pre-training (default: same)"
    )
    
    # Ablation Study Settings
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["full", "no_dual", "no_privacy", "no_evasion"],
        default="full",
        help="Ablation configuration (only used with --run_ablation)"
    )
    
    # Model Architecture (CONFIGURABLE)
    parser.add_argument(
        "--cut_layer",
        type=int,
        default=3,
        help="Cut layer for split (default: 3)"
    )
    parser.add_argument(
        "--stem_channels",
        type=int,
        default=32,
        help="Stem channels for backbone (default: 32)"
    )
    parser.add_argument(
        "--width_mult",
        type=float,
        default=1.0,
        help="Width multiplier for backbone (default: 1.0)"
    )
    parser.add_argument(
        "--server_hidden_dim",
        type=int,
        default=64,
        help="Server classifier hidden dim (default: 64)"
    )
    parser.add_argument(
        "--utility_hidden_dim",
        type=int,
        default=128,
        help="Utility projection hidden dim (default: 128)"
    )
    parser.add_argument(
        "--privacy_hidden_dim",
        type=int,
        default=64,
        help="Privacy projection hidden dim (default: 64)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)"
    )    
    # System Settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Base output directory (default: ./results)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="Random seed (default: 99)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Disable verbose output"
    )
    
    return parser.parse_args()


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build overrides dict only including values explicitly provided by the user.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Dict of override key-value pairs
    """
    overrides = {}
    
    # Training Settings
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
        overrides["main_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.use_class_weights:
        overrides["use_class_weights"] = True
    if args.oversample_target is not None:
        overrides["oversample_target"] = args.oversample_target
    if args.window_size is not None:
        overrides["window_size"] = args.window_size
    if args.alpha is not None:
        overrides["alpha"] = args.alpha
    
    # ASFL-specific Settings
    if args.lambda_1 is not None:
        overrides["lambda_1"] = args.lambda_1
    if args.lambda_2 is not None:
        overrides["lambda_2"] = args.lambda_2
    if args.lambda_3 is not None:
        overrides["lambda_3"] = args.lambda_3
    if args.warmup_epochs is not None:
        overrides["warmup_epochs"] = args.warmup_epochs
    
    # PGD Attack Settings
    if args.pgd_epsilon is not None:
        overrides["pgd_epsilon"] = args.pgd_epsilon
    if args.pgd_steps is not None:
        overrides["pgd_steps"] = args.pgd_steps
    if args.pgd_eval_steps is not None:
        overrides["pgd_eval_steps"] = args.pgd_eval_steps
    if args.pgd_alpha is not None:
        overrides["pgd_alpha"] = args.pgd_alpha
    
    # ResSFL-specific Settings
    if args.ressfl_pretrain_epochs is not None:
        overrides["ressfl_pretrain_epochs"] = args.ressfl_pretrain_epochs
    if args.ressfl_warmup_epochs is not None:
        overrides["ressfl_warmup_epochs"] = args.ressfl_warmup_epochs
    if args.ressfl_lambda_rec is not None:
        overrides["ressfl_lambda_rec"] = args.ressfl_lambda_rec
    if args.ressfl_teacher_lambda is not None:
        overrides["ressfl_teacher_lambda"] = args.ressfl_teacher_lambda
    if args.ressfl_finetune_lr is not None:
        overrides["ressfl_finetune_lr"] = args.ressfl_finetune_lr
    if args.ressfl_update_frequency is not None:
        overrides["ressfl_update_frequency"] = args.ressfl_update_frequency
    if args.ressfl_use_pretrain:
        overrides["ressfl_use_pretrain"] = args.ressfl_use_pretrain
    if args.ressfl_source_task != "same":
        overrides["ressfl_source_task"] = args.ressfl_source_task

    if args.cut_layer != 3:
        overrides["cut_layer"] = args.cut_layer
    if args.stem_channels != 32:
        overrides["stem_channels"] = args.stem_channels
    if args.width_mult != 1.0:
        overrides["width_mult"] = args.width_mult
    if args.server_hidden_dim != 64:
        overrides["server_hidden_dim"] = args.server_hidden_dim
    if args.utility_hidden_dim != 128:
        overrides["utility_hidden_dim"] = args.utility_hidden_dim
    if args.privacy_hidden_dim != 64:
        overrides["privacy_hidden_dim"] = args.privacy_hidden_dim
    if args.dropout != 0.2:
        overrides["dropout"] = args.dropout
 
    return overrides

def save_ablation_history_to_csv(trainer, exp_path, ablation_name):
    """
    Save training history from trainer's csv_log to CSV file with timestamp.
    
    Note: trainer.csv_log already contains all epoch data from training.
    This function just writes it to a CSV file.
    """
    import time
    
    # Use the csv_log that ASFL already maintains during training
    if hasattr(trainer, 'csv_log') and trainer.csv_log:
        # Add timestamp to filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(exp_path, f"training_history_{ablation_name}_{timestamp}.csv")
        
        # Write the csv_log to file
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header matching the csv_log format
            writer.writerow([
                'epoch', 'train_loss', 'train_acc', 'train_precision', 'train_recall', 
                'train_f1', 'train_auprc', 'train_rec_loss', 'train_adv_loss', 
                'train_privacy_penalty', 'train_evasion_f1',
                'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 
                'val_auprc', 'val_rec_loss', 'val_adv_loss', 'val_privacy_penalty',
                'val_evasion_f1', 'val_mse_l0', 'val_mse_best'
            ])
            # Write all rows
            writer.writerows(trainer.csv_log)
        
        print(f"  Training history ({len(trainer.csv_log)} epochs) saved to: {csv_path}")
        return True

    elif hasattr(trainer, 'history') and trainer.history:
        # Add timestamp to filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(exp_path, f"training_history_{ablation_name}_{timestamp}.csv")
        history_dict = trainer.history
        n_epochs = len(history_dict.get('train_loss', []))
        
        if n_epochs > 0:
            # Create a DataFrame with all available metrics
            df_data = {
                'epoch': list(range(1, n_epochs + 1)),
                'train_loss': history_dict.get('train_loss', [0.0] * n_epochs),
                'train_acc': history_dict.get('train_acc', [0.0] * n_epochs),
                'train_f1': history_dict.get('train_f1', [0.0] * n_epochs),
                'val_loss': history_dict.get('val_loss', [0.0] * n_epochs),
                'val_acc': history_dict.get('val_acc', [0.0] * n_epochs),
                'val_f1': history_dict.get('val_f1', [0.0] * n_epochs),
                'val_mse_best': history_dict.get('val_mse_best', [0.0] * n_epochs),
            }
            df = pd.DataFrame(df_data)
            df.to_csv(csv_path, index=False)
            print(f"  Training history ({n_epochs} epochs) saved to: {csv_path}")
            return True
    
    print(f"  No training history found to save for {ablation_name}")
    return False


def run_single_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run a single experiment with the given arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Dict[str, Any]: Experiment results
    """
    from experiments import ExperimentRunner
    
    runner = ExperimentRunner(
        base_output_dir=args.output_dir,
        device=args.device,
        verbose=not args.quiet,
        seed=args.seed
    )
    
    overrides = _build_overrides(args)
    
    result = runner.run_single_experiment(
        dataset=args.dataset,
        method=args.method,
        distribution=args.distribution,
        **overrides
    )
    
    return result


def run_single_ablation_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run a single ablation experiment.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Dict[str, Any]: Ablation experiment results
    """
    from trainer_asfl import create_asfl_trainer
    
    exp_path = os.path.join(args.output_dir, "ablation", args.ablation)
    ensure_dir(exp_path)
    
    # Build config with proper defaults
    cfg = ExperimentConfig(
        dataset=args.run_ablation_dataset,
        method="asfl",
        partition_mode="dirichlet" if args.alpha is not None else "iid",
        seed=args.seed,
        device=args.device,
        output_dir=exp_path,
    
        epochs=args.epochs if args.epochs is not None else 50,
        main_epochs=args.epochs if args.epochs is not None else 50,
        batch_size=args.batch_size if args.batch_size is not None else 32,
        learning_rate=args.learning_rate if args.learning_rate is not None else 1e-3,
        weight_decay=args.weight_decay if args.weight_decay is not None else 1e-5,
    
        window_size=args.window_size if args.window_size is not None else 8,
        cut_layer=args.cut_layer,
        alpha=args.alpha,
    
        use_class_weights=args.use_class_weights,
        lambda_1=args.lambda_1 if args.lambda_1 is not None else 1.0,
        lambda_2=args.lambda_2 if args.lambda_2 is not None else 1.0,
        lambda_3=args.lambda_3 if args.lambda_3 is not None else 0.01,
        warmup_epochs=args.warmup_epochs if args.warmup_epochs is not None else 10,
    
        pgd_epsilon=args.pgd_epsilon if args.pgd_epsilon is not None else 0.3,
        pgd_steps=args.pgd_steps if args.pgd_steps is not None else 40,
        pgd_eval_steps=args.pgd_eval_steps if args.pgd_eval_steps is not None else 40,
        pgd_alpha=args.pgd_alpha if args.pgd_alpha is not None else 0.01,
    )
    
    from experiments import ExperimentRunner
    
    loader_map = {
        "wesad": get_wesad_loaders,
        "bidmc": get_bidmc_loaders,
        "mhealth": get_mhealth_loaders,
    }

    # Use the same data paths as experiments.py (single source of truth)
    data_dir_map = {
        "wesad": ExperimentRunner.DATASET_PATHS["wesad"]["data_dir"],
        "bidmc": ExperimentRunner.DATASET_PATHS["bidmc"]["data_dir"],
        "mhealth": ExperimentRunner.DATASET_PATHS["mhealth"]["data_dir"],
    }

    loader_fn = loader_map[args.run_ablation_dataset]
    data_dir = data_dir_map[args.run_ablation_dataset]

    # Determine partition mode from args
    partition_mode = "dirichlet" if args.alpha is not None else "iid"
    non_iid_alpha = args.alpha if args.alpha is not None else None

    train_loaders, val_loaders = loader_fn(
        data_dir=data_dir,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        partition_mode=partition_mode,
        non_iid_alpha=non_iid_alpha,
        window_size=cfg.window_size,
        normalize=cfg.get_dataset_normalization(),
        val_split=cfg.val_split,
        seed=cfg.seed,
        oversample_target=args.oversample_target if hasattr(args, 'oversample_target') else None,
    )

    
    # Build trainer with ablation
    trainer = create_asfl_trainer(
        cfg, train_loaders, val_loaders, ablation=args.ablation
    )
    
    # Train
    try:
        history = trainer.fit()
        
        # Save training history using the csv_log from trainer
        #save_ablation_history_to_csv(trainer, exp_path, args.ablation)
            
    except Exception as e:
        print(f"Error during ablation training: {e}")
        import traceback
        traceback.print_exc()
        return {
            "ablation": args.ablation,
            "dataset": args.run_ablation_dataset,
            "status": "failed",
            "error": str(e),
        }
    
    # Evaluate inversion attack
    inv_results = trainer.evaluate_inversion_attack()
    
    result = {
        "ablation": args.ablation,
        "dataset": args.run_ablation_dataset,
        "status": "success",
        "best_val_acc": trainer.best_val_acc,
        "best_privacy_mse": trainer.best_privacy_mse,
        "best_mse": inv_results.get("best_mse", 0.0),
        "mse_gl0": inv_results.get("mse_gl0", 0.0),
        "mse_gl1": inv_results.get("mse_gl1", 0.0),
        "mse_gl2": inv_results.get("mse_gl2", 0.0),
        "mse_gl3": inv_results.get("mse_gl3", 0.0),
        "mse_adaptive": inv_results.get("mse_adaptive", 0.0),
        "best_attacker": inv_results.get("best_attacker", "None"),
    }
    
    return result


def run_ablation_study(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Run the ablation study for ASFL.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        List[Dict[str, Any]]: Ablation results
    """
    print_banner("ASFL Ablation Study")
    print(f"Dataset: {args.run_ablation_dataset.upper()}")
    # Single-ablation mode: run ONLY the requested ablation
    if args.ablation != "full":
        print(f"Running single ablation: {args.ablation}")
        print("=" * 60)
        return [run_single_ablation_experiment(args)]

    print(f"Ablations: no_dual, no_privacy, no_evasion")
    print("=" * 60)
    
    results = []
    ablations = ["no_dual", "no_privacy", "no_evasion"]
    
    for ablation in ablations:
        print(f"\n--- Running ablation: {ablation} ---")
        
        # Create a copy of args to avoid mutation
        ablation_args = argparse.Namespace(**vars(args))
        ablation_args.ablation = ablation
        
        result = run_single_ablation_experiment(ablation_args)
        results.append(result)
        
        if result.get("status") == "success":
            print(f"  Best Val Acc: {result['best_val_acc']:.2f}%")
            print(f"  Best Privacy MSE: {result['best_privacy_mse']:.4f}")
        else:
            print(f"  Failed: {result.get('error', 'Unknown error')}")
    
    # Save ablation results
    ablation_json_path = os.path.join(args.output_dir, "ablation", "ablation_results.json")
    save_json(results, ablation_json_path)
    
    # Also save as CSV
    ablation_csv_path = os.path.join(args.output_dir, "ablation", "ablation_results.csv")
    try:
        df = pd.DataFrame(results)
        df.to_csv(ablation_csv_path, index=False)
        print(f"Results also saved to CSV: {ablation_csv_path}")
    except Exception as e:
        print(f"Warning: Could not save CSV: {e}")
    
    print("\n" + "=" * 60)
    print("Ablation Study Complete!")
    print(f"Results saved to: {ablation_json_path}")
    print("=" * 60)
    
    return results


def main() -> None:
    """Main entry point for ASFL framework."""
    args = parse_args()
    
    verbose = not args.quiet
    
    # Print banner
    print_banner("ASFL Framework")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Print device info if verbose
    if verbose:
        print_device_info()
    
    overrides = _build_overrides(args)
    
    # Dispatch to appropriate runner
    if args.run_ablation:
        run_ablation_study(args)
    elif args.run_all:
        from experiments import ExperimentRunner
        runner = ExperimentRunner(
            base_output_dir=args.output_dir,
            device=args.device,
            verbose=verbose,
            seed=args.seed
        )
        runner.run_all_experiments(**overrides)
    elif args.run_iid:
        from experiments import ExperimentRunner
        runner = ExperimentRunner(
            base_output_dir=args.output_dir,
            device=args.device,
            verbose=verbose,
            seed=args.seed
        )
        runner.run_iid_suite(**overrides)
    elif args.run_noniid:
        from experiments import ExperimentRunner
        runner = ExperimentRunner(
            base_output_dir=args.output_dir,
            device=args.device,
            verbose=verbose,
            seed=args.seed
        )
        runner.run_noniid_suite(**overrides)
    elif args.dataset and args.method and args.distribution:
        run_single_experiment(args)
    else:
        print("Error: Please specify --dataset, --method, --distribution, or use --run_all / --run_ablation")
        print("\nExamples:")
        print("  python main.py --dataset wesad --method asfl --distribution iid")
        print("  python main.py --run_all")
        print("  python main.py --run_ablation")
        sys.exit(1)
    
    print_banner("ASFL Complete")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()