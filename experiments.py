# asfl/experiments.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import shutil
import csv
import argparse
import inspect
import traceback
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader

from config import ExperimentConfig, save_config
from data import (
    get_wesad_loaders,
    get_bidmc_loaders,
    get_mhealth_loaders,
)

# Trainer Registry

TRAINER_REGISTRY = {}

def register_trainer(name):
    """Decorator to register a trainer class."""
    def decorator(cls):
        TRAINER_REGISTRY[name] = cls
        return cls
    return decorator


def _get_trainer_factory(method: str):
    """
    Lazy import trainer factory function for a given method.
    Returns None if the trainer module is not available.
    """
    if method == "plain":
        try:
            from trainer_plain import create_plain_trainer
            return create_plain_trainer
        except ImportError:
            return None
    elif method == "ressfl":
        try:
            from trainer_ressfl import create_ressfl_trainer
            return create_ressfl_trainer
        except ImportError:
            return None
    elif method == "asfl":
        try:
            from trainer_asfl import create_asfl_trainer
            return create_asfl_trainer
        except ImportError:
            return None
    else:
        return None


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_experiment_id(dataset: str, method: str, distribution: str, seed: int) -> str:
    """Generate a unique experiment ID including seed."""
    return f"{dataset}_{method}_{distribution}_seed{seed}"


def convert_to_serializable(obj):
    """
    Convert numpy and torch types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, torch.nn.Module):
        return str(type(obj).__name__)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    return obj


class ExperimentRunner:
    """
    Experiment orchestration for ASFL.
    Runs all experiments:
    - 3 datasets (WESAD, BIDMC, MHEALTH)
    - 3 methods (Plain, ResSFL, ASFL)
    - 2 distributions (IID, Non-IID)
    Outputs per experiment:
    - checkpoint_best.pth
    - checkpoint_last.pth
    - metrics.json (optional, from trainers)
    - evaluation_results.json (from this script)
    - train_log.csv (optional)
    - config.json
    """

    DATASETS = ["wesad", "bidmc", "mhealth"]
    METHODS = ["plain", "ressfl", "asfl"]
    DISTRIBUTIONS = ["iid", "noniid"]

    DATASET_PATHS = {
        "wesad": {
            "data_dir": "./WESAD/WESAD",
            "num_clients": 15,
        },
        "bidmc": {
            "data_dir": "./BIDMC/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv",
            "num_clients": 53,
        },
        "mhealth": {
            "data_dir": "./MHEALTHDATASET",
            "num_clients": 10,
        }
    }

    DATASET_LOADER_FN = {
        "wesad": get_wesad_loaders,
        "bidmc": get_bidmc_loaders,
        "mhealth": get_mhealth_loaders,
    }

    CHECKPOINT_BEST = "checkpoint_best.pth"
    CHECKPOINT_LAST = "checkpoint_last.pth"
    EVAL_RESULTS_FILE = "evaluation_results.json"
    TRAIN_LOG_FILE = "train_log.csv"
    CONFIG_FILE = "config.json"

    def __init__(self, base_output_dir: str = "./results", device: str = "cuda",
                 verbose: bool = True, seed: int = 42):
        """
        Args:
            base_output_dir: Root directory for all experiment outputs
            device: Device to use ('cuda' or 'cpu')
            verbose: Print progress information
            seed: Random seed for reproducibility
        """
        self.base_output_dir = base_output_dir
        self.device = device if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        self.seed = seed
        set_seed(seed)
        self.results = []
        self._available_trainers = self._discover_trainers()

    def _discover_trainers(self) -> Dict[str, Any]:
        """Discover available trainers using lazy imports."""
        trainers = {}
        for method in self.METHODS:
            factory = _get_trainer_factory(method)
            if factory is not None:
                trainers[method] = factory
                if self.verbose:
                    print(f"Discovered trainer: {method}")
            else:
                if self.verbose:
                    print(f"Trainer '{method}' not available")
        return trainers

    def get_experiment_id(self, dataset: str, method: str, distribution: str) -> str:
        """Get unique experiment ID including seed."""
        return generate_experiment_id(dataset, method, distribution, self.seed)

    def get_experiment_path(self, dataset: str, method: str, distribution: str) -> str:
        """Get experiment output directory path."""
        exp_id = self.get_experiment_id(dataset, method, distribution)
        return os.path.join(self.base_output_dir, exp_id)

    def get_experiment_config_path(self, dataset: str, method: str, distribution: str) -> str:
        """Get config file path for a single experiment."""
        return os.path.join(self.get_experiment_path(dataset, method, distribution), self.CONFIG_FILE)

    def get_experiment_metrics_path(self, dataset: str, method: str, distribution: str) -> str:
        """Get metrics file path for a single experiment."""
        return os.path.join(self.get_experiment_path(dataset, method, distribution), self.METRICS_FILE)

    def get_experiment_eval_results_path(self, dataset: str, method: str, distribution: str) -> str:
        """Get evaluation results file path for a single experiment."""
        return os.path.join(self.get_experiment_path(dataset, method, distribution), self.EVAL_RESULTS_FILE)

    def _standardize_checkpoints(self, exp_path: str) -> None:
        """Standardize checkpoint filenames to thesis-required format."""
        if not os.path.exists(exp_path):
            return

        checkpoint_map = {
            "checkpoint_best_acc.pth": self.CHECKPOINT_BEST,
            "checkpoint_best.pth": self.CHECKPOINT_BEST,
            "checkpoint_best_privacy.pth": self.CHECKPOINT_BEST,
            "checkpoint_last.pth": self.CHECKPOINT_LAST,
            "checkpoint_asfl_best_acc.pth": self.CHECKPOINT_BEST,
            "checkpoint_asfl_best_privacy.pth": self.CHECKPOINT_BEST,
            "checkpoint_asfl_last.pth": self.CHECKPOINT_LAST,
            "checkpoint_asfl_final.pth": self.CHECKPOINT_LAST,
            "checkpoint_ressfl_best_acc.pth": self.CHECKPOINT_BEST,
            "checkpoint_ressfl_best_privacy.pth": self.CHECKPOINT_BEST,
            "checkpoint_ressfl_last.pth": self.CHECKPOINT_LAST,
            "checkpoint_ressfl_final.pth": self.CHECKPOINT_LAST,
            "checkpoint_plain_best.pth": self.CHECKPOINT_BEST,
            "checkpoint_plain_last.pth": self.CHECKPOINT_LAST,
            "checkpoint_plain_final.pth": self.CHECKPOINT_LAST,
        }

        for old_name, new_name in checkpoint_map.items():
            old_path = os.path.join(exp_path, old_name)
            new_path = os.path.join(exp_path, new_name)
            if os.path.exists(old_path) and not os.path.exists(new_path):
                try:
                    shutil.copy2(old_path, new_path)
                except Exception:
                    pass

    def _verify_experiment_outputs(self, exp_path: str) -> Dict[str, bool]:
        """
        Verify that required and optional output files exist.
        
        Returns:
            Dictionary with 'required_ok' and individual file status
        """
        required_files = [
            self.CHECKPOINT_BEST,
            self.CHECKPOINT_LAST,
            self.CONFIG_FILE,
        ]
        optional_files = [
            self.EVAL_RESULTS_FILE,
            self.TRAIN_LOG_FILE,
        ]
        
        status = {}
        
        # Check required files
        required_status = {}
        for filename in required_files:
            filepath = os.path.join(exp_path, filename)
            required_status[filename] = os.path.exists(filepath)
            status[filename] = required_status[filename]
        
        # Check optional files
        for filename in optional_files:
            filepath = os.path.join(exp_path, filename)
            status[filename] = os.path.exists(filepath)
        
        # Required outputs are only the mandatory files + reconstructions directory
        status["required_ok"] = all(required_status.values())
        
        return status

    def build_config(self, dataset: str, method: str, distribution: str,
                     **kwargs) -> ExperimentConfig:
        """
        Build ExperimentConfig for a single experiment.
        
        Order of precedence:
        1. kwargs overrides (from CLI or caller) - HIGHEST PRIORITY
        2. Method-specific defaults
        3. General config defaults
        """
        partition_mode = "iid" if distribution == "iid" else "dirichlet"
        # Filter out oversample_target from kwargs (it goes to data loader, not ExperimentConfig)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "oversample_target" and not (distribution == "iid" and k == "alpha")}
        self._oversample_target = kwargs.get("oversample_target", None)
        dataset_config = self.DATASET_PATHS.get(dataset, {})
        
        # Start with base config
        config_kwargs = {
            "dataset": dataset,
            "method": method,
            "partition_mode": partition_mode,
            "num_clients": dataset_config.get("num_clients", 10),
            "seed": self.seed,
            "device": self.device,
            "output_dir": self.get_experiment_path(dataset, method, distribution),
        }

        # Apply general kwargs (CLI overrides from main.py's overrides dict)
        config_kwargs.update(filtered_kwargs)

        # Apply method-specific defaults ONLY for keys NOT already provided in kwargs
        if method == "plain":
            if "lambda_1" not in filtered_kwargs:
                config_kwargs["lambda_1"] = 0.0
            if "lambda_2" not in filtered_kwargs:
                config_kwargs["lambda_2"] = 0.0
            if "lambda_3" not in filtered_kwargs:
                config_kwargs["lambda_3"] = 0.0
            if "warmup_epochs" not in filtered_kwargs:
                config_kwargs["warmup_epochs"] = 0
                
        elif method == "ressfl":
            if "lambda_1" not in filtered_kwargs:
                config_kwargs["lambda_1"] = 1.0
            if "lambda_2" not in filtered_kwargs:
                config_kwargs["lambda_2"] = 0.0
            if "lambda_3" not in filtered_kwargs:
                config_kwargs["lambda_3"] = 0.0
            if "ressfl_use_pretrain" not in filtered_kwargs:
                config_kwargs["ressfl_use_pretrain"] = True
            if "ressfl_source_task" not in filtered_kwargs:
                config_kwargs["ressfl_source_task"] = "same"
            if "ressfl_pretrain_epochs" not in filtered_kwargs:
                config_kwargs["ressfl_pretrain_epochs"] = 50
            if "ressfl_finetune_lr" not in filtered_kwargs:
                config_kwargs["ressfl_finetune_lr"] = 0.005
            if "ressfl_teacher_lambda" not in filtered_kwargs:
                config_kwargs["ressfl_teacher_lambda"] = 0.3
            if "ressfl_warmup_epochs" not in filtered_kwargs:
                config_kwargs["ressfl_warmup_epochs"] = 10
            if "ressfl_update_frequency" not in filtered_kwargs:
                config_kwargs["ressfl_update_frequency"] = 5
            if "ressfl_lambda_rec" not in filtered_kwargs:
                config_kwargs["ressfl_lambda_rec"] = 0.01
            
        elif method == "asfl":
            if "lambda_1" not in filtered_kwargs:
                config_kwargs["lambda_1"] = 1.0
            if "lambda_2" not in filtered_kwargs:
                config_kwargs["lambda_2"] = 1.0
            if "lambda_3" not in filtered_kwargs:
                config_kwargs["lambda_3"] = 0.01
            if "warmup_epochs" not in filtered_kwargs:
                config_kwargs["warmup_epochs"] = 10
            
            # Extract ablation if present
            if "ablation" in filtered_kwargs and filtered_kwargs["ablation"] != "full":
                config_kwargs["ablation"] = filtered_kwargs["ablation"]

        print("\n" + "="*60)
        print("FINAL CONFIG (build_config)")
        print("="*60)
        print(f"lambda_1 = {config_kwargs.get('lambda_1')}")
        print(f"lambda_2 = {config_kwargs.get('lambda_2')}")
        print(f"lambda_3 = {config_kwargs.get('lambda_3')}")
        print(f"warmup_epochs = {config_kwargs.get('warmup_epochs')}")
        print("="*60 + "\n")


        return ExperimentConfig(**config_kwargs)

    def build_dataloaders(self, cfg: ExperimentConfig) -> Tuple[List[DataLoader], List[DataLoader]]:
        """Build train and validation DataLoaders for a dataset."""
        loader_fn = self.DATASET_LOADER_FN.get(cfg.dataset)
        if loader_fn is None:
            raise ValueError(f"Unknown dataset: {cfg.dataset}")

        dataset_config = self.DATASET_PATHS.get(cfg.dataset, {})
        data_dir = dataset_config.get("data_dir", "")
        partition_mode = cfg.partition_mode
        non_iid_alpha = cfg.alpha if cfg.partition_mode == "dirichlet" else None
        num_clients = cfg.num_clients

        train_loaders, val_loaders = loader_fn(
            data_dir=data_dir,
            num_clients=num_clients,
            batch_size=cfg.batch_size,
            partition_mode=partition_mode,
            non_iid_alpha=non_iid_alpha,
            window_size=cfg.window_size,
            normalize=cfg.get_dataset_normalization(),
            val_split=cfg.val_split,
            seed=cfg.seed,
            oversample_target=self._oversample_target,
        )
        return train_loaders, val_loaders

    def build_trainer(self, cfg: ExperimentConfig,
                      train_loaders: List[DataLoader],
                      val_loaders: List[DataLoader]) -> object:
        """
        Build trainer based on method using lazy imports.
        Args:
            cfg: Experiment configuration
            train_loaders: List of training DataLoaders
            val_loaders: List of validation DataLoaders
        """
        method = cfg.method
        if method not in self._available_trainers:
            raise ValueError(
                f"Method '{method}' not available. "
                f"Available methods: {list(self._available_trainers.keys())}"
            )

        trainer_fn = self._available_trainers[method]
        
        # Check if ablation is present in cfg and method is asfl
        if method == "asfl" and hasattr(cfg, "ablation") and cfg.ablation != "full":
            trainer = trainer_fn(cfg, train_loaders, val_loaders, ablation=cfg.ablation)
        else:
            trainer = trainer_fn(cfg, train_loaders, val_loaders)
        
        return trainer

    def run_single_experiment(self, dataset: str, method: str,
                              distribution: str, **kwargs) -> Dict[str, Any]:
        """
        Run a single experiment.
        Args:
            dataset: 'wesad', 'bidmc', 'mhealth'
            method: 'plain', 'ressfl', 'asfl'
            distribution: 'iid', 'noniid'
            **kwargs: Additional config overrides
        Returns:
            Dictionary with experiment results
        """
        exp_path = self.get_experiment_path(dataset, method, distribution)
        os.makedirs(exp_path, exist_ok=True)

        exp_id = self.get_experiment_id(dataset, method, distribution)

        if self.verbose:
            print("=" * 60)
            print(f"Running: {exp_id.upper()}")
            print(f"Output: {exp_path}")
            print("=" * 60)

        cfg = self.build_config(dataset, method, distribution, **kwargs)
        save_config(cfg, os.path.join(exp_path, self.CONFIG_FILE))
        train_loaders, val_loaders = self.build_dataloaders(cfg)
        trainer = self.build_trainer(cfg, train_loaders, val_loaders)

        try:
            history = trainer.fit()
        except Exception:
            print("=" * 60)
            print("FULL TRACEBACK:")
            print("=" * 60)
            traceback.print_exc()
            print("=" * 60)
            return {
                "exp_id": exp_id,
                "dataset": dataset,
                "method": method,
                "distribution": distribution,
                "status": "failed",
                "error": str(sys.exc_info()[1]),
                "path": exp_path,
            }


        eval_results = self._evaluate_from_trainer(trainer, cfg, val_loaders)
        eval_results_serializable = convert_to_serializable(eval_results)
        eval_metrics_path = os.path.join(exp_path, self.EVAL_RESULTS_FILE)
        with open(eval_metrics_path, "w") as f:
            json.dump(eval_results_serializable, f, indent=2)

        result = self._collect_experiment_result(
            dataset, method, distribution, exp_id, exp_path, trainer, eval_results
        )


        result_serializable = convert_to_serializable(result)

        actual_exp_path = cfg.get_output_path()

        # self._standardize_checkpoints(actual_exp_path)
        verification = self._verify_experiment_outputs(actual_exp_path)
        result_serializable["output_verification"] = verification
        self.results.append(result_serializable)

        
        if self.verbose:
            print(f"Completed: {exp_id.upper()}")
            print(f"Best Val Acc: {result.get('best_val_acc', 0.0):.2f}%")
            print(f"Best Privacy MSE: {result.get('best_privacy_mse', 0.0):.4f}")
            print(f"Robust Acc: {result.get('robust_acc', 0.0):.2f}%")
#             print(f"Required outputs verified: {verification.get('required_ok', False)}")
            print("-" * 60)

        return result_serializable

    def _evaluate_from_trainer(self, trainer, cfg: ExperimentConfig,
                               val_loaders: List[DataLoader]) -> Dict[str, Any]:
        """Evaluate using trainer's built-in evaluation methods."""
        eval_results = {
            "classification": {},
            "robustness": {},
            "privacy": {},
            "inversion": {},
        }

        if hasattr(trainer, "evaluate_client"):
            total_acc = 0.0
            total_loss = 0.0
            client_count = 0
            for client_idx in range(len(val_loaders)):
                if len(val_loaders[client_idx].dataset) > 0:
                    sig = inspect.signature(trainer.evaluate_client)
                    kwargs = {}
                    if "compute_reconstruction" in sig.parameters:
                        kwargs["compute_reconstruction"] = True
                    try:
                        result = trainer.evaluate_client(
                            client_idx, val_loaders[client_idx], **kwargs
                        )
                    except TypeError:
                        result = trainer.evaluate_client(
                            client_idx, val_loaders[client_idx]
                        )
                    if isinstance(result, tuple) and len(result) >= 2:
                        loss = result[0]
                        acc = result[1]
                        total_acc += acc
                        total_loss += loss
                        client_count += 1

            if client_count > 0:
                eval_results["classification"]["accuracy"] = total_acc / client_count
                eval_results["classification"]["loss"] = total_loss / client_count

        if hasattr(trainer, "_generate_pgd_attack") and len(val_loaders) > 0:
            total_clean_correct = 0
            total_adv_correct = 0
            total_samples = 0
            trainer_type = type(trainer).__name__
            pgd_eval_steps = getattr(cfg, 'pgd_eval_steps', 40)

            for client_idx in range(len(val_loaders)):
                loader = val_loaders[client_idx]
                if len(loader.dataset) == 0:
                    continue

                for x, y in loader:
                    x = x.to(cfg.device)
                    y = y.to(cfg.device)
                    
                    # Convert one-hot labels to class indices if needed
                    if y.dim() > 1 and y.shape[1] > 1:
                        y = y.argmax(dim=1)
                    
                    total_samples += y.size(0)

                    with torch.no_grad():
                        if trainer_type == "PlainSFLTrainer":
                            smashed = trainer._client_forward(client_idx, x)
                            logits = trainer._server_forward(smashed)
                        elif trainer_type == "ResSFLTrainer":
                            zu, _ = trainer._client_forward(client_idx, x)
                            logits = trainer._server_forward(zu)
                        else:
                            zu, _, _ = trainer._client_forward(client_idx, x)
                            logits = trainer._server_forward(zu)

                        clean_preds = logits.argmax(dim=1)
                        total_clean_correct += (clean_preds == y).sum().item()

                    # Pass eval_mode=True to use pgd_eval_steps
                    x_adv = trainer._generate_pgd_attack(client_idx, x, y, eval_mode=True)

                    with torch.no_grad():
                        if trainer_type == "PlainSFLTrainer":
                            smashed_adv = trainer._client_forward(client_idx, x_adv)
                            adv_logits = trainer._server_forward(smashed_adv)
                        elif trainer_type == "ResSFLTrainer":
                            adv_zu, _ = trainer._client_forward(client_idx, x_adv)
                            adv_logits = trainer._server_forward(adv_zu)
                        else:
                            adv_zu, _, _ = trainer._client_forward(client_idx, x_adv)
                            adv_logits = trainer._server_forward(adv_zu)

                        adv_preds = adv_logits.argmax(dim=1)
                        total_adv_correct += (adv_preds == y).sum().item()

            if total_samples > 0:
                clean_acc = 100.0 * total_clean_correct / total_samples
                adv_acc = 100.0 * total_adv_correct / total_samples
                eval_results["robustness"]["clean_acc"] = clean_acc
                eval_results["robustness"]["adv_acc"] = adv_acc
                eval_results["robustness"]["robust_acc"] = adv_acc
                eval_results["robustness"]["pgd_eval_steps"] = pgd_eval_steps
            else:
                eval_results["robustness"]["clean_acc"] = 0.0
                eval_results["robustness"]["adv_acc"] = 0.0
                eval_results["robustness"]["robust_acc"] = 0.0
                eval_results["robustness"]["pgd_eval_steps"] = pgd_eval_steps

        if hasattr(trainer, "evaluate_inversion_attack"):
            inv_results = trainer.evaluate_inversion_attack()
            eval_results["inversion"] = inv_results
            eval_results["privacy"]["mse_gl0"] = inv_results.get("mse_gl0", 0.0)
            eval_results["privacy"]["mse_gl1"] = inv_results.get("mse_gl1", 0.0)
            eval_results["privacy"]["mse_gl2"] = inv_results.get("mse_gl2", 0.0)
            eval_results["privacy"]["mse_gl3"] = inv_results.get("mse_gl3", 0.0)
            eval_results["privacy"]["mse_adaptive"] = inv_results.get("mse_adaptive", 0.0)
            eval_results["privacy"]["privacy_mse"] = inv_results.get("privacy_mse", 0.0)
            eval_results["privacy"]["strongest_attack_mse"] = inv_results.get("strongest_attack_mse", 0.0)
            eval_results["privacy"]["strongest_attacker"] = inv_results.get("strongest_attacker", "None")
            eval_results["privacy"]["best_mse"] = inv_results.get("best_mse", 0.0)
            eval_results["privacy"]["best_attacker"] = inv_results.get("best_attacker", "None")

        return eval_results

    def _collect_experiment_result(self, dataset: str, method: str,
                                   distribution: str, exp_id: str,
                                   exp_path: str, trainer,
                                   eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Collect experiment results into a standardized format."""
        result = {
            "exp_id": exp_id,
            "dataset": dataset,
            "method": method,
            "distribution": distribution,
            "status": "success",
            "path": exp_path,
        }

        result["best_val_acc"] = getattr(trainer, "best_val_acc", 0.0)

        if hasattr(trainer, "history") and trainer.history:
            if "train_acc" in trainer.history and trainer.history["train_acc"]:
                result["final_train_acc"] = trainer.history["train_acc"][-1]
            else:
                result["final_train_acc"] = 0.0

            if "val_acc" in trainer.history and trainer.history["val_acc"]:
                result["final_val_acc"] = trainer.history["val_acc"][-1]
            else:
                result["final_val_acc"] = 0.0

            if "train_loss" in trainer.history and trainer.history["train_loss"]:
                result["final_train_loss"] = trainer.history["train_loss"][-1]
            else:
                result["final_train_loss"] = 0.0

            if "val_loss" in trainer.history and trainer.history["val_loss"]:
                result["final_val_loss"] = trainer.history["val_loss"][-1]
            else:
                result["final_val_loss"] = 0.0
        else:
            result["final_train_acc"] = 0.0
            result["final_val_acc"] = 0.0
            result["final_train_loss"] = 0.0
            result["final_val_loss"] = 0.0

        result["best_privacy_mse"] = getattr(trainer, "best_privacy_mse", 0.0)

        result["mse_gl0"] = eval_results.get("privacy", {}).get("mse_gl0", 0.0)
        result["mse_gl1"] = eval_results.get("privacy", {}).get("mse_gl1", 0.0)
        result["mse_gl2"] = eval_results.get("privacy", {}).get("mse_gl2", 0.0)
        result["mse_gl3"] = eval_results.get("privacy", {}).get("mse_gl3", 0.0)
        result["mse_adaptive"] = eval_results.get("privacy", {}).get("mse_adaptive", 0.0)
        result["privacy_mse"] = eval_results.get("privacy", {}).get("privacy_mse", 0.0)
        result["strongest_attack_mse"] = eval_results.get("privacy", {}).get("strongest_attack_mse", 0.0)
        result["strongest_attacker"] = eval_results.get("privacy", {}).get("strongest_attacker", "None")
        result["best_mse"] = eval_results.get("privacy", {}).get("best_mse", 0.0)
        result["best_attacker"] = eval_results.get("privacy", {}).get("best_attacker", "None")

        result["clean_acc"] = eval_results.get("robustness", {}).get("clean_acc", 0.0)
        result["robust_acc"] = eval_results.get("robustness", {}).get("robust_acc", 0.0)
        result["adv_acc"] = eval_results.get("robustness", {}).get("adv_acc", 0.0)
        result["pgd_eval_steps"] = eval_results.get("robustness", {}).get("pgd_eval_steps", 40)

        return result

    def run_iid_suite(self, **kwargs) -> List[Dict[str, Any]]:
        """Run all IID experiments."""
        results = []
        available_methods = list(self._available_trainers.keys())
        if not available_methods:
            print("No trainers available. Cannot run IID experiments.")
            return results

        for dataset in self.DATASETS:
            for method in available_methods:
                result = self.run_single_experiment(dataset, method, "iid", **kwargs)
                results.append(result)
        return results

    def run_noniid_suite(self, **kwargs) -> List[Dict[str, Any]]:
        """Run all Non-IID experiments."""
        results = []
        available_methods = list(self._available_trainers.keys())
        if not available_methods:
            print("No trainers available. Cannot run Non-IID experiments.")
            return results

        for dataset in self.DATASETS:
            for method in available_methods:
                result = self.run_single_experiment(dataset, method, "noniid", **kwargs)
                results.append(result)
        return results

    def run_all_experiments(self, **kwargs) -> List[Dict[str, Any]]:
        """Run all experiments."""
        self.results = []
        available_methods = list(self._available_trainers.keys())
        if not available_methods:
            print("No trainers available. Cannot run experiments.")
            return []

        if self.verbose:
            print("\n" + "=" * 60)
            print("Running All Available Experiments")
            print("=" * 60)
            print(f"Datasets: {self.DATASETS}")
            print(f"Methods: {available_methods}")
            print(f"Distributions: {self.DISTRIBUTIONS}")
            print("=" * 60 + "\n")

        results = self.run_iid_suite(**kwargs)
        results.extend(self.run_noniid_suite(**kwargs))
        self.save_summary()
        return results

    def collect_results(self) -> List[Dict[str, Any]]:
        """
        Collect results from all experiment directories.
        Supports both metrics.json (from trainers) and evaluation_results.json (from this script).
        """
        collected = []
        for dist in self.DISTRIBUTIONS:
            for dataset in self.DATASETS:
                for method in self.METHODS:
                    exp_path = self.get_experiment_path(dataset, method, dist)
                    eval_path = os.path.join(exp_path, self.EVAL_RESULTS_FILE)
                    
                    # Prefer metrics.json if it exists, otherwise use evaluation_results.json
                    if os.path.exists(eval_path):
                        path_to_read = eval_path
                    else:
                        continue
                    
                    try:
                        with open(path_to_read, "r") as f:
                            data = json.load(f)
                        
                        collected_data = {
                            "dataset": dataset,
                            "method": method,
                            "distribution": dist,
                        }

                        if "best_val_acc" in data:
                            collected_data["best_val_acc"] = data["best_val_acc"]
                        elif "classification" in data and "accuracy" in data["classification"]:
                            collected_data["best_val_acc"] = data["classification"]["accuracy"]
                        elif "history" in data and "val_acc" in data["history"]:
                            collected_data["best_val_acc"] = data["history"]["val_acc"][-1] if data["history"]["val_acc"] else 0.0
                        else:
                            collected_data["best_val_acc"] = 0.0

                        if "privacy_mse" in data:
                            collected_data["privacy_mse"] = data["privacy_mse"]
                        elif "privacy" in data and "privacy_mse" in data["privacy"]:
                            collected_data["privacy_mse"] = data["privacy"]["privacy_mse"]
                        elif "best_mse" in data:
                            collected_data["privacy_mse"] = data["best_mse"]
                        elif "privacy" in data and "best_mse" in data["privacy"]:
                            collected_data["privacy_mse"] = data["privacy"]["best_mse"]
                        else:
                            collected_data["privacy_mse"] = 0.0

                        if "strongest_attacker" in data:
                            collected_data["strongest_attacker"] = data["strongest_attacker"]
                        elif "privacy" in data and "strongest_attacker" in data["privacy"]:
                            collected_data["strongest_attacker"] = data["privacy"]["strongest_attacker"]
                        elif "best_attacker" in data:
                            collected_data["strongest_attacker"] = data["best_attacker"]
                        elif "privacy" in data and "best_attacker" in data["privacy"]:
                            collected_data["strongest_attacker"] = data["privacy"]["best_attacker"]
                        else:
                            collected_data["strongest_attacker"] = "None"

                        if "strongest_attack_mse" in data:
                            collected_data["strongest_attack_mse"] = data["strongest_attack_mse"]
                        elif "privacy" in data and "strongest_attack_mse" in data["privacy"]:
                            collected_data["strongest_attack_mse"] = data["privacy"]["strongest_attack_mse"]
                        else:
                            collected_data["strongest_attack_mse"] = 0.0

                        if "robust_acc" in data:
                            collected_data["robust_acc"] = data["robust_acc"]
                        elif "robustness" in data and "robust_acc" in data["robustness"]:
                            collected_data["robust_acc"] = data["robustness"]["robust_acc"]
                        elif "robustness" in data and "adv_acc" in data["robustness"]:
                            collected_data["robust_acc"] = data["robustness"]["adv_acc"]
                        else:
                            collected_data["robust_acc"] = 0.0

                        if "mse_gl0" in data:
                            collected_data["mse_gl0"] = data["mse_gl0"]
                        elif "privacy" in data and "mse_gl0" in data["privacy"]:
                            collected_data["mse_gl0"] = data["privacy"]["mse_gl0"]
                        else:
                            collected_data["mse_gl0"] = 0.0

                        if "pgd_eval_steps" in data:
                            collected_data["pgd_eval_steps"] = data["pgd_eval_steps"]
                        elif "robustness" in data and "pgd_eval_steps" in data["robustness"]:
                            collected_data["pgd_eval_steps"] = data["robustness"]["pgd_eval_steps"]
                        else:
                            collected_data["pgd_eval_steps"] = 40

                        collected.append(collected_data)
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        if self.verbose:
                            print(f"Warning: Could not read {path_to_read}: {e}")
                        pass

        return collected

    def aggregate_main_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Aggregate main results for all experiments."""
        aggregated = {
            "iid": [],
            "noniid": [],
        }
        for result in self.results:
            dist = result.get("distribution", "unknown")
            if dist in aggregated:
                aggregated[dist].append(result)
        return aggregated


    def save_summary(self, filename: str = "experiment_summary.csv") -> str:
        """Save experiment summary to CSV with timestamp to avoid overwriting."""
        import time
        
        if not self.results:
            self.results = self.collect_results()
    
        if not self.results:
            print("No results to save.")
            return ""
    
        # Add timestamp to filename to prevent overwriting
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        filename_with_timestamp = f"{name}_{timestamp}{ext}"
        summary_path = os.path.join(self.base_output_dir, filename_with_timestamp)
    
        columns = [
            "exp_id", "dataset", "method", "distribution", "status",
            "best_val_acc", "best_privacy_mse",
            "mse_gl0", "mse_gl1", "mse_gl2", "mse_gl3", "mse_adaptive",
            "privacy_mse", "strongest_attack_mse", "strongest_attacker",
            "clean_acc", "robust_acc", "adv_acc",
            "pgd_eval_steps"
        ]
    
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for result in self.results:
                if "exp_id" not in result:
                    result["exp_id"] = self.get_experiment_id(
                        result.get("dataset", ""),
                        result.get("method", ""),
                        result.get("distribution", "")
                    )
                result_serializable = convert_to_serializable(result)
                writer.writerow(result_serializable)
    
        if self.verbose:
            print(f"Summary saved to: {summary_path}")
        return summary_path


def get_experiment_parser() -> argparse.ArgumentParser:
    """Create argument parser for experiments."""
    parser = argparse.ArgumentParser(description="ASFL Experiment Runner")
    parser.add_argument("--dataset", type=str, choices=["wesad", "bidmc", "mhealth"],
                        help="Dataset to run")
    parser.add_argument("--method", type=str, choices=["plain", "ressfl", "asfl"],
                        help="Method to run")
    parser.add_argument("--distribution", type=str, choices=["iid", "noniid"],
                        help="Distribution to run")
    parser.add_argument("--run_all", action="store_true", default=False,
                        help="Run all experiments")
    parser.add_argument("--run_iid", action="store_true", default=False,
                        help="Run all IID experiments")
    parser.add_argument("--run_noniid", action="store_true", default=False,
                        help="Run all Non-IID experiments")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Base output directory")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=99,
                        help="Random seed")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs (sets both epochs and main_epochs)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--pgd_eval_steps", type=int, default=40,
                        help="Number of PGD steps for evaluation")
    return parser


def main():
    """Main entry point for experiment runner."""
    parser = get_experiment_parser()
    args = parser.parse_args()

    runner = ExperimentRunner(
        base_output_dir=args.output_dir,
        device=args.device,
        verbose=True,
        seed=args.seed
    )

    overrides = {}
    if args.epochs != 50:
        overrides["epochs"] = args.epochs
        overrides["main_epochs"] = args.epochs
    if args.batch_size != 32:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate != 1e-3:
        overrides["learning_rate"] = args.learning_rate
    if args.pgd_eval_steps != 40:
        overrides["pgd_eval_steps"] = args.pgd_eval_steps

    if args.run_all:
        runner.run_all_experiments(**overrides)
    elif args.run_iid:
        runner.run_iid_suite(**overrides)
    elif args.run_noniid:
        runner.run_noniid_suite(**overrides)
    elif args.dataset and args.method and args.distribution:
        runner.run_single_experiment(
            args.dataset, args.method, args.distribution, **overrides
        )
    else:
        print("Error: Please specify --dataset, --method, --distribution, or use --run_all")
        parser.print_help()
        sys.exit(1)

    runner.save_summary()
    print("\n" + "=" * 60)
    print("Experiments Complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()