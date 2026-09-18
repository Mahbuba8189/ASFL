# asfl/config.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import os
import json
import argparse


@dataclass
class ExperimentConfig:
   
    # General Settings
    seed: int = 99
    experiment_name: str = "asfl"
    output_dir: str = "./outputs"
    experiment_type: str = "main"

    # Method Selection
    method: str = "asfl"  # 'plain', 'ressfl', 'asfl'

    # Training Settings (CONFIGURABLE)
    epochs: int = 50                     
    main_epochs: int = 50                
    warmup_epochs: int = 10               
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-5
    use_class_weights: bool = True       

    # Dataset Settings
    dataset: str = "wesad"
    window_size: int = 8
    alpha: Optional[float] = None  # Dirichlet alpha for non-IID partitioning

    dataset_normalization: Dict[str, str] = field(default_factory=lambda: {
        "wesad": "zscore",
        "bidmc": "minmax",
        "mhealth": "zscore"
    })

    dataset_input_dims: Dict[str, int] = field(default_factory=lambda: {
        "wesad": 44800,
        "bidmc": 1000,
        "mhealth": 7000
    })

    dataset_num_classes: Dict[str, int] = field(default_factory=lambda: {
        "wesad": 3,
        "bidmc": 3,
        "mhealth": 12
    })

    dataset_class_names: Dict[str, List[str]] = field(default_factory=lambda: {
        "wesad": ["Baseline", "Stress", "Amusement"],
        "bidmc": ["Low HR (<60)", "Normal HR (60-100)", "High HR (>100)"],
        "mhealth": [
            "Standing", "Sitting", "Lying", "Walking",
            "Climbing Stairs", "Waist Bends", "Arm Elevation",
            "Knees Bending", "Cycling", "Jogging", "Running", "Jumping"
        ]
    })

    dataset_tier_dims: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "wesad": {
            "sensor": 64,
            "gateway": 128,
            "edge": 256,
        },
        "bidmc": {
            "sensor": 16,
            "gateway": 32,
            "edge": 64,
        },
        "mhealth": {
            "sensor": 16,
            "gateway": 32,
            "edge": 64,
        }
    })

    dataset_clients: Dict[str, int] = field(default_factory=lambda: {
        "wesad": 15,
        "bidmc": 53,
        "mhealth": 10
    })


    # Bottleneck Configuration
    device_tier: str = "gateway"

    # Client Partitioning (CONFIGURABLE)
    partition_mode: str = "natural"  # 'natural', 'iid', 'dirichlet'
    dirichlet_alpha: float = 0.5
    num_clients: Optional[int] = None
    val_split: float = 0.2
    temporal_split: bool = True

    # Model Architecture (CONFIGURABLE)
    cut_layer: int = 3
    stem_channels: int = 32
    width_mult: float = 1.0
    server_hidden_dim: int = 64
    utility_hidden_dim: int = 128
    privacy_hidden_dim: int = 64
    dropout: float = 0.2

    # Projection Dimensions (CONFIGURABLE)
    privacy_dim: int = 32

    # Loss Weights (CONFIGURABLE - ASFL)
    lambda_1: float = 1.0   # Reconstruction loss weight
    lambda_2: float = 1.0   # Adversarial loss weight
    lambda_3: float = 0.01  # Privacy penalty weight

    # Adversarial Settings (CONFIGURABLE)
    pgd_epsilon: float = 0.3
    pgd_steps: int = 40      # Training steps (or default eval)
    pgd_eval_steps: int = 40 # Evaluation steps
    pgd_alpha: float = 0.01
    adversary_level: int = 2  # 0=GL0, 1=GL1, 2=GL2, 3=GL3, 4=Adaptive

    # Device Settings
    device: str = "cuda"
    num_workers: int = 4

    # ResSFL-specific Settings (for expert pre-training)
 
    ressfl_pretrain_epochs: int = 50          # Expert pre-training epochs
    ressfl_warmup_epochs: int = 10             # ResSFL warm-up (λ_rec=0 during warmup)
    ressfl_lambda_rec: float = 0.01            # Reconstruction loss weight (λ_rec) for fine-tuning
    ressfl_teacher_lambda: float = 0.3         # Inversion score strength (λ) for expert pre-training
    ressfl_finetune_lr: float = 0.005          # Lower LR for fine-tuning (matches ResSFL paper)
    ressfl_update_frequency: int = 5           # Decoder update every N steps
    ressfl_use_pretrain: bool = False          # Enable/disable expert pre-training
    ressfl_source_task: str = "same"           # Source task ('same', 'wesad', 'bidmc', mhealth')
                                              

    # Logging and Checkpoint Settings
    log_interval: int = 10
    save_interval: int = 50
    eval_interval: int = 10

    def __post_init__(self):
        # Synchronize epochs and main_epochs (single source of truth)
        # If only epochs was set (backward compatibility), use it
        # If only main_epochs was set, use it
        # If both were set and different, main_epochs takes precedence
        if self.main_epochs == 50 and self.epochs != 50:
            # Backward compatibility: epochs was explicitly set, main_epochs wasn't
            self.main_epochs = self.epochs
        elif self.epochs == 50 and self.main_epochs != 50:
            # Forward compatibility: main_epochs was explicitly set
            self.epochs = self.main_epochs
        elif self.epochs != self.main_epochs:
            # Both were set but differ - main_epochs takes precedence
            self.epochs = self.main_epochs
        
        # Now both epochs and main_epochs are guaranteed to be equal

        if self.num_clients is None:
            self.num_clients = self.dataset_clients.get(self.dataset, 10)

        if self.dataset not in self.dataset_clients:
            raise ValueError(
                f"Unknown dataset: {self.dataset}. "
                f"Choose from: {list(self.dataset_clients.keys())}"
            )

        if self.partition_mode not in ["natural", "iid", "dirichlet"]:
            raise ValueError(
                f"Unknown partition_mode: {self.partition_mode}. "
                "Choose from: natural, iid, dirichlet"
            )

        if self.method not in ["plain", "ressfl", "asfl"]:
            raise ValueError(
                f"Unknown method: {self.method}. "
                "Choose from: plain, ressfl, asfl"
            )

        if self.device_tier != "gateway":
            raise ValueError(
                f"device_tier must be 'gateway', got '{self.device_tier}'"
            )
       
        # Validate ResSFL source task
        if self.ressfl_source_task not in ["same", "wesad", "bidmc", "mhealth"]:
            raise ValueError(
                f"Unknown ressfl_source_task: {self.ressfl_source_task}. "
                "Choose from: same, wesad, bidmc, mhealth"
            )

        # Validate adversary level (0-4, where 4 = Adaptive)
        if self.adversary_level not in [0, 1, 2, 3, 4]:
            raise ValueError(
                f"Unknown adversary_level: {self.adversary_level}. "
                "Choose from: 0 (GL0), 1 (GL1), 2 (GL2), 3 (GL3), 4 (Adaptive)"
            )

    # Getter Methods
    def get_bottleneck_dim(self) -> int:
        return self.dataset_tier_dims.get(self.dataset, {}).get("gateway", 64)

    def get_dataset_input_dim(self) -> int:
        """Get input dimension for the current dataset."""
        if self.dataset == "mhealth":
            return 14 * 50 * self.window_size
        return self.dataset_input_dims.get(self.dataset, 1000)

    def get_dataset_num_classes(self) -> int:
        """Get number of classes for the current dataset."""
        return self.dataset_num_classes.get(self.dataset, 3)

    def get_dataset_class_names(self) -> List[str]:
        """Get class names for the current dataset."""
        return self.dataset_class_names.get(self.dataset, [])

    def get_dataset_normalization(self) -> str:
        """Get normalization method for the current dataset."""
        return self.dataset_normalization.get(self.dataset, "zscore")

    def get_dataset_num_clients(self) -> int:
        """Get number of clients for the current dataset."""
        return self.dataset_clients.get(self.dataset, 10)

    def get_ressfl_source_task_name(self) -> str:
        """Get the name of the source task for display purposes."""
        if self.ressfl_source_task == "same":
            return self.dataset.upper()
        return self.ressfl_source_task.upper()

    def get_tier_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of bottleneck dimensions for all tiers."""
        summary = {}
        for tier in ["sensor", "gateway", "edge"]:
            summary[tier] = {
                "bottleneck_dim": self.dataset_tier_dims.get(self.dataset, {}).get(tier, 64),
            }
        return summary

    # Configuration Dictionaries

    def get_model_config(self) -> Dict[str, Any]:
        """Get model architecture configuration."""
        return {
            "cut_layer": self.cut_layer,
            "stem_channels": self.stem_channels,
            "width_mult": self.width_mult,
            "server_hidden_dim": self.server_hidden_dim,
            "utility_hidden_dim": self.utility_hidden_dim,
            "privacy_hidden_dim": self.privacy_hidden_dim,
            "dropout": self.dropout,
            "privacy_dim": self.privacy_dim,
        }

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
        }

    def get_pgd_config(self) -> Dict[str, Any]:
        """Get PGD attack configuration."""
        return {
            "epsilon": self.pgd_epsilon,
            "steps": self.pgd_steps,
            "eval_steps": self.pgd_eval_steps,
            "alpha": self.pgd_alpha,
        }

    def get_loss_config(self) -> Dict[str, Any]:
        """Get ASFL loss weights."""
        return {
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "lambda_3": self.lambda_3,
        }

    def get_ressfl_config(self) -> Dict[str, Any]:
        """Get ResSFL-specific configuration."""
        return {
            "pretrain_epochs": self.ressfl_pretrain_epochs,
            "warmup_epochs": self.ressfl_warmup_epochs,
            "lambda_rec": self.ressfl_lambda_rec,
            "teacher_lambda": self.ressfl_teacher_lambda,
            "finetune_lr": self.ressfl_finetune_lr,
            "update_frequency": self.ressfl_update_frequency,
            "use_pretrain": self.ressfl_use_pretrain,
            "source_task": self.ressfl_source_task,
            "source_task_name": self.get_ressfl_source_task_name(),
        }

    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return {
            "name": self.dataset,
            "input_dim": self.get_dataset_input_dim(),
            "num_classes": self.get_dataset_num_classes(),
            "num_clients": self.get_dataset_num_clients(),
            "bottleneck_dim": self.get_bottleneck_dim(),
            "normalization": self.get_dataset_normalization(),
            "window_size": self.window_size,
        }

    
    # Experiment Paths

    def get_experiment_name(self) -> str:
        """Generate experiment name."""
        base_name = (
            f"{self.experiment_name}_"
            f"{self.method}_"
            f"{self.dataset}_"
            f"{self.partition_mode}_"
            f"clients{self.num_clients}_"
            f"seed{self.seed}"
        )
        
        # Add ResSFL-specific info to experiment name if pre-training is enabled
        if self.method == "ressfl" and self.ressfl_use_pretrain:
            base_name += f"_pretrain_{self.ressfl_source_task}"
        
        return base_name

    def get_output_path(self) -> str:
        """Get output directory path for this experiment."""
        exp_dir = os.path.join(self.output_dir, self.get_experiment_name())
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            k: v for k, v in vars(self).items()
            if not k.startswith("_")
        }


# Helper Functions

def update_config_from_args(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Update config from command-line arguments."""
    for key, value in vars(args).items():
        if hasattr(cfg, key) and value is not None:
            setattr(cfg, key, value)
    
    # Ensure epochs and main_epochs stay synchronized after arg updates
    # (main_epochs takes precedence if both were provided)
    if hasattr(args, 'main_epochs') and args.main_epochs is not None:
        cfg.epochs = cfg.main_epochs
    elif hasattr(args, 'epochs') and args.epochs is not None:
        cfg.main_epochs = cfg.epochs
    
    return cfg

def save_config(cfg: ExperimentConfig, path: Optional[str] = None) -> str:
    if path is None:
        output_dir = cfg.get_output_path()
        base_path = os.path.join(output_dir, "config")
    else:
        base_path = os.path.splitext(path)[0]

    config_dict = cfg.to_dict()

    # Only save JSON
    json_path = f"{base_path}.json"
    with open(json_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    return json_path

def load_config(path: str) -> ExperimentConfig:
    """Load config from JSON file."""
    with open(path, "r") as f:
        config_dict = json.load(f)
    return ExperimentConfig(**config_dict)


def get_dataset_config(dataset: str, device_tier: str = "gateway") -> Dict[str, Any]:
    """
    Get dataset-specific configuration.
    
    Note: This function is a convenience wrapper around the dataclass getters.
    For consistency, prefer using ExperimentConfig.get_dataset_config().
    """
    # Use a temporary config to access dataset metadata
    temp_cfg = ExperimentConfig(dataset=dataset, device_tier=device_tier)
    return temp_cfg.get_dataset_config()


def get_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for command-line configuration."""
    parser = argparse.ArgumentParser(description="ASFL Experiment Configuration")

    # General
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--dataset", type=str, choices=["wesad", "bidmc", "mhealth"], default="wesad")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--experiment_type", type=str, choices=["main", "ablation"], default="main")

    # Method
    parser.add_argument("--method", type=str, choices=["plain", "ressfl", "asfl"], default="asfl")

    # Training
    parser.add_argument("--epochs", type=int, default=50, 
                        help="DEPRECATED: Use --main_epochs instead")
    parser.add_argument("--main_epochs", type=int, default=50, 
                        help="Primary training epochs (used by all trainers)")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--use_class_weights", action="store_true", 
                        help="Enable class-weighted CrossEntropyLoss")

    # Bottleneck preset
    parser.add_argument("--device_tier", type=str, choices=["gateway"], default="gateway")

    # Client partitioning
    parser.add_argument("--partition_mode", type=str, choices=["natural", "iid", "dirichlet"], default="natural")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha for non-IID partitioning (e.g., 0.1, 0.5, 1.0)")
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.2)

    # Model Architecture (CONFIGURABLE)
    parser.add_argument("--cut_layer", type=int, default=3,
                        help="Cut layer for split (default: 3)")
    parser.add_argument("--stem_channels", type=int, default=32,
                        help="Stem channels for backbone (default: 32)")
    parser.add_argument("--width_mult", type=float, default=1.0,
                        help="Width multiplier for backbone (default: 1.0)")
    parser.add_argument("--server_hidden_dim", type=int, default=64,
                        help="Server classifier hidden dim (default: 64)")
    parser.add_argument("--utility_hidden_dim", type=int, default=128,
                        help="Utility projection hidden dim (default: 128)")
    parser.add_argument("--privacy_hidden_dim", type=int, default=64,
                        help="Privacy projection hidden dim (default: 64)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (default: 0.2)")
    
    # Projection
    parser.add_argument("--privacy_dim", type=int, default=32)

    # Loss weights
    parser.add_argument("--lambda_1", type=float, default=1.0)
    parser.add_argument("--lambda_2", type=float, default=1.0)
    parser.add_argument("--lambda_3", type=float, default=0.01)

    # Adversarial
    parser.add_argument("--pgd_epsilon", type=float, default=0.3)
    parser.add_argument("--pgd_steps", type=int, default=40)
    parser.add_argument("--pgd_eval_steps", type=int, default=40)
    parser.add_argument("--pgd_alpha", type=float, default=0.01)


    # ResSFL-specific Arguments
    parser.add_argument("--ressfl_pretrain_epochs", type=int, default=50,
                        help="Expert pre-training epochs")
    parser.add_argument("--ressfl_warmup_epochs", type=int, default=10,
                        help="ResSFL warm-up (λ_rec=0 during warmup)")
    parser.add_argument("--ressfl_lambda_rec", type=float, default=0.01,
                        help="Reconstruction loss weight (λ_rec) for fine-tuning")
    parser.add_argument("--ressfl_teacher_lambda", type=float, default=0.3,
                        help="Inversion score strength (λ) for expert pre-training")
    parser.add_argument("--ressfl_finetune_lr", type=float, default=0.005,
                        help="Lower LR for fine-tuning (matches ResSFL paper)")
    parser.add_argument("--ressfl_update_frequency", type=int, default=5,
                        help="Decoder update every N steps")
    parser.add_argument("--ressfl_use_pretrain", action="store_true", default=False,
                        help="Enable/disable expert pre-training")
    parser.add_argument("--ressfl_source_task", type=str, 
                        choices=["same", "wesad", "bidmc", "mhealth"], default="same",
                        help="Source task for expert pre-training (override for transfer experiments)")

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=10)

    # Ablation study
    parser.add_argument("--run_ablation_dataset", type=str, 
                        choices=["wesad", "bidmc", "mhealth"], default="wesad",
                        help="Dataset for ablation study")

    return parser