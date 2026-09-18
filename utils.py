# asfl/utils.py

import os
import json
import csv
import random
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import torch
import numpy as np

# 1. Reproducibility

def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 2. Device Utilities

def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for PyTorch operations.
    
    Args:
        device: Optional device string ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        torch.device: The device to use
    """
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_device_info() -> None:
    """Print system and PyTorch device information."""
    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 60)


# 3. File Utilities

def ensure_dir(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        str: The directory path (for chaining)
    """
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def get_timestamp() -> str:
    """
    Get a formatted timestamp for file naming.
    
    Returns:
        str: Timestamp string (YYYY-MM-DD_HH-MM-SS)
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


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


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation level
    """
    directory = os.path.dirname(filepath)
    if directory:
        ensure_dir(directory)
    
    # Convert data to JSON serializable format
    serializable_data = convert_to_serializable(data)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=indent)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Input file path
    
    Returns:
        Dict[str, Any]: Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_csv(rows: List[Dict[str, Any]], filepath: str, fieldnames: Optional[List[str]] = None) -> None:
    """
    Save a list of dictionaries to a CSV file.
    
    Args:
        rows: List of dictionaries to save
        filepath: Output file path
        fieldnames: Optional list of column names (uses first row keys if None)
    """
    directory = os.path.dirname(filepath)
    if directory:
        ensure_dir(directory)
    
    if not rows:
        return
    
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    
    # Convert all rows to serializable format
    serializable_rows = [convert_to_serializable(row) for row in rows]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable_rows)


def append_csv(row: Dict[str, Any], filepath: str, fieldnames: Optional[List[str]] = None) -> None:
    """
    Append a single row to a CSV file.
    
    Args:
        row: Dictionary to append
        filepath: Output file path
        fieldnames: Optional list of column names
    """
    directory = os.path.dirname(filepath)
    if directory:
        ensure_dir(directory)
    
    write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
    
    if fieldnames is None:
        fieldnames = list(row.keys())
    
    # Convert row to serializable format
    serializable_row = convert_to_serializable(row)
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(serializable_row)


# 4. Logging Utilities

def print_banner(title: str, char: str = "=", width: int = 60) -> None:
    """
    Print a banner with a title for visual separation in logs.
    
    Args:
        title: Banner title
        char: Character to use for the border
        width: Total width of the banner
    """
    print(char * width)
    print(title.center(width))
    print(char * width)


def log_metrics(metrics: Dict[str, float], prefix: str = "", width: int = 30) -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for each line
        width: Width for metric names before printing values
    """
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{prefix}{name:<{width}}: {value:.4f}")
        else:
            print(f"{prefix}{name:<{width}}: {value}")


# 5. Metric Utilities

class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics across training epochs and batches.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter with a new value.
        
        Args:
            val: New value
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
    
    def __repr__(self) -> str:
        return f"AverageMeter(val={self.val:.4f}, avg={self.avg:.4f})"


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """
    Compute mean and standard deviation of a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Tuple[float, float]: (mean, std)
    """
    if not values:
        return 0.0, 0.0
    mean = np.mean(values)
    std = np.std(values)
    return float(mean), float(std)


# 6. Model Utilities

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Tuple[int, int]: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model: torch.nn.Module, name: str = "Model", device: Optional[torch.device] = None) -> None:
    """
    Print a summary of a PyTorch model.
    
    Args:
        model: PyTorch model
        name: Name of the model for display
        device: Optional device to display
    """
    total, trainable = count_parameters(model)
    
    print("=" * 60)
    print(f"{name} Summary")
    print("=" * 60)
    print(f"  Type: {type(model).__name__}")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    if device is not None:
        print(f"  Device: {device}")
    print("=" * 60)


def get_parameter_count_str(model: torch.nn.Module) -> str:
    """
    Get a formatted string of model parameter counts.
    
    Args:
        model: PyTorch model
    
    Returns:
        str: Formatted parameter count string
    """
    total, trainable = count_parameters(model)
    return f"Params: {total:,} (trainable: {trainable:,})"


# 7. Experiment Utilities

def generate_experiment_id(dataset: str, method: str, distribution: str, seed: int) -> str:
    """
    Generate a unique experiment ID including seed.
    
    Args:
        dataset: Dataset name ('wesad', 'bidmc', 'mhealth')
        method: Method name ('plain', 'ressfl', 'asfl')
        distribution: Distribution ('iid', 'noniid')
        seed: Random seed
    
    Returns:
        str: Experiment ID
    """
    return f"{dataset}_{method}_{distribution}_seed{seed}"


def format_result_row(result: Dict[str, Any]) -> Dict[str, str]:
    """
    Format a result dictionary for table output.
    
    Args:
        result: Result dictionary with numeric values
    
    Returns:
        Dict[str, str]: Formatted result dictionary
    """
    formatted = {}
    
    accuracy_fields = ["best_val_acc", "clean_acc", "robust_acc", "adv_acc"]
    
    # Copy string fields
    for key in ["dataset", "method", "distribution", "status", "best_attacker"]:
        if key in result:
            formatted[key] = str(result.get(key, "N/A"))
    
    # Format numeric fields
    numeric_fields = [
        "best_val_acc", "best_privacy_mse", "clean_acc", "robust_acc", "adv_acc",
        "mse_gl0", "mse_gl1", "mse_gl2", "mse_gl3", "mse_adaptive", "best_mse"
    ]
    for key in numeric_fields:
        value = result.get(key)
        if value is not None:
            if isinstance(value, float):
                if key in accuracy_fields:
                    formatted[key] = f"{value:.2f}"
                else:
                    formatted[key] = f"{value:.4f}"
            else:
                formatted[key] = str(value)
        else:
            formatted[key] = "N/A"
    
    return formatted


def get_experiment_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary of experiment results.
    
    Args:
        results: List of experiment result dictionaries
    
    Returns:
        Dict[str, Any]: Summary statistics
    """
    if not results:
        return {"total": 0}
    
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]
    
    accuracies = [r.get("best_val_acc", 0.0) for r in successful]
    mse_values = [r.get("best_mse", 0.0) for r in successful]
    robust_accs = [r.get("robust_acc", 0.0) for r in successful]
    
    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "accuracy": {
            "mean": np.mean(accuracies) if accuracies else 0.0,
            "std": np.std(accuracies) if accuracies else 0.0,
            "min": np.min(accuracies) if accuracies else 0.0,
            "max": np.max(accuracies) if accuracies else 0.0,
        },
        "mse": {
            "mean": np.mean(mse_values) if mse_values else 0.0,
            "std": np.std(mse_values) if mse_values else 0.0,
            "min": np.min(mse_values) if mse_values else 0.0,
            "max": np.max(mse_values) if mse_values else 0.0,
        },
        "robust_accuracy": {
            "mean": np.mean(robust_accs) if robust_accs else 0.0,
            "std": np.std(robust_accs) if robust_accs else 0.0,
            "min": np.min(robust_accs) if robust_accs else 0.0,
            "max": np.max(robust_accs) if robust_accs else 0.0,
        },
    }


# 8. Quick Test

if __name__ == "__main__":
    print("=" * 60)
    print("Utils Module Verification")
    print("=" * 60)
    
    # Test set_seed
    print("\n[1] Testing set_seed...")
    set_seed(42)
    print("  Seed set to 42")
    
    # Test device utilities
    print("\n[2] Testing device utilities...")
    device = get_device()
    print(f"  Device: {device}")
    print_device_info()
    
    # Test file utilities
    print("\n[3] Testing file utilities...")
    test_dir = "./test_outputs"
    ensure_dir(test_dir)
    print(f"  Directory created: {test_dir}")
    
    timestamp = get_timestamp()
    print(f"  Timestamp: {timestamp}")
    
    test_data = {"name": "test", "value": 42, "float_val": np.float32(3.14)}
    save_json(test_data, os.path.join(test_dir, "test.json"))
    loaded = load_json(os.path.join(test_dir, "test.json"))
    print(f"  JSON save/load: {loaded}")
    
    test_rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    save_csv(test_rows, os.path.join(test_dir, "test.csv"))
    print("  CSV saved")
    
    # Test logging utilities
    print("\n[4] Testing logging utilities...")
    print_banner("Test Banner")
    log_metrics({"accuracy": 88.50, "loss": 0.234, "f1": 0.856})
    
    # Test metric utilities
    print("\n[5] Testing metric utilities...")
    meter = AverageMeter()
    for i in range(5):
        meter.update(i * 0.5)
    print(f"  AverageMeter: {meter}")
    
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, std = compute_mean_std(values)
    print(f"  Mean/Std of {values}: {mean:.2f}, {std:.2f}")
    
    # Test model utilities
    print("\n[6] Testing model utilities...")
    import torch.nn as nn
    model = nn.Linear(10, 5)
    total, trainable = count_parameters(model)
    print(f"  Model parameters: {total:,} total, {trainable:,} trainable")
    print_model_summary(model, "LinearModel")
    
    # Test experiment utilities
    print("\n[7] Testing experiment utilities...")
    exp_id = generate_experiment_id("wesad", "asfl", "iid", 42)
    print(f"  Experiment ID: {exp_id}")
    
    result = {
        "dataset": "wesad",
        "method": "asfl",
        "distribution": "iid",
        "status": "success",
        "best_val_acc": 88.50,
        "best_mse": np.float32(0.045),
        "best_attacker": "Adaptive"
    }
    formatted = format_result_row(result)
    print(f"  Formatted result: {formatted}")
    
    summary = get_experiment_summary([result])
    print(f"  Summary: {summary}")
    
    # Clean up
    print("\n[8] Cleaning up...")
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    print("  Cleanup complete")
    
    print("\n" + "=" * 60)
    print("Utils Module: COMPLETE")
    print("=" * 60)