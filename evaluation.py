# asfl/evaluation.py

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

from config import ExperimentConfig
from attacks import build_decoder, InversionAttack, PGDAttack


# 1. Classification Metrics

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score."""
    # Convert one-hot labels to 1D if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred)


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray, 
                      average: str = 'weighted') -> float:
    """Compute precision score."""
    # Convert one-hot labels to 1D if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray, 
                   average: str = 'weighted') -> float:
    """Compute recall score."""
    # Convert one-hot labels to 1D if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, 
               average: str = 'weighted') -> float:
    """Compute F1 score."""
    # Convert one-hot labels to 1D if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """

    For multi-class, use one-vs-rest and average.
    """
    # Convert one-hot labels to 1D if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Determine number of classes from y_prob
    if y_prob.ndim > 1:
        num_classes = y_prob.shape[1]
    else:
        num_classes = 1
    
    # Handle multi-class by using one-vs-rest
    if num_classes > 2:
        auprc_values = []
        for cls in range(num_classes):
            y_true_binary = (y_true == cls).astype(int)
            # Skip classes not present in this batch
            if np.sum(y_true_binary) == 0:
                continue
            y_prob_class = y_prob[:, cls] if y_prob.ndim > 1 else y_prob
            auprc_values.append(average_precision_score(y_true_binary, y_prob_class))
        return np.mean(auprc_values) if auprc_values else 0.0
    
    # Binary classification
    if y_prob.ndim > 1 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    
    return average_precision_score(y_true, y_prob)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix."""
    # Convert one-hot labels to 1D if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return confusion_matrix(y_true, y_pred)


# 2. Classification Evaluator

class ClassificationEvaluator:
    """
    Evaluates classification performance.
    
    Supports:
    - Accuracy, Precision, Recall, F1, AUPRC
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for AUPRC)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        metrics['accuracy'] = compute_accuracy(y_true, y_pred)
        metrics['precision'] = compute_precision(y_true, y_pred)
        metrics['recall'] = compute_recall(y_true, y_pred)
        metrics['f1'] = compute_f1(y_true, y_pred)
        
        if y_prob is not None:
            metrics['auprc'] = compute_auprc(y_true, y_prob)
        else:
            metrics['auprc'] = 0.0
        
        return metrics
    
    def evaluate_clients(self, client_results: List[Dict[str, Any]],
                         aggregation_mode: str = 'all') -> Dict[str, float]:
        """
        Evaluate multiple clients with optional positive-client filtering.
        
        Args:
            client_results: List of dicts with:
                - 'y_true': list of ground truth labels
                - 'y_pred': list of predicted labels
                - 'y_prob': list of predicted probabilities (optional)
                - 'has_positive': bool indicating if client has positive samples
        
        Returns:
            Aggregated metrics computed on pooled predictions/samples
        """
        if aggregation_mode == 'positive_only':
            valid_results = [r for r in client_results if r.get('has_positive', True)]
        else:
            valid_results = client_results
        
        if not valid_results:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auprc': 0.0,
                'num_clients': 0,
                'num_samples': 0
            }
        
        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        
        for r in valid_results:
            all_y_true.extend(r['y_true'])
            all_y_pred.extend(r['y_pred'])
            if 'y_prob' in r and r['y_prob'] is not None and len(r['y_prob']) > 0:
                all_y_prob.extend(r['y_prob'])
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_prob = np.array(all_y_prob) if all_y_prob else None
        
        metrics = self.evaluate(all_y_true, all_y_pred, all_y_prob)
        metrics['num_clients'] = len(valid_results)
        metrics['num_samples'] = len(all_y_true)
        
        return metrics


# 3. Privacy Metrics (MSE Only)

def compute_mse(x: np.ndarray, reconstruction: np.ndarray) -> float:
    """
    Compute Mean Squared Error between original and reconstruction.
    
    For IoMT time-series data, MSE is used as the primary privacy metric.
    Higher MSE indicates poorer reconstruction quality and therefore
    stronger privacy protection.
    
    Note: For privacy evaluation, HIGHER MSE = BETTER privacy.
    Lower MSE indicates the attacker reconstructed the input more accurately.
    """
    return np.mean((x - reconstruction) ** 2)


# 4. Privacy Evaluator

class PrivacyEvaluator:
    """
    Evaluates privacy using reconstruction MSE.
    
    For IoMT time-series data, MSE is used as the primary privacy metric.
    Higher MSE indicates poorer reconstruction quality and therefore
    stronger privacy protection.
    
    Metrics:
    - MSE: Mean Squared Error (HIGHER is better for privacy)
    """
    
    def __init__(self, data_range: float = 1.0):
        self.data_range = data_range
    
    def evaluate(self, x: np.ndarray, reconstruction: np.ndarray) -> Dict[str, float]:
        """Evaluate privacy metrics (MSE only)."""
        mse = compute_mse(x, reconstruction)
        
        return {
            'mse': mse,
        }


# 5. Evaluate Trained Inversion Hierarchy
def evaluate_trained_inversion_hierarchy(
    cfg: ExperimentConfig,
    zu_train: torch.Tensor,
    x_train: torch.Tensor,
    zu_test: torch.Tensor,
    x_test: torch.Tensor,
    device: str = "cuda",
    attack_epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    verbose: bool = False,
    include_adaptive: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, torch.nn.Module]]:

    """
    Evaluate full inversion attack hierarchy with trained decoders.
    
    This correctly trains each decoder (GL0-GL3 + Adaptive) before evaluation.
    
    Args:
        cfg: Experiment configuration
        zu_train: Training utility projections
        x_train: Training inputs
        zu_test: Test utility projections
        x_test: Test inputs
        device: Device
        attack_epochs: Number of epochs to train each decoder
        learning_rate: Learning rate for decoder training
        batch_size: Batch size for decoder training
        verbose: Print progress
    
    Returns:
        Tuple of (metrics_dict, reconstructions_dict, trained_decoders_dict)
        
    Key metrics returned:
        - strongest_attacker: Name of the attacker with lowest MSE (best reconstruction)
        - strongest_attack_mse: Lowest MSE achieved by any attacker
        - privacy_mse: MSE against the strongest attacker (HIGHER = BETTER privacy)
        - mse_gl0, mse_gl1, mse_gl2, mse_gl3, mse_adaptive: MSE per attacker
    """
    bottleneck_dim = zu_train.shape[1]
    input_dim = cfg.get_dataset_input_dim()
    
    levels = {
        0: "GL0",
        1: "GL1",
        2: "GL2",
        3: "GL3",
        4: "Adaptive"
    }
    
    results = {}
    reconstructions = {}
    trained_decoders = {}
    
    x_test_np = x_test.cpu().numpy()

    for level, name in levels.items():
        if verbose:
            print(f"Training {name} decoder...")
    
        # Reseed before each decoder
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
    
        decoder = build_decoder(level, bottleneck_dim, input_dim).to(device)
       
        attack = InversionAttack(
            decoder=decoder,
            learning_rate=learning_rate,
            epochs=attack_epochs,
            device=device
        )
        
        attack.train(zu_train, x_train, batch_size=batch_size, verbose=verbose)
        
        with torch.no_grad():
            zu_test_t = zu_test.to(device)
            recon = attack.reconstruct(zu_test_t).cpu().numpy()
            reconstructions[name] = recon
        
        trained_decoders[name] = attack.decoder
        
        evaluator = PrivacyEvaluator()
        results[name] = evaluator.evaluate(x_test_np, recon)
    
    # Extract MSE values for each attacker
    mse_gl0 = results.get('GL0', {}).get('mse', 0.0)
    mse_gl1 = results.get('GL1', {}).get('mse', 0.0)
    mse_gl2 = results.get('GL2', {}).get('mse', 0.0)
    mse_gl3 = results.get('GL3', {}).get('mse', 0.0)
    mse_adaptive = results.get('Adaptive', {}).get('mse', 0.0)
    
    mse_values = {
        'GL0': mse_gl0,
        'GL1': mse_gl1,
        'GL2': mse_gl2,
        'GL3': mse_gl3,
        'Adaptive': mse_adaptive,
    }
    
    # Strongest attacker = lowest MSE (best reconstruction)
    strongest_attacker = min(mse_values.keys(), key=lambda k: mse_values[k])
    strongest_attack_mse = mse_values[strongest_attacker]
    
    # Privacy is evaluated against the strongest attacker.
    # privacy_mse = MSE obtained by the strongest attacker.
    # Higher values indicate better privacy.
    privacy_mse = strongest_attack_mse
    
    results['strongest_attacker'] = strongest_attacker
    results['strongest_attack_mse'] = strongest_attack_mse
    results['privacy_mse'] = privacy_mse
    
    # Keep original keys for backward compatibility
    results['best_attacker'] = strongest_attacker
    results['best_mse'] = strongest_attack_mse
    
    results['mse_gl0'] = mse_gl0
    results['mse_gl1'] = mse_gl1
    results['mse_gl2'] = mse_gl2
    results['mse_gl3'] = mse_gl3
    results['mse_adaptive'] = mse_adaptive
    
    return results, reconstructions, trained_decoders


# 6. Robustness Metrics (PGD Attack)

def compute_attack_success_rate(y_true: np.ndarray, clean_pred: np.ndarray,
                                adv_pred: np.ndarray) -> float:
    """
    Compute Attack Success Rate.
    
    ASR = (correctly classified clean but misclassified adversarial) / (correctly classified clean)
    """
    # Convert one-hot labels to 1D if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if clean_pred.ndim > 1 and clean_pred.shape[1] > 1:
        clean_pred = np.argmax(clean_pred, axis=1)
    if adv_pred.ndim > 1 and adv_pred.shape[1] > 1:
        adv_pred = np.argmax(adv_pred, axis=1)
    
    correctly_classified = (clean_pred == y_true)
    misclassified_adv = (adv_pred != y_true)
    
    if np.sum(correctly_classified) == 0:
        return 0.0
    
    asr = np.sum(misclassified_adv & correctly_classified) / np.sum(correctly_classified)
    return asr


# 7. Robustness Evaluator

class RobustnessEvaluator:
    """
    Evaluates robustness against adversarial attacks (PGD).
    """
    
    def __init__(self, model: Optional[torch.nn.Module] = None,
                 epsilon: float = 0.3, steps: int = 40, 
                 alpha: float = 0.01, device: str = "cuda",
                 class_weights: Optional[torch.Tensor] = None):
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.class_weights = class_weights
        self.pgd_attack = None
        
        if model is not None:
            self.pgd_attack = PGDAttack(
                model=model,
                epsilon=epsilon,
                steps=steps,
                alpha=alpha,
                device=device,
                class_weights=class_weights
            )
    
    def evaluate(self, model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                 device: Optional[str] = None) -> Dict[str, float]:
        if device is None:
            device = self.device
        
        self.model = model
        
        self.pgd_attack = PGDAttack(
            model=model,
            epsilon=self.epsilon,
            steps=self.steps,
            alpha=self.alpha,
            device=device,
            class_weights=self.class_weights
        )
        
        model.eval()
        
        clean_correct = 0
        adv_correct = 0
        total = 0
        all_clean_preds = []
        all_adv_preds = []
        all_labels = []
        
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # Convert one-hot labels to class indices if needed
            if y.dim() > 1 and y.shape[1] > 1:
                y = y.argmax(dim=1)
            
            with torch.no_grad():
                outputs = model(x)
                clean_preds = outputs["logits"].argmax(dim=1)
            
            x_adv, _ = self.pgd_attack.attack(x, y)
            with torch.no_grad():
                outputs = model(x_adv)
                adv_preds = outputs["logits"].argmax(dim=1)
            
            clean_correct += (clean_preds == y).sum().item()
            adv_correct += (adv_preds == y).sum().item()
            total += x.size(0)
            
            all_clean_preds.extend(clean_preds.cpu().numpy())
            all_adv_preds.extend(adv_preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        clean_acc = 100.0 * clean_correct / total
        adv_acc = 100.0 * adv_correct / total
        
        all_clean_preds = np.array(all_clean_preds)
        all_adv_preds = np.array(all_adv_preds)
        all_labels = np.array(all_labels)
        
        asr = compute_attack_success_rate(all_labels, all_clean_preds, all_adv_preds)
        
        return {
            'clean_accuracy': clean_acc,
            'pgd_accuracy': adv_acc,
            'attack_success_rate': asr,
        }


# 8. Full Classification Evaluation (returns predictions)

def evaluate_classification_full(model: torch.nn.Module, 
                                 loader: torch.utils.data.DataLoader,
                                 device: str = "cuda") -> Dict[str, Any]:
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # Convert one-hot labels to class indices if needed
            if y.dim() > 1 and y.shape[1] > 1:
                y = y.argmax(dim=1)
            
            outputs = model(x)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    evaluator = ClassificationEvaluator()
    metrics = evaluator.evaluate(all_labels, all_preds, all_probs)
    
    has_positive = np.sum(all_labels == 1) > 0
    
    return {
        'metrics': metrics,
        'y_true': all_labels.tolist(),
        'y_pred': all_preds.tolist(),
        'y_prob': all_probs.tolist(),
        'has_positive': has_positive,
    }


# 9. Results Saving

def save_evaluation_results(results: Dict[str, Any], output_dir: str, 
                            name: str = "evaluation") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{name}.json")
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, torch.nn.Module):
            return str(type(obj).__name__)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {
                k: convert_to_serializable(v) for k, v in value.items()
            }
        else:
            serializable_results[key] = convert_to_serializable(value)
    
    with open(filepath, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    return filepath


# 10. Phase 6 Verification

def verify_evaluation():
    print("=" * 60)
    print("Phase 6: Evaluation Verification")
    print("=" * 60)
    
    print("\n[1] Classification Metrics:")
    print("-" * 40)
    
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.6, 0.9, 0.8, 0.7])
    
    print(f"  Accuracy: {compute_accuracy(y_true, y_pred):.4f}")
    print(f"  Precision: {compute_precision(y_true, y_pred):.4f}")
    print(f"  Recall: {compute_recall(y_true, y_pred):.4f}")
    print(f"  F1: {compute_f1(y_true, y_pred):.4f}")
    print(f"  AUPRC: {compute_auprc(y_true, y_prob):.4f}")
    
    print("\n[2] Privacy Metrics (MSE Only):")
    print("-" * 40)
    
    x = np.random.randn(100)
    recon = x + 0.1 * np.random.randn(100)
    
    print(f"  MSE: {compute_mse(x, recon):.6f}")
    
    
    client_results = [
        {'client_id': 0, 'has_positive': True, 
         'y_true': [0, 1, 0, 1], 'y_pred': [0, 1, 0, 1]},
        {'client_id': 1, 'has_positive': True,
         'y_true': [0, 1, 0, 1], 'y_pred': [0, 0, 0, 1]},
        {'client_id': 2, 'has_positive': False,
         'y_true': [0, 0, 0, 0], 'y_pred': [0, 0, 0, 0]},
    ]
    
    classifier = ClassificationEvaluator()
    result = classifier.evaluate_clients(client_results, aggregation_mode='positive_only')
    print(f"  Positive clients only: {result['num_clients']} clients")
    print(f"  Samples: {result['num_samples']}")
    print(f"  Aggregated accuracy: {result['accuracy']:.4f}")
    
    print("\n[4] Attack Success Rate:")
    print("-" * 40)
    
    clean_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    adv_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1])
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    
    asr = compute_attack_success_rate(y_true, clean_pred, adv_pred)
    print(f"  ASR: {asr:.4f}")
    
    print("\n" + "=" * 60)
    print("Phase 6: COMPLETE")
    print("=" * 60)


# 11. Robustness Evaluation with Configurable PGD Parameters

def evaluate_robustness_with_pgd(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    epsilon: float = 0.3,
    steps: int = 40,
    alpha: float = 0.01,
    device: str = "cuda",
    class_weights: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Evaluate robustness with configurable PGD parameters.
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        epsilon: PGD perturbation bound
        steps: Number of PGD steps
        alpha: PGD step size
        device: Device
        class_weights: Optional class weights for loss
    
    Returns:
        Dictionary with clean_acc, clean_f1, adv_acc, adv_f1
    """
    
    pgd = PGDAttack(
        model=model,
        epsilon=epsilon,
        steps=steps,
        alpha=alpha,
        device=device,
        class_weights=class_weights
    )
    
    return pgd.compute_robust_metrics(model, loader, attack=True)


if __name__ == "__main__":
    verify_evaluation()