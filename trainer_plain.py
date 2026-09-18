# asfl/trainer_plain.py

import copy
import os
import json
import csv
import random
import torch
import torch.nn as nn
import numpy as np
from experiments import register_trainer
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader

from config import ExperimentConfig, save_config
from utils import convert_to_serializable
from models import build_plain_model
from evaluation import (
    compute_accuracy, compute_precision, compute_recall, compute_f1, compute_auprc,
    compute_mse,
    evaluate_trained_inversion_hierarchy
)
from attacks import PGDAttack

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class TemporaryPlainModelForPGD(nn.Module):
    """
    Wrapper for Plain SFL model to support PGD attack.
    Matches PlainSFLTrainer._client_forward() which returns a single tensor (smashed data).
    """
    def __init__(self, backbone, server_classifier):
        super().__init__()
        self.backbone = backbone
        self.server_classifier = server_classifier

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        smashed = self.backbone(x)
        logits = self.server_classifier(smashed)
        return {"logits": logits}

@register_trainer('plain')
class PlainSFLTrainer:
    """
    Plain Split Federated Learning trainer.
    Implements the SplitFed-V2 communication protocol:
    1. Each client has its own backbone
    2. Server has shared classifier
    3. Client forward pass -> smashed data
    4. Server forward pass on smashed data -> logits
    5. Server backward pass -> gradients
    6. Client backward pass using server gradients
    7. Federated averaging of client backbones
    """

    def __init__(self, cfg: ExperimentConfig, train_loaders: List[DataLoader],
                 val_loaders: List[DataLoader], device: str = "cuda"):
        """
        Args:
            cfg: Experiment configuration
            train_loaders: List of training DataLoaders per client
            val_loaders: List of validation DataLoaders per client
            device: Device to use ('cuda' or 'cpu')
        """
        set_seed(cfg.seed)

        self.cfg = cfg
        self.device = device if torch.cuda.is_available() else "cpu"
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.num_clients = len(train_loaders)

        self.num_classes = cfg.get_dataset_num_classes()

        # Identify valid clients (non-empty datasets)
        self.valid_train_clients = [
            i for i in range(self.num_clients)
            if len(self.train_loaders[i].dataset) > 0
        ]
        self.valid_val_clients = [
            i for i in range(self.num_clients)
            if len(self.val_loaders[i].dataset) > 0
        ]

        # Build one model to get architecture
        template_model = build_plain_model(cfg).to(self.device)

        # Create separate backbones for each client
        self.client_backbones = []
        for _ in range(self.num_clients):
            backbone = copy.deepcopy(template_model.backbone).to(self.device)
            self.client_backbones.append(backbone)

        # Shared server classifier (one global copy)
        self.server_classifier = copy.deepcopy(template_model.classifier).to(self.device)

        # Client optimizers (one per client backbone)
        self.client_optimizers = []
        for backbone in self.client_backbones:
            opt = torch.optim.Adam(
                backbone.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
            self.client_optimizers.append(opt)

        # Server optimizer
        self.server_optimizer = torch.optim.Adam(
            self.server_classifier.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )

        # Initialize default criterion first
        self.criterion = nn.CrossEntropyLoss()
        self.class_weights = None
        self.class_counts = None

        if getattr(cfg, 'use_class_weights', False):
            all_labels = []
            for loader in train_loaders:
                for _, y in loader:
                    all_labels.extend(y.tolist())

            if len(all_labels) > 0:
                class_counts = np.bincount(all_labels, minlength=self.num_classes)
                total = len(all_labels)
                n_classes = len(class_counts)
                class_weights = []
                for count in class_counts:
                    if count > 0:
                        weight = total / (n_classes * count)
                        class_weights.append(weight)
                    else:
                        class_weights.append(1.0)

                self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
                self.class_counts = class_counts
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                print(f" Using class-weighted CE: {self.class_weights}")
                print(f" Class counts: {self.class_counts}")
            else:
                self.class_weights = None
                self.class_counts = None
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.class_weights = None
            self.class_counts = None
            self.criterion = nn.CrossEntropyLoss()

        # History for tracking metrics with F1 and AUPRC
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1": [],
            "train_auprc": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_auprc": [],
            "val_evasion_f1": [],
            "val_privacy_mse": [],
        }

        # CSV log
        self.csv_log = []
        self._csv_row_count = 0

        # Track best validation accuracy
        self.best_val_acc = 0.0
        self.best_epoch = 0

        # Track best validation F1
        self.best_val_f1 = 0.0
        self.best_f1_epoch = 0

        # Final metrics for paper reporting
        self.final_metrics = {
            "final_loss": 0.0,
            "final_acc": 0.0,
            "final_f1": 0.0,
            "final_privacy_mse": 0.0,
            "final_evasion_f1": 0.0,
        }

        # Save config at initialization
        save_config(cfg)

    def _get_client_weights(self, client_indices: List[int]) -> List[float]:
        """
        Compute weights based on dataset sizes for federated averaging.
        Args:
            client_indices: List of client indices to average
        Returns:
            List of weights for each client
        """
        sample_sizes = [len(self.train_loaders[i].dataset) for i in client_indices]
        total_samples = sum(sample_sizes)
        return [n / total_samples for n in sample_sizes]

    def federated_average(self, state_dicts: List[Dict[str, torch.Tensor]],
                          client_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Weighted Federated Averaging of client backbone state dicts.
        Uses weighted averaging based on client dataset sizes for fair comparison
        under non-IID partitioning.
        
        Args:
            state_dicts: List of state dicts to average
            client_indices: List of client indices (for weighting)
        
        Returns:
            Averaged state dict
        """
        if not state_dicts:
            return {}

        if client_indices is None:
            client_indices = list(range(len(state_dicts)))

        # Compute weights based on dataset sizes
        weights = self._get_client_weights(client_indices)

        # Weighted average with explicit accumulation
        avg_state = {}
        for key in state_dicts[0].keys():
            avg_state[key] = weights[0] * state_dicts[0][key]
            for i in range(1, len(state_dicts)):
                avg_state[key] += weights[i] * state_dicts[i][key]

        return avg_state

    def _server_forward(self, smashed_data: torch.Tensor) -> torch.Tensor:
        """Server forward pass on smashed data."""
        return self.server_classifier(smashed_data)

    def _server_backward(self, smashed_data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Server backward pass.
        Returns:
            gradients to send back to client, loss value
        """
        self.server_optimizer.zero_grad()
        labels = labels.to(self.device)

        # Forward pass
        logits = self._server_forward(smashed_data)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Ensure gradients exist
        assert smashed_data.grad is not None, "No gradients computed for smashed data"

        # Get gradients of smashed data
        grad_smashed = smashed_data.grad.clone().detach()

        # Update server
        self.server_optimizer.step()

        return grad_smashed, loss.item()

    def _client_forward(self, client_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Client forward pass: x -> smashed data."""
        return self.client_backbones[client_idx](x)

    def _client_backward(self, client_idx: int, client_output: torch.Tensor,
                         grad_smashed: torch.Tensor) -> None:
        """
        Client backward pass using server gradients.
        Args:
            client_idx: Client index
            client_output: Original client output (with computational graph)
            grad_smashed: Gradients from server
        """
        self.client_optimizers[client_idx].zero_grad()
        client_output.backward(grad_smashed)
        self.client_optimizers[client_idx].step()

    def _generate_pgd_attack(self, client_idx: int, x: torch.Tensor, y: torch.Tensor,
                             eval_mode: bool = False) -> torch.Tensor:
        """
        Generate PGD adversarial examples.
        Args:
            client_idx: Client index
            x: Input batch
            y: Labels
            eval_mode: If True, use pgd_eval_steps; otherwise use pgd_steps
        """
        temp_model = TemporaryPlainModelForPGD(
            backbone=self.client_backbones[client_idx],
            server_classifier=self.server_classifier
        ).to(self.device)

        # Always use untargeted attack for all datasets
        steps = self.cfg.pgd_eval_steps if eval_mode else self.cfg.pgd_steps

        pgd = PGDAttack(
            model=temp_model,
            epsilon=self.cfg.pgd_epsilon,
            steps=steps,
            alpha=self.cfg.pgd_alpha,
            device=self.device,
            class_weights=None   
        )

        # Always untargeted
        x_adv, _ = pgd.attack(x, y, targeted=False)
        return x_adv

    def train_client(self, client_idx: int, local_epochs: int = 1) -> Dict[str, torch.Tensor]:
        """
        Train one client for local_epochs.
        Returns:
            client backbone state dict after training
        """
        if len(self.train_loaders[client_idx].dataset) == 0:
            return self.client_backbones[client_idx].state_dict()

        loader = self.train_loaders[client_idx]
        self.client_backbones[client_idx].train()
        self.server_classifier.train()

        for epoch in range(local_epochs):
            for batch_idx, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # Client forward: x -> client_output 
                client_output = self._client_forward(client_idx, x)

                server_input = client_output.detach().requires_grad_(True)

                # Server forward/backward
                grad_smashed, loss = self._server_backward(server_input, y)

                # Client backward using original client_output
                self._client_backward(client_idx, client_output, grad_smashed)

        return self.client_backbones[client_idx].state_dict()

    def evaluate_client(self, client_idx: int, loader: DataLoader,
                        compute_metrics: bool = False) -> Tuple:
        """
        Evaluate one client on given dataloader.
        Returns:
            (avg_loss, avg_accuracy, metrics_dict)
        """
        if len(loader.dataset) == 0:
            return 0.0, 0.0, {}

        self.client_backbones[client_idx].eval()
        self.server_classifier.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                # Client forward
                smashed_data = self._client_forward(client_idx, x)

                # Server forward
                logits = self._server_forward(smashed_data)

                loss = self.criterion(logits, y)
                total_loss += loss.item() * x.size(0)

                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0)

                if compute_metrics:
                    all_y_true.extend(y.cpu().numpy())
                    all_y_pred.extend(preds.cpu().numpy())
                    all_y_prob.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())

        if total_samples == 0:
            return 0.0, 0.0, {}

        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples

        metrics = {}
        if compute_metrics and len(all_y_true) > 0:
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_y_prob = np.array(all_y_prob) if all_y_prob else None

            metrics['accuracy'] = compute_accuracy(all_y_true, all_y_pred)
            metrics['precision'] = compute_precision(all_y_true, all_y_pred, average='weighted')
            metrics['recall'] = compute_recall(all_y_true, all_y_pred, average='weighted')
            metrics['f1'] = compute_f1(all_y_true, all_y_pred, average='weighted')
            if all_y_prob is not None:
                metrics['auprc'] = compute_auprc(all_y_true, all_y_prob)
            else:
                metrics['auprc'] = 0.0

        return avg_loss, avg_acc, metrics

    def evaluate_client_with_metrics(self, client_idx: int, loader: DataLoader,
                                     compute_robustness: bool = False,
                                     compute_metrics: bool = False) -> Tuple:
        """
        Evaluate one client with PGD attack metrics.
        Returns:
            (avg_loss, avg_acc, metrics_dict, adv_metrics_dict, evasion_f1)
        """
        if len(loader.dataset) == 0:
            return 0.0, 0.0, {}, {}, 0.0

        self.client_backbones[client_idx].eval()
        self.server_classifier.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        all_adv_y_true = []
        all_adv_y_pred = []
        all_adv_y_prob = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                # Client forward
                smashed_data = self._client_forward(client_idx, x)

                # Server forward
                logits = self._server_forward(smashed_data)

                loss = self.criterion(logits, y)
                total_loss += loss.item() * x.size(0)

                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0)

                if compute_metrics:
                    all_y_true.extend(y.cpu().numpy())
                    all_y_pred.extend(preds.cpu().numpy())
                    all_y_prob.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())

        evasion_f1 = 0.0

        # Only run PGD evaluation if compute_robustness is True
        if compute_robustness:
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                # PGD attack with eval_mode=True
                x_adv = self._generate_pgd_attack(client_idx, x, y, eval_mode=True)
                adv_smashed = self._client_forward(client_idx, x_adv)
                adv_logits = self._server_forward(adv_smashed)
                adv_preds = adv_logits.argmax(dim=1)

                all_adv_y_true.extend(y.cpu().numpy())
                all_adv_y_pred.extend(adv_preds.cpu().numpy())
                all_adv_y_prob.extend(torch.softmax(adv_logits, dim=1).detach().cpu().numpy())

            # Compute weighted F1 for adversarial predictions
            if len(all_adv_y_true) > 0:
                all_adv_y_true = np.array(all_adv_y_true)
                all_adv_y_pred = np.array(all_adv_y_pred)
                evasion_f1 = compute_f1(all_adv_y_true, all_adv_y_pred, average='weighted')

        if total_samples == 0:
            return 0.0, 0.0, {}, {}, 0.0

        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples

        # Clean metrics (only if compute_metrics is True)
        metrics = {}
        if compute_metrics and len(all_y_true) > 0:
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_y_prob = np.array(all_y_prob) if all_y_prob else None

            metrics['accuracy'] = compute_accuracy(all_y_true, all_y_pred)
            metrics['precision'] = compute_precision(all_y_true, all_y_pred, average='weighted')
            metrics['recall'] = compute_recall(all_y_true, all_y_pred, average='weighted')
            metrics['f1'] = compute_f1(all_y_true, all_y_pred, average='weighted')
            if all_y_prob is not None:
                metrics['auprc'] = compute_auprc(all_y_true, all_y_prob)
            else:
                metrics['auprc'] = 0.0

        # Adversarial metrics (F1 only, for consistency)
        adv_metrics = {
            'f1': evasion_f1,
        }

        return avg_loss, avg_acc, metrics, adv_metrics, evasion_f1

    def evaluate_inversion_attack(self) -> Dict[str, Any]:
        """
        Evaluate inversion attack for Plain SFL.
        Uses validation set for fair comparison with ASFL and ResSFL.
        """
        all_zu = []
        all_x = []

        for client_idx in self.valid_val_clients:
            loader = self.val_loaders[client_idx]
            self.client_backbones[client_idx].eval()

            with torch.no_grad():
                for x, y in loader:
                    x = x.to(self.device)
                    # In Plain SFL, zu is the smashed data (no bottleneck projection)
                    zu = self._client_forward(client_idx, x)
                    all_zu.append(zu.cpu())
                    all_x.append(x.cpu())

        if not all_zu:
            return {
                "mse_gl0": 0.0,
                "mse_gl1": 0.0,
                "mse_gl2": 0.0,
                "mse_gl3": 0.0,
                "mse_adaptive": 0.0,
                "privacy_mse": 0.0,
                "strongest_attack_mse": 0.0,
                "strongest_attacker": "None",
                "inversion_available": False
            }

        zu = torch.cat(all_zu, dim=0)
        x = torch.cat(all_x, dim=0)

        # Split into train/val for decoder training
        split_idx = int(0.8 * len(zu))
        zu_train = zu[:split_idx]
        x_train = x[:split_idx]
        zu_test = zu[split_idx:]
        x_test = x[split_idx:]

        # Evaluate full inversion hierarchy
        results, _, _ = evaluate_trained_inversion_hierarchy(
            cfg=self.cfg,
            zu_train=zu_train,
            x_train=x_train,
            zu_test=zu_test,
            x_test=x_test,
            device=self.device,
            attack_epochs=50,
            learning_rate=1e-3,
            batch_size=64,
            verbose=False, include_adaptive=True
        )

        results["inversion_available"] = True
        return results

    def train_round(self, round_idx: int, compute_robustness: bool = False) -> Dict[str, float]:
        """
        One training round: all clients train locally, then federated averaging.
        Returns:
            metrics dict for this round
        """
        client_state_dicts = []
        client_indices = []
        client_train_losses = []
        client_train_accs = []
        client_train_precisions = []
        client_train_recalls = []
        client_train_f1s = []
        client_train_auprcs = []

        # Train each valid client
        for client_idx in self.valid_train_clients:
            # Train client
            state_dict = self.train_client(client_idx, local_epochs=1)
            client_state_dicts.append(state_dict)
            client_indices.append(client_idx)

            # Evaluate client on its training set
            loss, acc, metrics = self.evaluate_client(
                client_idx, self.train_loaders[client_idx], compute_metrics=True
            )
            client_train_losses.append(loss)
            client_train_accs.append(acc)
            if metrics:
                client_train_precisions.append(metrics.get('precision', 0.0))
                client_train_recalls.append(metrics.get('recall', 0.0))
                client_train_f1s.append(metrics.get('f1', 0.0))
                client_train_auprcs.append(metrics.get('auprc', 0.0))

        # Federated averaging of client backbones (only over valid clients)
        if client_state_dicts:
            avg_state = self.federated_average(client_state_dicts, client_indices)

            # Load averaged weights into all client backbones
            for client_idx in range(self.num_clients):
                self.client_backbones[client_idx].load_state_dict(avg_state)

        # Evaluate all valid clients on validation sets
        val_losses = []
        val_accs = []
        val_precisions = []
        val_recalls = []
        val_f1s = []
        val_auprcs = []
        val_evasion_f1s = []

        for client_idx in self.valid_val_clients:
            loss, acc, metrics, adv_metrics, evasion_f1 = self.evaluate_client_with_metrics(
                client_idx, self.val_loaders[client_idx],
                compute_robustness=compute_robustness,
                compute_metrics=True
            )
            val_losses.append(loss)
            val_accs.append(acc)
            val_evasion_f1s.append(evasion_f1)
            if metrics:
                val_precisions.append(metrics.get('precision', 0.0))
                val_recalls.append(metrics.get('recall', 0.0))
                val_f1s.append(metrics.get('f1', 0.0))
                val_auprcs.append(metrics.get('auprc', 0.0))

        # Aggregate metrics over valid clients only
        round_metrics = {
            "train_loss": np.mean(client_train_losses) if client_train_losses else 0.0,
            "train_acc": np.mean(client_train_accs) if client_train_accs else 0.0,
            "train_precision": np.mean(client_train_precisions) if client_train_precisions else 0.0,
            "train_recall": np.mean(client_train_recalls) if client_train_recalls else 0.0,
            "train_f1": np.mean(client_train_f1s) if client_train_f1s else 0.0,
            "train_auprc": np.mean(client_train_auprcs) if client_train_auprcs else 0.0,
            "val_loss": np.mean(val_losses) if val_losses else 0.0,
            "val_acc": np.mean(val_accs) if val_accs else 0.0,
            "val_precision": np.mean(val_precisions) if val_precisions else 0.0,
            "val_recall": np.mean(val_recalls) if val_recalls else 0.0,
            "val_f1": np.mean(val_f1s) if val_f1s else 0.0,
            "val_auprc": np.mean(val_auprcs) if val_auprcs else 0.0,
            "val_evasion_f1": np.mean(val_evasion_f1s) if val_evasion_f1s else 0.0,
        }

        return round_metrics

    def _save_csv_log(self) -> None:
        """
        Save the complete training log to CSV with timestamp.
        """
        import time
        output_dir = self.cfg.get_output_path()
        
        # Add timestamp to filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, "train_log.csv")
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_acc", "train_precision", "train_recall", "train_f1", "train_auprc",
                "val_loss", "val_acc", "val_precision", "val_recall", "val_f1", "val_auprc",
                "train_rec_loss", "val_rec_loss",
                "val_mse_gl0", "val_mse_gl1", "val_mse_gl2", "val_mse_gl3", "val_mse_adaptive",
                "val_privacy_mse", "val_strongest_attack_mse", "val_strongest_attacker",
                "privacy_utility",
                "val_evasion_f1",
                "best_val_acc", "best_val_f1", "best_privacy_mse", "best_privacy_utility",
                "train_grad_norm"
            ])
            if self.csv_log:
                writer.writerows(self.csv_log)

    def fit(self) -> Dict[str, List[float]]:
        """
        Run full training for main_epochs rounds.
        Returns:
            history dictionary with metrics
        """
        main_epochs = self.cfg.main_epochs

        print("=" * 60)
        print(f"Plain SFL Training on {self.cfg.dataset}")
        print(f" Clients: {self.num_clients}")
        print(f" Valid train clients: {len(self.valid_train_clients)}")
        print(f" Valid val clients: {len(self.valid_val_clients)}")
        print(f" Main epochs: {main_epochs}")
        print(f" Device: {self.device}")
        print("=" * 60)

        for epoch_idx in range(1, main_epochs + 1):
            compute_robustness = (epoch_idx % 10 == 0) or (epoch_idx == main_epochs)

            metrics = self.train_round(epoch_idx, compute_robustness=compute_robustness)

            if (epoch_idx % 10 == 0) or (epoch_idx == main_epochs):
                inv_results = self.evaluate_inversion_attack()
                val_privacy_mse = inv_results.get("privacy_mse", 0.0)
            else:
                # Carry forward last known value instead of resetting to 0
                val_privacy_mse = (
                    self.history["val_privacy_mse"][-1]
                    if self.history["val_privacy_mse"]
                    else 0.0
                )

            # Track best validation accuracy
            if metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = metrics["val_acc"]
                self.best_epoch = epoch_idx
                self.save_checkpoint("best")

            # Track best validation F1
            val_f1 = metrics.get("val_f1", 0.0)
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_f1_epoch = epoch_idx
                self.save_checkpoint("best_f1")

            # Store history
            self.history["train_loss"].append(metrics["train_loss"])
            self.history["train_acc"].append(metrics["train_acc"])
            self.history["train_precision"].append(metrics.get("train_precision", 0.0))
            self.history["train_recall"].append(metrics.get("train_recall", 0.0))
            self.history["train_f1"].append(metrics.get("train_f1", 0.0))
            self.history["train_auprc"].append(metrics.get("train_auprc", 0.0))
            self.history["val_loss"].append(metrics["val_loss"])
            self.history["val_acc"].append(metrics["val_acc"])
            self.history["val_precision"].append(metrics.get("val_precision", 0.0))
            self.history["val_recall"].append(metrics.get("val_recall", 0.0))
            self.history["val_f1"].append(val_f1)
            self.history["val_auprc"].append(metrics.get("val_auprc", 0.0))
            self.history["val_evasion_f1"].append(metrics.get("val_evasion_f1", 0.0))
            self.history["val_privacy_mse"].append(val_privacy_mse)

            # Append to CSV log with updated best values
            self.csv_log.append([
                epoch_idx,
                metrics["train_loss"],
                metrics["train_acc"],
                metrics.get("train_precision", 0.0),
                metrics.get("train_recall", 0.0),
                metrics.get("train_f1", 0.0),
                metrics.get("train_auprc", 0.0),
                metrics["val_loss"],
                metrics["val_acc"],
                metrics.get("val_precision", 0.0),
                metrics.get("val_recall", 0.0),
                val_f1,
                metrics.get("val_auprc", 0.0),
                metrics.get("val_evasion_f1", 0.0),
                val_privacy_mse,
                self.best_val_acc,
                self.best_val_f1
            ])

            # Save last checkpoint every round
            self.save_checkpoint("last")

            # Save periodic checkpoint every save_interval epochs
            if epoch_idx % self.cfg.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch_idx}")

            # Save CSV log every round (in append mode)
            self._save_csv_log()

            # Print progress
            if epoch_idx % self.cfg.log_interval == 0:
                print(f"Epoch {epoch_idx}/{main_epochs}: "
                      f"Train Acc: {metrics['train_acc']:.2f}% | "
                      f"Val Acc: {metrics['val_acc']:.2f}% | "
                      f"Val F1: {val_f1:.4f} | "
                      f"Val Evasion F1: {metrics.get('val_evasion_f1', 0.0):.4f} | "
                      f"Privacy MSE: {val_privacy_mse:.4f} | "
                      f"Best Acc: {self.best_val_acc:.2f}% | "
                      f"Best F1: {self.best_val_f1:.4f}")

        # Final inversion attack evaluation for accurate final privacy metric
        final_inv_results = self.evaluate_inversion_attack()

        # Store final metrics before saving
        if len(self.history["val_loss"]) > 0:
            self.final_metrics["final_loss"] = self.history["val_loss"][-1]
            self.final_metrics["final_acc"] = self.history["val_acc"][-1]
            self.final_metrics["final_f1"] = self.history["val_f1"][-1]
            # Use final inversion evaluation for accurate privacy metric
            self.final_metrics["final_privacy_mse"] = final_inv_results.get("privacy_mse", 0.0)
            self.final_metrics["final_evasion_f1"] = self.history["val_evasion_f1"][-1]

        # Save final checkpoint and logs
        self.save_checkpoint("final")
        self.save_metrics()
        self._save_csv_log()

        print("\n" + "=" * 60)
        print("Final Inversion Attack Evaluation")
        print("=" * 60)
        print(f" Privacy MSE: {final_inv_results.get('privacy_mse', 0.0):.4f}")
        print(f" Strongest Attacker: {final_inv_results.get('strongest_attacker', 'None')}")

        print("\n" + "=" * 60)
        print("Plain SFL Training Complete!")
        print(f" Main epochs: {main_epochs}")
        print(f" Best validation accuracy: {self.best_val_acc:.2f}% (main epoch {self.best_epoch})")
        print(f" Best validation F1: {self.best_val_f1:.4f} (main epoch {self.best_f1_epoch})")
        print(f" Final epoch F1 (reported): {self.final_metrics['final_f1']:.4f}")
        print(f" Final epoch Privacy MSE (reported): {self.final_metrics['final_privacy_mse']:.4f}")
        print(f" Final epoch Evasion F1 (reported): {self.final_metrics['final_evasion_f1']:.4f}")
        print("=" * 60)

        return self.history

    def save_checkpoint(self, name: str = "last") -> None:
        """
        Save checkpoint including all client backbones, server classifier,
        and optimizer states.
        """
        checkpoint = {
            "main_epoch": len(self.history["train_loss"]),
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "best_val_f1": self.best_val_f1,
            "best_f1_epoch": self.best_f1_epoch,
            "final_metrics": self.final_metrics,
            "server_classifier": self.server_classifier.state_dict(),
            "server_optimizer": self.server_optimizer.state_dict(),
            "client_backbones": [
                backbone.state_dict() for backbone in self.client_backbones
            ],
            "client_optimizers": [
                opt.state_dict() for opt in self.client_optimizers
            ],
            "history": self.history,
            "cfg": self.cfg.to_dict(),
        }

        output_dir = self.cfg.get_output_path()
        checkpoint_path = os.path.join(output_dir, f"checkpoint_plain_{name}.pth")
        torch.save(checkpoint, checkpoint_path)

    def save_metrics(self) -> None:
        """Save final metrics to JSON file with timestamp."""
        import time
        output_dir = self.cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(output_dir, f"metrics_plain_{timestamp}.json")

        # Get final inversion attack results
        final_inv_results = self.evaluate_inversion_attack()

        metrics = {
            "history": self.history,
            "best_val_acc": float(self.best_val_acc) if isinstance(self.best_val_acc, (np.float32, np.float64)) else self.best_val_acc,
            "best_epoch": int(self.best_epoch) if isinstance(self.best_epoch, (np.int32, np.int64)) else self.best_epoch,
            "best_val_f1": float(self.best_val_f1) if isinstance(self.best_val_f1, (np.float32, np.float64)) else self.best_val_f1,
            "best_f1_epoch": int(self.best_f1_epoch) if isinstance(self.best_f1_epoch, (np.int32, np.int64)) else self.best_f1_epoch,
            "final_metrics": {
                "final_loss": float(self.final_metrics.get("final_loss", 0.0)),
                "final_acc": float(self.final_metrics.get("final_acc", 0.0)),
                "final_f1": float(self.final_metrics.get("final_f1", 0.0)),
                "final_privacy_mse": float(self.final_metrics.get("final_privacy_mse", 0.0)),
                "final_evasion_f1": float(self.final_metrics.get("final_evasion_f1", 0.0)),
            },
            "final_privacy_metrics": {
                "mse_gl0": float(final_inv_results.get("mse_gl0", 0.0)),
                "mse_gl1": float(final_inv_results.get("mse_gl1", 0.0)),
                "mse_gl2": float(final_inv_results.get("mse_gl2", 0.0)),
                "mse_gl3": float(final_inv_results.get("mse_gl3", 0.0)),
                "mse_adaptive": float(final_inv_results.get("mse_adaptive", 0.0)),
                "privacy_mse": float(final_inv_results.get("privacy_mse", 0.0)),
                "strongest_attack_mse": float(final_inv_results.get("strongest_attack_mse", 0.0)),
                "strongest_attacker": final_inv_results.get("strongest_attacker", "None"),
            },
            "cfg": self.cfg.to_dict(),
            "main_epochs": int(self.cfg.main_epochs),
        }

        with open(metrics_path, "w") as f:
            json.dump(convert_to_serializable(metrics), f, indent=2)

        print(f"Metrics saved: {metrics_path}")

    def get_metrics(self) -> Dict[str, List[float]]:
        """Return training history."""
        return self.history


# Factory function for Plain SFL
def create_plain_trainer(cfg: ExperimentConfig,
                         train_loaders: List[DataLoader],
                         val_loaders: List[DataLoader]) -> PlainSFLTrainer:
    """Factory for creating PlainSFLTrainer from config and dataloaders."""
    return PlainSFLTrainer(cfg, train_loaders, val_loaders, cfg.device)