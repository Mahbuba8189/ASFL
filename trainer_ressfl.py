# asfl/trainer_ressfl.py

import copy
import os
import json
import csv
import random
import torch
import torch.nn as nn
import numpy as np
import traceback
from experiments import register_trainer
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader, Subset, TensorDataset
from experiments import ExperimentRunner

from config import ExperimentConfig, save_config
from utils import convert_to_serializable
from models import build_ressfl_model
from attacks import build_decoder, InversionAttack, PGDAttack, get_adversary_name
from evaluation import (
    compute_accuracy, compute_precision, compute_recall, compute_f1, compute_auprc,
    compute_mse,
    ClassificationEvaluator, PrivacyEvaluator,
    evaluate_trained_inversion_hierarchy, save_evaluation_results
)

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

class TemporaryResSFLModelForPGD(nn.Module):
    """
    Wrapper for ResSFL model to support PGD attack.
    Matches ResSFLTrainer._client_forward() which returns (zu, h).
    """
    def __init__(self, backbone, bottleneck, server_classifier):
        super().__init__()
        self.backbone = backbone
        self.bottleneck = bottleneck
        self.server_classifier = server_classifier

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        zu = self.bottleneck(h)
        logits = self.server_classifier(zu)
        return {"logits": logits}


@register_trainer('ressfl')
class ResSFLTrainer:
    """
    Adapted from: Li et al., "ResSFL: A Resistance Transfer Framework for Defending
    Model Inversion Attack in Split Federated Learning," CVPR 2022.
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
        self.verbose = True

        # ResSFL-specific hyperparameters (from original paper, adapted for IoMT)
        self.pretrain_epochs = getattr(cfg, 'ressfl_pretrain_epochs', 50)
        self.warmup_epochs = getattr(cfg, 'ressfl_warmup_epochs', 10)
        self.lambda_rec = getattr(cfg, 'ressfl_lambda_rec', 0.01)
        self.teacher_lambda = getattr(cfg, 'ressfl_teacher_lambda', 0.3)
        self.finetune_lr = getattr(cfg, 'ressfl_finetune_lr', 0.005)
        self.update_frequency = getattr(cfg, 'ressfl_update_frequency', 5)
        self.use_pretrain = getattr(cfg, 'ressfl_use_pretrain', False)
        self.source_task = getattr(cfg, 'ressfl_source_task', 'same')

        # Decoder-loss calibration state (global, shared across clients) 
        self.decoder_loss_baseline = None
        self.decoder_loss_calibration = []  
        self.decoder_loss_threshold = None
        self.decoder_loss_skip_count = 0  
        self._epoch_decoder_skip_count = 0  
        self._epoch_max_decoder_rec_loss = 0.0 
        self.decoder_loss_history = []  
        
        # Calibration constants
        self.CALIBRATION_UPDATES = 50
        self.DIVERGENCE_MULTIPLIER = 20.0
        self.decoder_loss_baseline_logged = False

        # Identify valid clients (non-empty datasets)
        self.valid_train_clients = [
            i for i in range(self.num_clients)
            if len(self.train_loaders[i].dataset) > 0
        ]
        self.valid_val_clients = [
            i for i in range(self.num_clients)
            if len(self.val_loaders[i].dataset) > 0
        ]

        # Build ResSFL model (backbone + bottleneck + classifier)
        self.model = build_ressfl_model(cfg).to(self.device)

        # Extract components
        self.backbone_template = self.model.backbone
        self.bottleneck_template = self.model.bottleneck
        self.classifier_template = self.model.classifier

        # Create separate backbones and bottlenecks for each client
        self.client_backbones = []
        self.client_bottlenecks = []
        for _ in range(self.num_clients):
            backbone = copy.deepcopy(self.backbone_template).to(self.device)
            bottleneck = copy.deepcopy(self.bottleneck_template).to(self.device)
            self.client_backbones.append(backbone)
            self.client_bottlenecks.append(bottleneck)

        # Shared server classifier (one global copy)
        self.server_classifier = copy.deepcopy(self.classifier_template).to(self.device)

        # Decoder for resistance-transfer (attacker-aware training)
        self.decoder = build_decoder(
            level=3,
            bottleneck_dim=cfg.get_bottleneck_dim(),
            output_dim=cfg.get_dataset_input_dim()
        ).to(self.device)

        # Client optimizers
        self.client_optimizers = []
        for i in range(self.num_clients):
            opt = torch.optim.Adam(
                list(self.client_backbones[i].parameters()) +
                list(self.client_bottlenecks[i].parameters()),
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

        # Decoder optimizer
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=cfg.learning_rate * 0.1,
            weight_decay=cfg.weight_decay
        )

        # IoMT Adaptation: Use MSE for reconstruction loss.
        self.reconstruction_criterion = nn.MSELoss()

        # Initialize default criterion first
        self.classification_criterion = nn.CrossEntropyLoss()
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
                self.classification_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                if self.verbose:
                    print(f" Using class-weighted CE: {self.class_weights}")
            else:
                self.class_weights = None
                self.class_counts = None
                self.classification_criterion = nn.CrossEntropyLoss()
        else:
            self.class_weights = None
            self.class_counts = None
            self.classification_criterion = nn.CrossEntropyLoss()


        # History tracking
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
            "train_rec_loss": [],
            "val_rec_loss": [],
            "val_mse_gl0": [],
            "val_mse_gl1": [],
            "val_mse_gl2": [],
            "val_mse_gl3": [],
            "val_mse_adaptive": [],   
            "val_privacy_mse": [],
            "val_strongest_attack_mse": [],
            "val_strongest_attacker": [],
            "privacy_utility": [],
            "val_evasion_f1": [],
            "train_grad_norm": [],
            "decoder_baseline": [],
            "decoder_threshold": [],
            "decoder_skip_count": [],
            "decoder_max_rec_loss": [],
        }

        # CSV log
        self.csv_log = []

        # Track best metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_val_f1 = 0.0
        self.best_f1_epoch = 0
        self.best_privacy_mse = -float('inf')
        self.best_privacy_epoch = 0
        self.best_privacy_utility = 0.0
        self.best_privacy_utility_epoch = 0

        # Final metrics
        self.final_metrics = {
            "final_loss": 0.0,
            "final_acc": 0.0,
            "final_f1": 0.0,
            "final_privacy_mse": 0.0,
            "final_evasion_f1": 0.0,
        }

        self.expert_trained = False
        save_config(cfg)

        # Adversary level mapping - ADDED Adaptive
        self.adversary_levels = {
            0: "GL0",
            1: "GL1",
            2: "GL2",
            3: "GL3",
            4: "Adaptive",
        }

    def _get_source_loader(self, source_task: str) -> Optional[DataLoader]:
        """
        Get data loader for source task (used in expert pre-training).
        For IoMT datasets, we use the same dataset but with a different
        split/partition to create a "source" task for expert pre-training.
        Args:
            source_task: Name of source dataset ('same', 'wesad', 'bidmc', 'mhealth')
        Returns:
            DataLoader for source task, or None if not available
        """
        from experiments import ExperimentRunner
        
        # If source_task is 'same', use the current dataset
        if source_task == 'same':
            source_task = self.cfg.dataset
    
        try:
            # Use the same dataset type as current
            if source_task == 'wesad':
                from data import get_wesad_loaders
                data_dir = ExperimentRunner.DATASET_PATHS.get('wesad', {}).get('data_dir', '')
                train_loaders, _ = get_wesad_loaders(
                    data_dir=data_dir,
                    num_clients=1,
                    batch_size=self.cfg.batch_size,
                    partition_mode='iid',
                    val_split=0.0,
                    seed=self.cfg.seed
                )
                return train_loaders[0] if train_loaders else None
            elif source_task == 'bidmc':
                from data import get_bidmc_loaders
                data_dir = ExperimentRunner.DATASET_PATHS.get('bidmc', {}).get('data_dir', '')
                train_loaders, _ = get_bidmc_loaders(
                    data_dir=data_dir,
                    num_clients=1,
                    batch_size=self.cfg.batch_size,
                    partition_mode='iid',
                    val_split=0.0,
                    seed=self.cfg.seed
                )
                return train_loaders[0] if train_loaders else None
            elif source_task == 'mhealth':
                from data import get_mhealth_loaders
                data_dir = ExperimentRunner.DATASET_PATHS.get('mhealth', {}).get('data_dir', '')
                train_loaders, _ = get_mhealth_loaders(
                    data_dir=data_dir,
                    num_clients=1,
                    batch_size=self.cfg.batch_size,
                    partition_mode='iid',
                    val_split=0.0,
                    seed=self.cfg.seed
                )
                return train_loaders[0] if train_loaders else None
            else:
                # Use current dataset's first client as source
                if self.valid_train_clients:
                    all_data = []
                    all_labels = []
                    for client_idx in self.valid_train_clients:
                        loader = self.train_loaders[client_idx]
                        for x, y in loader:
                            all_data.append(x)
                            all_labels.append(y)
                    if all_data:
                        combined_x = torch.cat(all_data, dim=0)
                        combined_y = torch.cat(all_labels, dim=0)
                        dataset = TensorDataset(combined_x, combined_y)
                        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
                return None
    
        except ImportError as e:
            return None
        except Exception as e:
            return None

    def _get_source_task_dataloader(self) -> Optional[DataLoader]:
        """
        Wrapper for _get_source_loader with error handling.
        Returns:
            DataLoader for source task, or None if not available
        """
        try:
            return self._get_source_loader(self.source_task)
        except Exception as e:
            return None

    def _get_source_task_name(self) -> str:
        """Get the name of the source task for display purposes."""
        if self.source_task == 'same':
            return self.cfg.dataset.upper()
        return self.source_task.upper()

    # ResSFL Two-Step Process: Expert Pre-training + Resistance Transfer
    def _compute_reconstruction_loss(self, zu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss on zu using MSE.
        For IoMT time-series data, MSE is more appropriate than SSIM
        (which is designed for images).
        Args:
            zu: Latent representation
            x: Original input
        Returns:
            MSE reconstruction loss
        """
        recon = self.decoder(zu)
        return self.reconstruction_criterion(recon, x)

    def pretrain_expert(self, source_loader: DataLoader) -> Optional[str]:
        """
        Step A: Expert pre-training with strong attacker-aware training.
        From ResSFL paper:
        - Use strong inversion model (L3) to closely simulate attacker
        - Use high lambda (1.2 for CIFAR-10/SVHN, 0.3 for CIFAR-100)
        - Use bottleneck layers (C8-S1, C12-S1)
        - Build MI-resistant feature extractor
        - Uses MSE for reconstruction (IoMT adaptation)
        Args:
            source_loader: DataLoader for source task
        Returns:
            Path to saved expert model, or None if failed
        """
        if source_loader is None:
            return None

        # Use strong inversion model (GL3) - strongest attacker
        self.decoder = build_decoder(
            level=3,  # GL3 - strongest
            bottleneck_dim=self.cfg.get_bottleneck_dim(),
            output_dim=self.cfg.get_dataset_input_dim()
        ).to(self.device)

        # Re-initialize decoder optimizer
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=self.cfg.learning_rate * 0.1,
            weight_decay=self.cfg.weight_decay
        )

        expert_lambda = self.teacher_lambda

        # If using default 0.3, override based on source task characteristics
        if expert_lambda == 0.3:
            source_task_name = self._get_source_task_name().lower()
            if source_task_name in ['wesad', 'bidmc']:
                expert_lambda = 0.5  # WESAD/BIDMC: moderate lambda
            elif source_task_name == 'mhealth':
                expert_lambda = 0.5  # MHEALTH: moderate lambda (12 classes)

        # Use only client 0 for expert pre-training
        if 0 in self.valid_train_clients:
            expert_client_idx = 0
        else:
            expert_client_idx = self.valid_train_clients[0] if self.valid_train_clients else 0

        # Save original optimizer state to restore later
        original_optimizer_state = None
        if expert_client_idx < len(self.client_optimizers):
            original_optimizer_state = copy.deepcopy(self.client_optimizers[expert_client_idx].state_dict())

        # source_loader is already a DataLoader
        expert_loader = source_loader

        # Track best expert accuracy
        best_expert_acc = 0.0

        for epoch in range(self.pretrain_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0

            for batch_idx, (x, y) in enumerate(expert_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # Convert one-hot labels to class indices if needed
                if y.dim() > 1 and y.shape[1] > 1:
                    y = y.argmax(dim=1)

                # Client forward (using expert client)
                zu, h = self._client_forward(expert_client_idx, x)

                # Server forward/backward
                server_input = zu.detach().requires_grad_(True)
                grad_zu_classification, class_loss = self._server_backward(server_input, y)

                # Reconstruction loss (MSE)
                rec_loss = self._compute_reconstruction_loss(zu, x)

                # Compute gradient of reconstruction loss w.r.t zu
                grad_zu_reconstruction = torch.autograd.grad(
                    rec_loss, zu,
                    grad_outputs=torch.ones_like(rec_loss),
                    retain_graph=True
                )[0]

                # Combined gradient: classification - lambda * reconstruction
                grad_zu_combined = grad_zu_classification - expert_lambda * grad_zu_reconstruction.detach()

                # Client backward
                self.client_optimizers[expert_client_idx].zero_grad()
                zu.backward(grad_zu_combined)
                self.client_optimizers[expert_client_idx].step()

                # Update decoder (attacker-aware training)
                self.decoder_optimizer.zero_grad()
                decoder_rec_loss = self._compute_reconstruction_loss(zu.detach(), x)
                decoder_rec_loss.backward()
                self.decoder_optimizer.step()

                # Track metrics
                epoch_loss += class_loss

                # Compute accuracy
                with torch.no_grad():
                    logits = self._server_forward(zu)
                    preds = logits.argmax(dim=1)
                    epoch_acc += (preds == y).sum().item() / x.size(0)

                num_batches += 1

            # Compute avg_loss and avg_acc for EVERY epoch
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            avg_acc = 100.0 * epoch_acc / num_batches if num_batches > 0 else 0.0

            # avg_acc is now always defined
            # Save best expert model
            if avg_acc > best_expert_acc:
                best_expert_acc = avg_acc
                expert_model_path = os.path.join(self.cfg.get_output_path(), "expert_model_best.pth")
                os.makedirs(os.path.dirname(expert_model_path), exist_ok=True)
                torch.save({
                    'backbone': self.client_backbones[expert_client_idx].state_dict(),
                    'bottleneck': self.client_bottlenecks[expert_client_idx].state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'epoch': epoch + 1,
                    'expert_lambda': expert_lambda,
                    'accuracy': avg_acc,
                }, expert_model_path)

        # Save final expert model
        expert_model_path = os.path.join(self.cfg.get_output_path(), "expert_model.pth")
        os.makedirs(os.path.dirname(expert_model_path), exist_ok=True)
        torch.save({
            'backbone': self.client_backbones[expert_client_idx].state_dict(),
            'bottleneck': self.client_bottlenecks[expert_client_idx].state_dict(),
            'decoder': self.decoder.state_dict(),
            'epoch': self.pretrain_epochs,
            'expert_lambda': expert_lambda,
            'best_accuracy': best_expert_acc,
        }, expert_model_path)

        self.expert_trained = True

        # Restore original optimizer state
        if original_optimizer_state is not None and expert_client_idx < len(self.client_optimizers):
            self.client_optimizers[expert_client_idx].load_state_dict(original_optimizer_state)

        return expert_model_path

    def transfer_resistance(self, expert_model_path: str) -> None:
        """
        Step B: Transfer resistance from expert to clients.
        From ResSFL paper:
        - Initialize clients with expert model
        - Use lower learning rate (0.005)
        - Use weaker inversion model (L0)
        - Update decoder every 5 steps (optimization)
        Args:
            expert_model_path: Path to saved expert model
        """
        if not os.path.exists(expert_model_path):
            return

        # 1. Load expert model
        checkpoint = torch.load(expert_model_path, map_location=self.device)

        # 2. Initialize all clients with expert model
        for client_idx in range(self.num_clients):
            self.client_backbones[client_idx].load_state_dict(checkpoint['backbone'])
            self.client_bottlenecks[client_idx].load_state_dict(checkpoint['bottleneck'])

        # 3. Set up fine-tuning with lower LR (0.005 from ResSFL paper)
        self.finetune_lr = 0.005
        for client_idx in range(self.num_clients):
            self.client_optimizers[client_idx] = torch.optim.Adam(
                list(self.client_backbones[client_idx].parameters()) +
                list(self.client_bottlenecks[client_idx].parameters()),
                lr=self.finetune_lr,
                weight_decay=self.cfg.weight_decay
            )

        # 4. Use weaker inversion model (GL0) for fine-tuning
        self.decoder = build_decoder(
            level=0,  # GL0 - weakest (lowest computational cost)
            bottleneck_dim=self.cfg.get_bottleneck_dim(),
            output_dim=self.cfg.get_dataset_input_dim()
        ).to(self.device)

        # Re-initialize decoder optimizer
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=self.cfg.learning_rate * 0.1,
            weight_decay=self.cfg.weight_decay
        )

        # Reset calibration state after transfer
        self._reset_calibration_state()

    # Helper method to reset calibration state 
    def _reset_calibration_state(self) -> None:
        """Reset decoder-loss calibration state. Called after warmup and before main training."""
        self.decoder_loss_baseline = None
        self.decoder_loss_calibration = []
        self.decoder_loss_threshold = None
        self.decoder_loss_skip_count = 0
        self._epoch_decoder_skip_count = 0
        self._epoch_max_decoder_rec_loss = 0.0
        self.decoder_loss_history = []
        self.decoder_loss_baseline_logged = False

    # Core Training Functions
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

    def _federated_average_separate(
        self, backbone_states: List[Dict[str, torch.Tensor]],
        bottleneck_states: List[Dict[str, torch.Tensor]],
        client_indices: Optional[List[int]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Federated averaging of client backbones and bottlenecks separately.
        Uses weighted averaging based on client dataset sizes.
        Args:
            backbone_states: List of backbone state dicts
            bottleneck_states: List of bottleneck state dicts
            client_indices: List of client indices (for weighting)
        Returns:
            (avg_backbone_state, avg_bottleneck_state)
        """
        if not backbone_states or not bottleneck_states:
            return {}, {}

        if client_indices is None:
            client_indices = list(range(len(backbone_states)))

        # Compute weights based on dataset sizes
        weights = self._get_client_weights(client_indices)

        # Average backbones
        avg_backbone = {}
        for key in backbone_states[0].keys():
            avg_backbone[key] = sum(
                weights[i] * backbone_states[i][key]
                for i in range(len(backbone_states))
            )

        # Average bottlenecks
        avg_bottleneck = {}
        for key in bottleneck_states[0].keys():
            avg_bottleneck[key] = sum(
                weights[i] * bottleneck_states[i][key]
                for i in range(len(bottleneck_states))
            )

        return avg_backbone, avg_bottleneck

    def _server_forward(self, zu: torch.Tensor) -> torch.Tensor:
        """Server forward pass: zu -> logits."""
        return self.server_classifier(zu)

    def _server_backward(self, zu: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Server backward pass.
        Returns:
            (gradients to send back to client, classification loss value for logging)
        """
        self.server_optimizer.zero_grad()
        labels = labels.to(self.device)

        # Convert one-hot labels to class indices if needed
        if labels.dim() > 1 and labels.shape[1] > 1:
            labels = labels.argmax(dim=1)

        # Forward pass
        logits = self._server_forward(zu)

        # Compute loss
        loss = self.classification_criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Ensure gradients exist
        if zu.grad is None:
            raise RuntimeError("No gradients computed for zu. Check graph connection.")

        # Get gradients of zu
        grad_zu = zu.grad.clone().detach()

        # Update server
        self.server_optimizer.step()

        return grad_zu, loss.item()

    def _client_forward(self, client_idx: int, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Client forward pass: x -> h -> zu."""
        h = self.client_backbones[client_idx](x)
        zu = self.client_bottlenecks[client_idx](h)
        return zu, h

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
        temp_model = TemporaryResSFLModelForPGD(
            backbone=self.client_backbones[client_idx],
            bottleneck=self.client_bottlenecks[client_idx],
            server_classifier=self.server_classifier
        ).to(self.device)

        temp_model.eval()

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

    def _compute_classification_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute classification metrics from logits and labels.
        Returns:
            Dictionary with accuracy, precision, recall, f1, auprc
        """
        with torch.no_grad():
            # Convert one-hot labels to class indices if needed
            if labels.dim() > 1 and labels.shape[1] > 1:
                labels = labels.argmax(dim=1)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            # Convert to numpy for sklearn metrics
            y_true = labels.cpu().numpy()
            y_pred = preds.cpu().numpy()
            y_prob = probs.cpu().numpy()

            # Compute metrics
            acc = compute_accuracy(y_true, y_pred)

            # For multi-class, use weighted average
            if len(np.unique(y_true)) > 1:
                precision = compute_precision(y_true, y_pred, average='weighted')
                recall = compute_recall(y_true, y_pred, average='weighted')
                f1 = compute_f1(y_true, y_pred, average='weighted')
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0

            # AUPRC
            auprc = compute_auprc(y_true, y_prob) if y_prob is not None else 0.0

            return {
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auprc': auprc,
            }

    # train_client with gradient clipping and decoder-loss guard
    def train_client(self, client_idx: int, local_epochs: int = 1,
                     lambda_rec: float = 0.01) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
        """
        Train client with reconstruction-gradient norm clipping to prevent collapse.
        Also includes global decoder-loss calibration guard to detect and handle
        reconstruction loss explosions.

        Returns:
            (backbone_state_dict, bottleneck_state_dict, avg_gradient_norm)
        """
        if len(self.train_loaders[client_idx].dataset) == 0:
            return (
                self.client_backbones[client_idx].state_dict(),
                self.client_bottlenecks[client_idx].state_dict(),
                0.0
            )

        loader = self.train_loaders[client_idx]

        self.client_backbones[client_idx].train()
        self.client_bottlenecks[client_idx].train()
        self.server_classifier.train()
        self.decoder.train()

        # Track update frequency for decoder (every 5 steps, from ResSFL paper)
        step_counter = 0
        
        # Track gradient norm for monitoring
        running_grad_norm = 0.0
        num_steps = 0

        for epoch in range(local_epochs):
            for batch_idx, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # Convert one-hot labels to class indices if needed
                if y.dim() > 1 and y.shape[1] > 1:
                    y = y.argmax(dim=1)

                # 1. Client Forward: x -> zu (with computational graph)
                zu, h = self._client_forward(client_idx, x)

                # 2. Detach for Server (values only, no graph)
                server_input = zu.detach().requires_grad_(True)

                # 3. Server Forward/Backward
                grad_zu_classification, class_loss = self._server_backward(server_input, y)

                # 4. Compute Reconstruction Loss on zu
                rec_loss = self._compute_reconstruction_loss(zu, x)

                # 5. Compute Gradient of Reconstruction Loss w.r.t zu
                grad_zu_reconstruction = torch.autograd.grad(
                    rec_loss, zu,
                    grad_outputs=torch.ones_like(rec_loss),
                    retain_graph=True,
                    create_graph=False
                )[0]

                # Reconstruction-gradient norm clipping
                REC_GRAD_MAX_NORM = 1.0

                grad_zu_rec_norm = grad_zu_reconstruction.norm()

                # Track gradient norm
                running_grad_norm += grad_zu_rec_norm.item()
                num_steps += 1

                if grad_zu_rec_norm > REC_GRAD_MAX_NORM:
                    grad_zu_reconstruction = (
                        grad_zu_reconstruction *
                        (REC_GRAD_MAX_NORM / grad_zu_rec_norm)
                    )

                # 5.5 Sanity Check: Verify gradient shapes match
                assert grad_zu_classification.shape == grad_zu_reconstruction.shape, \
                    f"Gradient shape mismatch: classification {grad_zu_classification.shape} vs " \
                    f"reconstruction {grad_zu_reconstruction.shape}"

                # 6. Combined Gradient: classification - lambda_rec * reconstruction
                grad_zu_combined = grad_zu_classification - lambda_rec * grad_zu_reconstruction.detach()

                # 7. Client Backward using Combined Gradient
                self.client_optimizers[client_idx].zero_grad()
                zu.backward(grad_zu_combined)

                # Add gradient clipping to prevent training collapse (BIDMC issue)
                torch.nn.utils.clip_grad_norm_(
                    list(self.client_backbones[client_idx].parameters()) +
                    list(self.client_bottlenecks[client_idx].parameters()),
                    max_norm=1.0
                )

                self.client_optimizers[client_idx].step()

                # 8. Update Decoder Separately (Attacker-Aware Training)
                step_counter += 1
                if step_counter % self.update_frequency == 0:
                    # Compute decoder loss inside the update block 
                    decoder_rec_loss = self._compute_reconstruction_loss(zu.detach(), x)
                    loss_val = decoder_rec_loss.item()
                    
                    # Append to history
                    self.decoder_loss_history.append(loss_val)
                    
                    # Track per-epoch maximum
                    self._epoch_max_decoder_rec_loss = max(
                        self._epoch_max_decoder_rec_loss,
                        loss_val
                    )

                    # Calibration and guard logic 
                    skip_decoder_update = False
                    
                    # Calibration phase: collect losses until we have enough
                    if self.decoder_loss_baseline is None:
                        self.decoder_loss_calibration.append(loss_val)
                        
                        if len(self.decoder_loss_calibration) >= self.CALIBRATION_UPDATES:
                            # Compute baseline as median of collected losses
                            self.decoder_loss_baseline = np.median(self.decoder_loss_calibration)
                            self.decoder_loss_threshold = (
                                self.decoder_loss_baseline * self.DIVERGENCE_MULTIPLIER
                            )
                            self.decoder_loss_baseline_logged = True
                            
                            if self.verbose:
                                print(
                                    f"  [decoder calibration] baseline={self.decoder_loss_baseline:.2f}, "
                                    f"threshold={self.decoder_loss_threshold:.2f} "
                                    f"(based on {len(self.decoder_loss_calibration)} decoder updates)"
                                )
                    
                    # Guard phase: check if current loss exceeds threshold
                    if self.decoder_loss_threshold is not None:
                        if loss_val > self.decoder_loss_threshold:
                            # Skip this decoder update to prevent explosion
                            self.decoder_loss_skip_count += 1
                            self._epoch_decoder_skip_count += 1
                            
                            if self.verbose:
                                print(
                                    f"  [decoder guard] SKIPPED update: loss={loss_val:.2f} > "
                                    f"threshold={self.decoder_loss_threshold:.2f} "
                                    f"(skip #{self.decoder_loss_skip_count})"
                                )
                            skip_decoder_update = True

                    if not skip_decoder_update:
                        self.decoder_optimizer.zero_grad()
                        decoder_rec_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                        self.decoder_optimizer.step()

        # Compute average gradient norm
        avg_grad_norm = running_grad_norm / num_steps if num_steps > 0 else 0.0

        return (
            self.client_backbones[client_idx].state_dict(),
            self.client_bottlenecks[client_idx].state_dict(),
            avg_grad_norm
        )

    def evaluate_client(self, client_idx: int, loader: DataLoader,
                        compute_reconstruction: bool = False,
                        compute_metrics: bool = False) -> Tuple:
        """Evaluate client without adversarial attack."""
        if len(loader.dataset) == 0:
            return 0.0, 0.0, None, {}

        self.client_backbones[client_idx].eval()
        self.client_bottlenecks[client_idx].eval()
        self.server_classifier.eval()
        self.decoder.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_rec_loss = 0.0
        rec_count = 0

        # Collect all predictions for metric computation
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                # Convert one-hot labels to class indices if needed
                if y.dim() > 1 and y.shape[1] > 1:
                    y = y.argmax(dim=1)

                # Client forward
                zu, h = self._client_forward(client_idx, x)

                # Server forward
                logits = self._server_forward(zu)

                # Classification loss
                loss = self.classification_criterion(logits, y)
                total_loss += loss.item() * x.size(0)

                # Accuracy
                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0)

                # Collect for metrics
                if compute_metrics:
                    all_y_true.extend(y.cpu().numpy())
                    all_y_pred.extend(preds.cpu().numpy())
                    all_y_prob.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())

                # Reconstruction loss
                if compute_reconstruction:
                    rec_loss = self._compute_reconstruction_loss(zu, x)
                    total_rec_loss += rec_loss.item() * x.size(0)
                    rec_count += x.size(0)

        if total_samples == 0:
            return 0.0, 0.0, None, {}

        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples

        # Compute metrics
        metrics = {}
        if compute_metrics and len(all_y_true) > 0:
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_y_prob = np.array(all_y_prob) if all_y_prob else None

            # Convert one-hot labels to 1D before computing metrics
            if all_y_true.ndim > 1 and all_y_true.shape[1] > 1:
                all_y_true = np.argmax(all_y_true, axis=1)
            if all_y_pred.ndim > 1 and all_y_pred.shape[1] > 1:
                all_y_pred = np.argmax(all_y_pred, axis=1)

            metrics['accuracy'] = compute_accuracy(all_y_true, all_y_pred)
            metrics['precision'] = compute_precision(all_y_true, all_y_pred, average='weighted')
            metrics['recall'] = compute_recall(all_y_true, all_y_pred, average='weighted')
            metrics['f1'] = compute_f1(all_y_true, all_y_pred, average='weighted')
            metrics['auprc'] = compute_auprc(all_y_true, all_y_prob)

        if compute_reconstruction:
            avg_rec_loss = total_rec_loss / rec_count if rec_count > 0 else None
            return avg_loss, avg_acc, avg_rec_loss, metrics

        return avg_loss, avg_acc, None, metrics

    def evaluate_client_with_metrics(self, client_idx: int, loader: DataLoader,
                                     compute_reconstruction: bool = False,
                                     compute_robustness: bool = False,
                                     compute_metrics: bool = False) -> Tuple:
        """
        Evaluate client with PGD attack metrics.
        
        Args:
            client_idx: Client index
            loader: DataLoader
            compute_reconstruction: If True, compute reconstruction loss
            compute_robustness: If True, evaluate PGD attack on ALL batches
            compute_metrics: If True, compute classification metrics
        
        Returns:
            (avg_loss, avg_acc, avg_rec_loss, avg_adv_loss, avg_privacy_penalty, metrics, evasion_f1)
        """
        if len(loader.dataset) == 0:
            return 0.0, 0.0, None, None, None, {}, 0.0

        self.client_backbones[client_idx].eval()
        self.client_bottlenecks[client_idx].eval()
        self.server_classifier.eval()
        self.decoder.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_rec_loss = 0.0
        total_adv_loss = 0.0
        rec_count = 0
        adv_count = 0

        # Collect all predictions for metric computation
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                # Convert one-hot labels to class indices if needed
                if y.dim() > 1 and y.shape[1] > 1:
                    y = y.argmax(dim=1)

                # Client forward
                zu, h = self._client_forward(client_idx, x)

                # Server forward
                logits = self._server_forward(zu)

                # Classification loss
                loss = self.classification_criterion(logits, y)
                total_loss += loss.item() * x.size(0)

                # Accuracy
                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0)

                # Collect for metrics
                if compute_metrics:
                    all_y_true.extend(y.cpu().numpy())
                    all_y_pred.extend(preds.cpu().numpy())
                    all_y_prob.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())

                # Reconstruction loss
                if compute_reconstruction:
                    rec_loss = self._compute_reconstruction_loss(zu, x)
                    total_rec_loss += rec_loss.item() * x.size(0)
                    rec_count += x.size(0)

        # Initialize adversarial prediction collections for F1 computation
        all_y_true_adv = []
        all_y_pred_adv = []
        evasion_f1 = 0.0

        # Evaluate PGD attack on ALL batches when compute_robustness is True
        if compute_robustness:
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                # Convert one-hot labels to class indices if needed
                if y.dim() > 1 and y.shape[1] > 1:
                    y = y.argmax(dim=1)

                with torch.enable_grad():
                    x_adv = self._generate_pgd_attack(client_idx, x, y, eval_mode=True)

                with torch.no_grad():
                    adv_zu, _ = self._client_forward(client_idx, x_adv)
                    adv_logits = self._server_forward(adv_zu)
                    adv_preds = adv_logits.argmax(dim=1)
                    
                    # Collect for adversarial F1 computation
                    all_y_true_adv.extend(y.cpu().numpy())
                    all_y_pred_adv.extend(adv_preds.cpu().numpy())

                    adv_loss = self.classification_criterion(adv_logits, y)
                    total_adv_loss += adv_loss.item() * x.size(0)
                    adv_count += x.size(0)

            # Compute weighted F1 for adversarial predictions
            if len(all_y_true_adv) > 0:
                evasion_f1 = compute_f1(
                    np.array(all_y_true_adv),
                    np.array(all_y_pred_adv),
                    average='weighted'
                )

        if total_samples == 0:
            return 0.0, 0.0, None, None, None, {}, 0.0

        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        avg_rec_loss = total_rec_loss / rec_count if rec_count > 0 else None
        avg_adv_loss = total_adv_loss / adv_count if adv_count > 0 else None

        metrics = {}
        if compute_metrics and len(all_y_true) > 0:
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_y_prob = np.array(all_y_prob) if all_y_prob else None

            # Convert one-hot labels to 1D before computing metrics
            if all_y_true.ndim > 1 and all_y_true.shape[1] > 1:
                all_y_true = np.argmax(all_y_true, axis=1)
            if all_y_pred.ndim > 1 and all_y_pred.shape[1] > 1:
                all_y_pred = np.argmax(all_y_pred, axis=1)

            metrics['accuracy'] = compute_accuracy(all_y_true, all_y_pred)
            metrics['precision'] = compute_precision(all_y_true, all_y_pred, average='weighted')
            metrics['recall'] = compute_recall(all_y_true, all_y_pred, average='weighted')
            metrics['f1'] = compute_f1(all_y_true, all_y_pred, average='weighted')
            metrics['auprc'] = compute_auprc(all_y_true, all_y_prob)

        return avg_loss, avg_acc, avg_rec_loss, avg_adv_loss, None, metrics, evasion_f1

    # train_round with gradient norm tracking and decoder diagnostics
    def train_round(self, round_idx: int, lambda_rec: float = 0.01,
                    compute_robustness: bool = False) -> Dict[str, float]:
        """
        Train one round: all clients train locally, then federated averaging.
        
        Args:
            round_idx: The actual training round index
            lambda_rec: Reconstruction loss weight
            compute_robustness: If True, compute evasion F1 during validation
        """
        self._epoch_decoder_skip_count = 0
        self._epoch_max_decoder_rec_loss = 0.0

        client_backbone_states = []
        client_bottleneck_states = []

        # Collect metrics from all clients
        train_losses = []
        train_accs = []
        train_precisions = []
        train_recalls = []
        train_f1s = []
        train_auprcs = []
        train_rec_losses = []
        train_grad_norms = []

        # Train each valid client
        for client_idx in self.valid_train_clients:
            # Train client - now returns 3 values
            backbone_state, bottleneck_state, avg_grad_norm = self.train_client(
                client_idx, local_epochs=1, lambda_rec=lambda_rec
            )
            client_backbone_states.append(backbone_state)
            client_bottleneck_states.append(bottleneck_state)
            
            # Track gradient norm
            train_grad_norms.append(avg_grad_norm)

            # Evaluate client on its training set (no PGD robustness for efficiency)
            loss, acc, rec_loss, adv_loss, _, metrics, _ = self.evaluate_client_with_metrics(
                client_idx, self.train_loaders[client_idx],
                compute_reconstruction=True, compute_robustness=False, compute_metrics=True
            )
            train_losses.append(loss)
            train_accs.append(acc)
            if rec_loss is not None:
                train_rec_losses.append(rec_loss)
            if metrics:
                train_precisions.append(metrics.get('precision', 0.0))
                train_recalls.append(metrics.get('recall', 0.0))
                train_f1s.append(metrics.get('f1', 0.0))
                train_auprcs.append(metrics.get('auprc', 0.0))

        # Federated averaging of client backbones and bottlenecks separately
        if client_backbone_states and client_bottleneck_states:
            avg_backbone, avg_bottleneck = self._federated_average_separate(
                client_backbone_states,
                client_bottleneck_states,
                self.valid_train_clients
            )

            # Load averaged weights into all client backbones and bottlenecks
            if avg_backbone:
                for client_idx in range(self.num_clients):
                    self.client_backbones[client_idx].load_state_dict(avg_backbone)

            if avg_bottleneck:
                for client_idx in range(self.num_clients):
                    self.client_bottlenecks[client_idx].load_state_dict(avg_bottleneck)

        # Evaluate all valid clients on validation sets
        val_losses = []
        val_accs = []
        val_precisions = []
        val_recalls = []
        val_f1s = []
        val_auprcs = []
        val_rec_losses = []
        val_evasion_f1s = []

        for client_idx in self.valid_val_clients:
            loss, acc, rec_loss, adv_loss, _, metrics, evasion_f1 = self.evaluate_client_with_metrics(
                client_idx, self.val_loaders[client_idx],
                compute_reconstruction=True, compute_robustness=compute_robustness, compute_metrics=True
            )
            val_losses.append(loss)
            val_accs.append(acc)
            val_evasion_f1s.append(evasion_f1)
            if rec_loss is not None:
                val_rec_losses.append(rec_loss)
            if metrics:
                val_precisions.append(metrics.get('precision', 0.0))
                val_recalls.append(metrics.get('recall', 0.0))
                val_f1s.append(metrics.get('f1', 0.0))
                val_auprcs.append(metrics.get('auprc', 0.0))

        # Aggregate metrics over valid clients
        round_metrics = {
            "train_loss": np.mean(train_losses) if train_losses else 0.0,
            "train_acc": np.mean(train_accs) if train_accs else 0.0,
            "train_precision": np.mean(train_precisions) if train_precisions else 0.0,
            "train_recall": np.mean(train_recalls) if train_recalls else 0.0,
            "train_f1": np.mean(train_f1s) if train_f1s else 0.0,
            "train_auprc": np.mean(train_auprcs) if train_auprcs else 0.0,
            "train_rec_loss": np.mean(train_rec_losses) if train_rec_losses else 0.0,
            "train_grad_norm": np.mean(train_grad_norms) if train_grad_norms else 0.0,
            "val_loss": np.mean(val_losses) if val_losses else 0.0,
            "val_acc": np.mean(val_accs) if val_accs else 0.0,
            "val_precision": np.mean(val_precisions) if val_precisions else 0.0,
            "val_recall": np.mean(val_recalls) if val_recalls else 0.0,
            "val_f1": np.mean(val_f1s) if val_f1s else 0.0,
            "val_auprc": np.mean(val_auprcs) if val_auprcs else 0.0,
            "val_rec_loss": np.mean(val_rec_losses) if val_rec_losses else 0.0,
            "val_evasion_f1": np.mean(val_evasion_f1s) if val_evasion_f1s else 0.0,
            "decoder_baseline": self.decoder_loss_baseline if self.decoder_loss_baseline is not None else 0.0,
            "decoder_threshold": self.decoder_loss_threshold if self.decoder_loss_threshold is not None else 0.0,
            "decoder_skip_count": self._epoch_decoder_skip_count,
            "decoder_max_rec_loss": self._epoch_max_decoder_rec_loss,
        }

        return round_metrics

    # Inversion Attack Evaluation with Adaptive Attacker
    def evaluate_inversion_attack(self) -> Dict[str, Any]:
        """Evaluate inversion attack using GL0-GL3 hierarchy PLUS Adaptive attacker."""
        # Collect test data from all clients
        all_zu = []
        all_x = []

        for client_idx in self.valid_val_clients:
            loader = self.val_loaders[client_idx]
            self.client_backbones[client_idx].eval()
            self.client_bottlenecks[client_idx].eval()

            with torch.no_grad():
                for x, y in loader:
                    x = x.to(self.device)
                    zu, h = self._client_forward(client_idx, x)
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

        # Evaluate inversion hierarchy (GL0-GL3 + Adaptive)
        results, reconstructions, trained_decoders = evaluate_trained_inversion_hierarchy(
            cfg=self.cfg,
            zu_train=zu_train,
            x_train=x_train,
            zu_test=zu_test,
            x_test=x_test,
            device=self.device,
            attack_epochs=50,
            learning_rate=1e-3,
            batch_size=64,
            verbose=False,
            include_adaptive=True
        )

        # Ensure adaptive MSE is included in results
        if 'mse_adaptive' not in results:
            results['mse_adaptive'] = 0.0

        return results

    # verify_reconstruction_hierarchy with Adaptive
    def verify_reconstruction_hierarchy(self) -> Dict[str, float]:
        """Verify reconstruction hierarchy ordering for GL0-GL3 + Adaptive."""
        results = self.evaluate_inversion_attack()

        mse_values = {
            "GL0": results.get("mse_gl0", 0.0),
            "GL1": results.get("mse_gl1", 0.0),
            "GL2": results.get("mse_gl2", 0.0),
            "GL3": results.get("mse_gl3", 0.0),
            "Adaptive": results.get("mse_adaptive", 0.0),
        }

        # Check ordering (lower is better for attacker)
        # GL0 should have highest MSE (weakest attacker)
        # GL3 should have lowest MSE (strongest attacker)
        ordering_ok = True
        tol = 0.15
        levels = ["GL0", "GL1", "GL2", "GL3", "Adaptive"]
        for i in range(len(levels) - 1):
            if mse_values[levels[i]] + tol < mse_values[levels[i + 1]]:
                ordering_ok = False
                print(f" Warning: {levels[i]} MSE ({mse_values[levels[i]]:.4f}) < "
                      f"{levels[i + 1]} MSE ({mse_values[levels[i + 1]]:.4f})")

        print("\nReconstruction Hierarchy Verification:")
        for level in levels:
            print(f" {level}: {mse_values[level]:.4f}")
        print(f" Ordering OK: {ordering_ok}")

        return mse_values


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
                "train_grad_norm",
                "decoder_baseline", "decoder_threshold", "decoder_skip_count", "decoder_max_rec_loss"
            ])
            if self.csv_log:
                writer.writerows(self.csv_log)

    def fit(self) -> Dict[str, List[float]]:
        """
        Run full training for cfg.epochs rounds.
        If use_pretrain is True, performs:
        1. Expert pre-training on source task
        2. Resistance transfer to clients
        3. Warm-up phase (lambda_rec=0) - only if use_pretrain is True
        4. Main training on target task
        Returns:
            history dictionary with metrics
        """
        warmup_epochs = self.warmup_epochs if self.use_pretrain else 0
        main_epochs = self.cfg.main_epochs

        # Step A: Expert Pre-training (if enabled)
        if self.use_pretrain:
            source_loader = self._get_source_task_dataloader()
            if source_loader is not None:
                expert_path = self.pretrain_expert(source_loader)
                if expert_path is not None:
                    self.transfer_resistance(expert_path)
                    print("\n" + "=" * 60)

        # Reset calibration state after transfer and before warmup 
        self._reset_calibration_state()

        # Warm-up Phase (lambda_rec = 0 for first N epochs)
        start_epoch = 0

        if warmup_epochs > 0:
            for warmup_idx in range(warmup_epochs):
                # Use warmup_idx as round_idx (no history storage for warmup)
                metrics = self.train_round(
                    warmup_idx,
                    lambda_rec=0.0,
                    compute_robustness=False
                )

            start_epoch = warmup_epochs

        #Reset calibration state before main training
        # This guarantees calibration is based on main training data only
        self._reset_calibration_state()

        # Main Training Loop (fine-tuning if pretrain was used)
        print(f"\n--- Main Training ({main_epochs} epochs) ---")
        for epoch_idx in range(1, main_epochs + 1):
            round_idx = start_epoch + epoch_idx - 1

            compute_robustness = (epoch_idx % 10 == 0) or (epoch_idx == main_epochs)

            metrics = self.train_round(
                round_idx,
                lambda_rec=self.lambda_rec,
                compute_robustness=compute_robustness
            )

            if (epoch_idx % 10 == 0) or (epoch_idx == main_epochs):
                inv_results = self.evaluate_inversion_attack()
                val_mse_gl0 = inv_results.get("mse_gl0", 0.0)
                val_mse_gl1 = inv_results.get("mse_gl1", 0.0)
                val_mse_gl2 = inv_results.get("mse_gl2", 0.0)
                val_mse_gl3 = inv_results.get("mse_gl3", 0.0)
                val_mse_adaptive = inv_results.get("mse_adaptive", 0.0)
                val_privacy_mse = inv_results.get("privacy_mse", 0.0)
                val_strongest_attack_mse = inv_results.get("strongest_attack_mse", 0.0)
                val_strongest_attacker = inv_results.get("strongest_attacker", "None")
            else:
                val_mse_gl0 = self.history["val_mse_gl0"][-1] if self.history["val_mse_gl0"] else 0.0
                val_mse_gl1 = self.history["val_mse_gl1"][-1] if self.history["val_mse_gl1"] else 0.0
                val_mse_gl2 = self.history["val_mse_gl2"][-1] if self.history["val_mse_gl2"] else 0.0
                val_mse_gl3 = self.history["val_mse_gl3"][-1] if self.history["val_mse_gl3"] else 0.0
                val_mse_adaptive = self.history["val_mse_adaptive"][-1] if self.history["val_mse_adaptive"] else 0.0
                val_privacy_mse = self.history["val_privacy_mse"][-1] if self.history["val_privacy_mse"] else 0.0
                val_strongest_attack_mse = self.history["val_strongest_attack_mse"][-1] if self.history["val_strongest_attack_mse"] else 0.0
                val_strongest_attacker = self.history["val_strongest_attacker"][-1] if self.history["val_strongest_attacker"] else "None"

            # Compute Privacy-Utility Score
            val_f1 = metrics.get("val_f1", 0.0)
            privacy_utility = val_f1 * val_privacy_mse

            # Track Best Models
            if metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = metrics["val_acc"]
                # Store main epoch number (not total)
                self.best_epoch = epoch_idx
                self.save_checkpoint("best_acc")

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                # Store main epoch number (not total)
                self.best_f1_epoch = epoch_idx
                self.save_checkpoint("best_f1")

            if val_privacy_mse > self.best_privacy_mse:
                self.best_privacy_mse = val_privacy_mse               
                self.best_privacy_epoch = epoch_idx
                self.save_checkpoint("best_privacy")

            if privacy_utility > self.best_privacy_utility:
                self.best_privacy_utility = privacy_utility
                # Store main epoch number (not total)
                self.best_privacy_utility_epoch = epoch_idx
                self.save_checkpoint("best_privacy_utility")

            # Store History (main epochs only)
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
            self.history["train_rec_loss"].append(metrics.get("train_rec_loss", 0.0))
            self.history["val_rec_loss"].append(metrics.get("val_rec_loss", 0.0))
            self.history["val_mse_gl0"].append(val_mse_gl0)
            self.history["val_mse_gl1"].append(val_mse_gl1)
            self.history["val_mse_gl2"].append(val_mse_gl2)
            self.history["val_mse_gl3"].append(val_mse_gl3)
            self.history["val_mse_adaptive"].append(val_mse_adaptive)
            self.history["val_privacy_mse"].append(val_privacy_mse)
            self.history["val_strongest_attack_mse"].append(val_strongest_attack_mse)
            self.history["val_strongest_attacker"].append(val_strongest_attacker)
            self.history["privacy_utility"].append(privacy_utility)
            self.history["val_evasion_f1"].append(metrics.get("val_evasion_f1", 0.0))
            self.history["train_grad_norm"].append(metrics.get("train_grad_norm", 0.0))
            self.history["decoder_baseline"].append(metrics.get("decoder_baseline", 0.0))
            self.history["decoder_threshold"].append(metrics.get("decoder_threshold", 0.0))
            self.history["decoder_skip_count"].append(metrics.get("decoder_skip_count", 0))
            self.history["decoder_max_rec_loss"].append(metrics.get("decoder_max_rec_loss", 0.0))

            # Append to CSV Log (main epochs only, epoch_idx is 1-based)
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
                metrics.get("train_rec_loss", 0.0),
                metrics.get("val_rec_loss", 0.0),
                val_mse_gl0,
                val_mse_gl1,
                val_mse_gl2,
                val_mse_gl3,
                val_mse_adaptive,
                val_privacy_mse,
                val_strongest_attack_mse,
                val_strongest_attacker,
                privacy_utility,
                metrics.get("val_evasion_f1", 0.0),
                self.best_val_acc,
                self.best_val_f1,
                self.best_privacy_mse,
                self.best_privacy_utility,
                metrics.get("train_grad_norm", 0.0),
                metrics.get("decoder_baseline", 0.0),
                metrics.get("decoder_threshold", 0.0),
                metrics.get("decoder_skip_count", 0),
                metrics.get("decoder_max_rec_loss", 0.0)
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
                phase = "Fine-tuning" if self.use_pretrain and self.expert_trained else "Training"
                print(f"Epoch {epoch_idx}/{main_epochs} [{phase}]: "
                      f"Train Acc: {metrics['train_acc']:.2f}% | "
                      f"Val Acc: {metrics['val_acc']:.2f}% | "
                      f"Val F1: {val_f1:.4f} | "
                      f"Privacy MSE: {val_privacy_mse:.4f} | "
                      f"Privacy-Utility: {privacy_utility:.4f} | "
                      f"Val Evasion F1: {metrics.get('val_evasion_f1', 0.0):.4f} | "
                      f"Grad Norm: {metrics.get('train_grad_norm', 0.0):.4f} | "
                      f"Decoder Skips: {metrics.get('decoder_skip_count', 0)} | "
                      f"Max Rec Loss: {metrics.get('decoder_max_rec_loss', 0.0):.2f} | "
                      f"Best Acc: {self.best_val_acc:.2f}% | "
                      f"Best F1: {self.best_val_f1:.4f} | "
                      f"Best Privacy: {self.best_privacy_mse:.4f}")

        # Store final metrics before saving
        if len(self.history["val_loss"]) > 0:
            self.final_metrics["final_loss"] = self.history["val_loss"][-1]
            self.final_metrics["final_acc"] = self.history["val_acc"][-1]
            self.final_metrics["final_f1"] = self.history["val_f1"][-1]
            self.final_metrics["final_privacy_mse"] = self.history["val_privacy_mse"][-1]
            self.final_metrics["final_evasion_f1"] = self.history["val_evasion_f1"][-1]

        # Save final checkpoint and logs
        self.save_checkpoint("final")
        self.save_metrics()

        # Final CSV save ensures all rows are written
        self._save_csv_log()

        # Final inversion attack evaluation and hierarchy verification
        print("\n" + "=" * 60)
        print("Final Inversion Attack Evaluation")
        print("=" * 60)
        self.verify_reconstruction_hierarchy()

        # Check convergence
        print("\nConvergence Check:")
        train_losses = self.history["train_loss"]
        val_losses = self.history["val_loss"]
        train_accs = self.history["train_acc"]
        val_accs = self.history["val_acc"]

        if len(train_losses) > 10:
            has_nan = any(np.isnan(l) for l in train_losses) or any(np.isnan(l) for l in val_losses)
            print(f" No NaN values: {not has_nan}")

            train_loss_trend = np.mean(train_losses[-10:]) < np.mean(train_losses[:10])
            val_loss_trend = np.mean(val_losses[-10:]) < np.mean(val_losses[:10])
            print(f" Train loss decreasing: {train_loss_trend}")
            print(f" Val loss decreasing: {val_loss_trend}")

            train_acc_trend = np.mean(train_accs[-10:]) > np.mean(train_accs[:10])
            val_acc_trend = np.mean(val_accs[-10:]) > np.mean(val_accs[:10])
            print(f" Train acc increasing: {train_acc_trend}")
            print(f" Val acc increasing: {val_acc_trend}")

        # Print decoder calibration summary with safe None handling
        print("\nDecoder Calibration Summary:")
        baseline_str = (
            f"{self.decoder_loss_baseline:.2f}"
            if self.decoder_loss_baseline is not None
            else "Not calibrated"
        )
        threshold_str = (
            f"{self.decoder_loss_threshold:.2f}"
            if self.decoder_loss_threshold is not None
            else "Not calibrated"
        )
        print(f" Baseline: {baseline_str}")
        print(f" Threshold: {threshold_str}")
        print(f" Total skips: {self.decoder_loss_skip_count}")
        print(f" Calibration updates collected: {len(self.decoder_loss_calibration)}")

        print("\n" + "=" * 60)
        print(f" Final epoch F1 (reported): {self.final_metrics['final_f1']:.4f}")
        print(f" Final epoch Privacy MSE (reported): {self.final_metrics['final_privacy_mse']:.4f}")
        print(f" Final epoch Evasion F1 (reported): {self.final_metrics['final_evasion_f1']:.4f}")
        print("=" * 60)

        return self.history

    def save_checkpoint(self, name: str = "last") -> None:
        """
        Save checkpoint including all client models, server classifier, and decoder.
        """
        checkpoint = {
            "main_epoch": len(self.history["train_loss"]),
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "best_val_f1": self.best_val_f1,
            "best_f1_epoch": self.best_f1_epoch,
            "best_privacy_mse": self.best_privacy_mse,
            "best_privacy_epoch": self.best_privacy_epoch,
            "best_privacy_utility": self.best_privacy_utility,
            "best_privacy_utility_epoch": self.best_privacy_utility_epoch,
            "final_metrics": self.final_metrics,
            "server_classifier": self.server_classifier.state_dict(),
            "server_optimizer": self.server_optimizer.state_dict(),
            "decoder": self.decoder.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),
            "client_backbones": [
                bb.state_dict() for bb in self.client_backbones
            ],
            "client_bottlenecks": [
                bn.state_dict() for bn in self.client_bottlenecks
            ],
            "client_optimizers": [
                opt.state_dict() for opt in self.client_optimizers
            ],
            "history": self.history,
            "cfg": self.cfg.to_dict(),
            "use_pretrain": self.use_pretrain,
            "expert_trained": self.expert_trained,
            # Decoder calibration state 
            "decoder_loss_baseline": self.decoder_loss_baseline,
            "decoder_loss_threshold": self.decoder_loss_threshold,
            "decoder_loss_skip_count": self.decoder_loss_skip_count,
            "decoder_loss_calibration": self.decoder_loss_calibration,
            "decoder_loss_history": self.decoder_loss_history[-1000:] if len(self.decoder_loss_history) > 1000 else self.decoder_loss_history,
        }

        output_dir = self.cfg.get_output_path()
        checkpoint_path = os.path.join(output_dir, f"checkpoint_ressfl_{name}.pth")
        torch.save(checkpoint, checkpoint_path)

    def save_metrics(self) -> None:
        """Save final metrics to JSON file with timestamp."""
        import time
        output_dir = self.cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(output_dir, f"metrics_ressfl_{timestamp}.json")

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        final_inv_results = self.evaluate_inversion_attack()

        final_privacy_utility = self.history["privacy_utility"][-1] if self.history["privacy_utility"] else 0.0

        if self.history["val_privacy_mse"] and self.history["val_strongest_attacker"]:
            best_idx = np.argmax(self.history["val_privacy_mse"])
            best_strongest_attacker = self.history["val_strongest_attacker"][best_idx] if best_idx < len(self.history["val_strongest_attacker"]) else "None"
        else:
            best_strongest_attacker = "None"

        # This dynamically finds the attacker with the lowest MSE (strongest attacker)
        mse_values = {
            "GL0": final_inv_results.get("mse_gl0", 0.0),
            "GL1": final_inv_results.get("mse_gl1", 0.0),
            "GL2": final_inv_results.get("mse_gl2", 0.0),
            "GL3": final_inv_results.get("mse_gl3", 0.0),
            "Adaptive": final_inv_results.get("mse_adaptive", 0.0),
        }

        # Find the attacker with the lowest MSE (strongest attacker)
        # Lower MSE = better reconstruction = stronger attacker
        if mse_values and any(v > 0 for v in mse_values.values()):
            strongest_attacker = min(mse_values, key=mse_values.get)
            strongest_attack_mse = mse_values[strongest_attacker]
        else:
            strongest_attacker = "None"
            strongest_attack_mse = 0.0

        metrics = {
            "history": convert_to_serializable(self.history),
            "best_val_acc": float(self.best_val_acc) if isinstance(self.best_val_acc, (np.float32, np.float64)) else self.best_val_acc,
            "best_epoch": int(self.best_epoch) if isinstance(self.best_epoch, (np.int32, np.int64)) else self.best_epoch,
            "best_val_f1": float(self.best_val_f1) if isinstance(self.best_val_f1, (np.float32, np.float64)) else self.best_val_f1,
            "best_f1_epoch": int(self.best_f1_epoch) if isinstance(self.best_f1_epoch, (np.int32, np.int64)) else self.best_f1_epoch,
            "best_privacy_mse": float(self.best_privacy_mse) if isinstance(self.best_privacy_mse, (np.float32, np.float64)) else self.best_privacy_mse,
            "best_privacy_epoch": int(self.best_privacy_epoch) if isinstance(self.best_privacy_epoch, (np.int32, np.int64)) else self.best_privacy_epoch,
            "best_privacy_utility": float(self.best_privacy_utility) if isinstance(self.best_privacy_utility, (np.float32, np.float64)) else self.best_privacy_utility,
            "best_privacy_utility_epoch": int(self.best_privacy_utility_epoch) if isinstance(self.best_privacy_utility_epoch, (np.int32, np.int64)) else self.best_privacy_utility_epoch,
            "final_privacy_utility": float(final_privacy_utility) if isinstance(final_privacy_utility, (np.float32, np.float64)) else final_privacy_utility,
            "best_strongest_attacker": best_strongest_attacker,
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
                "strongest_attack_mse": float(strongest_attack_mse),
                "strongest_attacker": strongest_attacker,
            },
            "cfg": self.cfg.to_dict(),
            "adversary_level": self.cfg.adversary_level,
            "adversary_name": get_adversary_name(self.cfg.adversary_level),
            "use_pretrain": self.use_pretrain,
            "expert_trained": self.expert_trained,
            "warmup_epochs": int(self.warmup_epochs),
            "main_epochs": int(self.cfg.main_epochs),
            # Decoder calibration summary
            "decoder_calibration": {
                "baseline": float(self.decoder_loss_baseline) if self.decoder_loss_baseline is not None else None,
                "threshold": float(self.decoder_loss_threshold) if self.decoder_loss_threshold is not None else None,
                "total_skips": int(self.decoder_loss_skip_count),
                "calibration_updates": len(self.decoder_loss_calibration),
                "divergence_multiplier": self.DIVERGENCE_MULTIPLIER,
            },
        }

        with open(metrics_path, "w") as f:
            json.dump(convert_to_serializable(metrics), f, indent=2)

        print(f"Metrics saved: {metrics_path}")

    def get_metrics(self) -> Dict[str, List[float]]:
        """Return training history."""
        return self.history


def create_ressfl_trainer(cfg: ExperimentConfig,
                          train_loaders: List[DataLoader],
                          val_loaders: List[DataLoader]) -> ResSFLTrainer:
    """Factory for creating ResSFLTrainer from config and dataloaders."""
    return ResSFLTrainer(cfg, train_loaders, val_loaders, cfg.device)