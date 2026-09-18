# asfl/trainer_asfl.py

import copy
import os
import json
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score, recall_score

from config import ExperimentConfig, save_config
from models import build_model
from attacks import build_decoder, PGDAttack, get_adversary_name
from evaluation import (
    compute_accuracy, compute_precision, compute_recall, compute_f1, compute_auprc,
    compute_mse,
    ClassificationEvaluator, PrivacyEvaluator,
    evaluate_trained_inversion_hierarchy, save_evaluation_results
)
from experiments import register_trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TemporaryASFLModelForPGD(nn.Module):
    def __init__(self, backbone, Pu, Pp, server_classifier,
                 has_privacy=True, use_single_projection=False):
        super().__init__()
        self.backbone = backbone
        self.Pu = Pu
        self.Pp = Pp
        self.server_classifier = server_classifier
        self.has_privacy = has_privacy
        self.use_single_projection = use_single_projection

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        if self.use_single_projection:
            zu = self.Pu(h)
            zp = zu
        else:
            zu = self.Pu(h)
            zp = self.Pp(h) if (self.has_privacy and self.Pp is not None) else torch.zeros_like(zu)
        logits = self.server_classifier(zu)
        return {"logits": logits, "zu": zu, "zp": zp, "h": h}


@register_trainer('asfl')
class ASFLTrainer:
    VALID_ABLATIONS = ["full", "no_dual", "no_privacy", "no_evasion"]

    def __init__(self, cfg: ExperimentConfig, train_loaders: List[DataLoader],
                 val_loaders: List[DataLoader], device: str = "cuda",
                 ablation: str = "full"):
        set_seed(cfg.seed)

        self.cfg = cfg
        self.device = device if torch.cuda.is_available() else "cpu"
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders

        self.num_clients = len(train_loaders)
        self.ablation = ablation
        self.verbose = True
        self.num_classes = cfg.get_dataset_num_classes()

        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown ablation: {ablation}")

        self.has_dual_projection = True
        self.has_privacy_projection = True
        self.has_evasion = True
        self.use_single_projection = False

        if ablation == "no_dual":
            self.has_dual_projection = False
            self.use_single_projection = True
            self.has_privacy_projection = False
        elif ablation == "no_privacy":
            self.has_privacy_projection = False
        elif ablation == "no_evasion":
            self.has_evasion = False

        self.valid_train_clients = [i for i in range(self.num_clients) if len(self.train_loaders[i].dataset) > 0]
        self.valid_val_clients = [i for i in range(self.num_clients) if len(self.val_loaders[i].dataset) > 0]

        self.model = build_model(cfg).to(self.device)

        self.backbone_template = self.model.client.backbone
        self.Pu_template = self.model.client.Pu
        self.Pp_template = self.model.client.Pp

        if self.Pp_template is not None:
            if self.ablation != "no_privacy" and self.ablation != "no_dual":
                self.has_privacy_projection = True
            else:
                self.has_privacy_projection = False
        else:
            self.has_privacy_projection = False

        self.classifier_template = self.model.server

        self.client_backbones = []
        self.client_Pu = []
        self.client_Pp = []

        for _ in range(self.num_clients):
            backbone = copy.deepcopy(self.backbone_template).to(self.device)
            Pu = copy.deepcopy(self.Pu_template).to(self.device)
            if (self.Pp_template is not None and self.has_privacy_projection and not self.use_single_projection):
                Pp = copy.deepcopy(self.Pp_template).to(self.device)
            else:
                Pp = None
            self.client_backbones.append(backbone)
            self.client_Pu.append(Pu)
            self.client_Pp.append(Pp)

        self.server_classifier = copy.deepcopy(self.classifier_template).to(self.device)

        self.decoder = build_decoder(4, cfg.get_bottleneck_dim(), cfg.get_dataset_input_dim()).to(self.device)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=cfg.learning_rate * 0.1, weight_decay=cfg.weight_decay)

        self.client_optimizers = []
        for i in range(self.num_clients):
            params = list(self.client_backbones[i].parameters()) + list(self.client_Pu[i].parameters())
            if self.has_privacy_projection and not self.use_single_projection and self.client_Pp[i] is not None:
                params += list(self.client_Pp[i].parameters())
            self.client_optimizers.append(torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay))

        self.server_optimizer = torch.optim.Adam(self.server_classifier.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        # Class-Weighted CrossEntropyLoss - computed from training data only 
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
                    print(f" Class counts (training only): {self.class_counts}")
            else:
                self.class_weights = None
                self.class_counts = None
                self.classification_criterion = nn.CrossEntropyLoss()
                if self.verbose:
                    print(" No training labels found. Using standard CE.")
        else:
            self.class_weights = None
            self.classification_criterion = nn.CrossEntropyLoss()
            if self.verbose:
                print(" Using standard CE")

            all_labels = []
            for loader in train_loaders:
                for _, y in loader:
                    all_labels.extend(y.tolist())
            self.class_counts = np.bincount(all_labels, minlength=self.num_classes) if all_labels else None

        self.reconstruction_criterion = nn.MSELoss()

        self.client_samplers = [None] * self.num_clients

        self.scheduler = None
        self.client_schedulers = []

        # ASFL calibrated divergence monitoring 
        self.rec_loss_baseline = None
        self.rec_loss_calibration = []
        self.CALIBRATION_STEPS = 50
        self.rec_loss_warning_count = 0

        self.history = {
            "train_loss": [], "train_acc": [],
            "train_precision": [], "train_recall": [], "train_f1": [], "train_auprc": [],
            "val_loss": [], "val_acc": [],
            "val_precision": [], "val_recall": [], "val_f1": [], "val_auprc": [],
            "train_rec_loss": [], "val_rec_loss": [],
            "train_adv_loss": [], "val_adv_loss": [],
            "train_privacy_penalty": [], "val_privacy_penalty": [],
            "train_evasion_f1": [], "val_evasion_f1": [],
            "val_mse_l0": [], "val_mse_best": [],
            "val_macro_f1": [],
            "val_weighted_f1": [],
            "val_balanced_acc": [],
            "val_per_class_recall": [],
            "lr_history": [],
            "rec_loss_baseline": [],
            "rec_loss_warning_count": [],
 
        }

        self.csv_log = []

        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_privacy_mse = -float('inf')
        self.best_macro_f1 = -float('inf')
        self.best_macro_epoch = 0

        self.final_metrics = {
            "final_loss": 0.0,
            "final_acc": 0.0,
            "final_f1": 0.0,
            "final_privacy_mse": 0.0,
            "final_evasion_f1": 0.0,
            "final_macro_f1": 0.0,
            "final_weighted_f1": 0.0,
            "final_balanced_acc": 0.0,
        }

        save_config(cfg)

    def _get_client_weights(self, client_indices: List[int]) -> List[float]:
        sample_sizes = [len(self.train_loaders[i].dataset) for i in client_indices]
        total_samples = sum(sample_sizes)
        return [n / total_samples for n in sample_sizes]

    def _federated_average_separate(self, backbone_states, Pu_states, Pp_states=None, client_indices=None):
        if not backbone_states or not Pu_states:
            return {}, {}, None

        if client_indices is None:
            client_indices = list(range(len(backbone_states)))

        weights = self._get_client_weights(client_indices)

        avg_backbone = {}
        for key in backbone_states[0].keys():
            avg_backbone[key] = weights[0] * backbone_states[0][key]
            for i in range(1, len(backbone_states)):
                avg_backbone[key] += weights[i] * backbone_states[i][key]

        avg_Pu = {}
        for key in Pu_states[0].keys():
            avg_Pu[key] = weights[0] * Pu_states[0][key]
            for i in range(1, len(Pu_states)):
                avg_Pu[key] += weights[i] * Pu_states[i][key]

        avg_Pp = None
        if Pp_states and len(Pp_states) > 0:
            valid_indices = []
            valid_weights = []
            valid_Pp_states = []

            for i, state in enumerate(Pp_states):
                if state is not None:
                    valid_indices.append(i)
                    valid_weights.append(weights[i])
                    valid_Pp_states.append(state)

            if valid_Pp_states:
                total_weight = sum(valid_weights)
                normalized_weights = [w / total_weight for w in valid_weights]

                avg_Pp = {}
                for key in valid_Pp_states[0].keys():
                    avg_Pp[key] = normalized_weights[0] * valid_Pp_states[0][key]
                    for j in range(1, len(valid_Pp_states)):
                        avg_Pp[key] += normalized_weights[j] * valid_Pp_states[j][key]

        return avg_backbone, avg_Pu, avg_Pp

    def _client_forward(self, client_idx, x):
        h = self.client_backbones[client_idx](x)
        if self.use_single_projection:
            zu = self.client_Pu[client_idx](h)
            return zu, zu, h

        zu = self.client_Pu[client_idx](h)
        zp = self.client_Pp[client_idx](h) if (self.has_privacy_projection and self.client_Pp[client_idx] is not None) else None
        return zu, zp, h

    def _server_forward(self, zu):
        return self.server_classifier(zu)

    def _compute_reconstruction_loss(self, zu, x):
        return self.reconstruction_criterion(self.decoder(zu), x)

    def _compute_privacy_penalty(self, zp):
        if zp is None or not self.has_privacy_projection:
            return torch.tensor(0.0, device=self.device)
        return (zp ** 2).mean()

    def _generate_pgd_attack(self, client_idx, x, y, eval_mode: bool = False):
        Pp = self.client_Pp[client_idx] if (self.has_privacy_projection and not self.use_single_projection) else None

        temp_model = TemporaryASFLModelForPGD(
            backbone=self.client_backbones[client_idx],
            Pu=self.client_Pu[client_idx],
            Pp=Pp,
            server_classifier=self.server_classifier,
            has_privacy=self.has_privacy_projection,
            use_single_projection=self.use_single_projection
        ).to(self.device)

        steps = self.cfg.pgd_eval_steps if eval_mode else self.cfg.pgd_steps

        pgd = PGDAttack(
            model=temp_model,
            epsilon=self.cfg.pgd_epsilon,
            steps=steps,
            alpha=self.cfg.pgd_alpha,
            device=self.device
        )

        x_adv, _ = pgd.attack(x, y)
        return x_adv

    def _freeze_current_client(self, client_idx, freeze=True):
        modules = [self.client_backbones[client_idx], self.client_Pu[client_idx]]
        if self.has_privacy_projection and not self.use_single_projection and self.client_Pp[client_idx] is not None:
            modules.append(self.client_Pp[client_idx])
        for module in modules:
            for p in module.parameters():
                p.requires_grad = not freeze

    # train_client with calibrated divergence monitoring
    def train_client(self, client_idx, local_epochs=1, lambda_1=1.0,
                     lambda_2=1.0, lambda_3=0.01, warmup=False):
        if len(self.train_loaders[client_idx].dataset) == 0:
            return (self.client_backbones[client_idx].state_dict(),
                    self.client_Pu[client_idx].state_dict(),
                    self.client_Pp[client_idx].state_dict() if (self.has_privacy_projection and not self.use_single_projection) else None)

        loader = self.train_loaders[client_idx]

        self.client_backbones[client_idx].train()
        self.client_Pu[client_idx].train()
        if self.has_privacy_projection and not self.use_single_projection and self.client_Pp[client_idx] is not None:
            self.client_Pp[client_idx].train()
        self.server_classifier.train()
        self.decoder.train()

        for _ in range(local_epochs):
            for batch_idx, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)

                zu, zp, _ = self._client_forward(client_idx, x)
                logits = self._server_forward(zu)

                ce_loss = self.classification_criterion(logits, y)
                rec_loss = self._compute_reconstruction_loss(zu, x)
                
                # Calibrated divergence monitoring 
                loss_val = rec_loss.item()
                
                if self.rec_loss_baseline is None:
                    self.rec_loss_calibration.append(loss_val)
                    
                    if len(self.rec_loss_calibration) >= self.CALIBRATION_STEPS:
                        self.rec_loss_baseline = float(
                            np.median(self.rec_loss_calibration)
                        )
                        
                        if self.verbose:
                            print(
                                f"  [ASFL calibration] "
                                f"baseline={self.rec_loss_baseline:.4f} "
                                f"(based on {len(self.rec_loss_calibration)} batches)"
                            )
                
                elif loss_val > self.rec_loss_baseline * 20.0:
                    self.rec_loss_warning_count += 1
                    
                    if self.verbose:
                        print(
                            f"  [ASFL divergence warning] "
                            f"loss={loss_val:.2f}, "
                            f"baseline={self.rec_loss_baseline:.2f}, "
                            f"ratio={loss_val / self.rec_loss_baseline:.1f}x"
                        )

                privacy_penalty = self._compute_privacy_penalty(zp)

                if not warmup and self.has_evasion and lambda_2 > 0:
                    x_adv = self._generate_pgd_attack(client_idx, x, y)
                    adv_zu, _, _ = self._client_forward(client_idx, x_adv)
                    adv_logits = self._server_forward(adv_zu)
                    adv_ce_loss = self.classification_criterion(adv_logits, y)
                else:
                    adv_ce_loss = torch.tensor(0.0, device=self.device)

                if warmup:
                    total_loss = ce_loss
                else:
                    total_loss = ce_loss
                    if self.has_dual_projection:
                        total_loss = total_loss - lambda_1 * rec_loss
                    if self.has_evasion:
                        total_loss = total_loss + lambda_2 * adv_ce_loss
                    if self.has_privacy_projection:
                        total_loss = total_loss + lambda_3 * privacy_penalty

                self.client_optimizers[client_idx].zero_grad()
                self.server_optimizer.zero_grad()

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.client_backbones[client_idx].parameters()) +
                    list(self.client_Pu[client_idx].parameters()) +
                    (list(self.client_Pp[client_idx].parameters()) if self.has_privacy_projection and not self.use_single_projection and self.client_Pp[client_idx] is not None else []),
                    max_norm=1.0
                )

                self.client_optimizers[client_idx].step()
                self.server_optimizer.step()

                self.decoder_optimizer.zero_grad()
                decoder_rec_loss = self._compute_reconstruction_loss(zu.detach(), x.detach())
                decoder_rec_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                self.decoder_optimizer.step()

        return (self.client_backbones[client_idx].state_dict(),
                self.client_Pu[client_idx].state_dict(),
                self.client_Pp[client_idx].state_dict() if (self.has_privacy_projection and not self.use_single_projection) else None)

    def _compute_evasion_f1(self, client_idx, loader):
        if len(loader.dataset) == 0:
            return 0.0

        self.client_backbones[client_idx].eval()
        self.client_Pu[client_idx].eval()
        if self.has_privacy_projection and not self.use_single_projection and self.client_Pp[client_idx] is not None:
            self.client_Pp[client_idx].eval()
        self.server_classifier.eval()
        self.decoder.eval()

        all_y_true = []
        all_y_pred_adv = []

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            with torch.enable_grad():
                x_adv = self._generate_pgd_attack(client_idx, x, y, eval_mode=True)

            with torch.no_grad():
                adv_zu, _, _ = self._client_forward(client_idx, x_adv)
                adv_logits = self._server_forward(adv_zu)
                adv_preds = adv_logits.argmax(dim=1)

                all_y_true.extend(y.cpu().numpy())
                all_y_pred_adv.extend(adv_preds.cpu().numpy())

        if len(all_y_true) == 0:
            return 0.0

        all_y_true = np.array(all_y_true)
        all_y_pred_adv = np.array(all_y_pred_adv)

        evasion_f1 = compute_f1(all_y_true, all_y_pred_adv, average='weighted')
        return evasion_f1

    def evaluate_client_with_metrics(self, client_idx, loader,
                                     compute_reconstruction=False,
                                     compute_robustness=False,
                                     compute_metrics=False):
        if len(loader.dataset) == 0:
            return 0.0, 0.0, None, None, None, {}, 0.0

        self.client_backbones[client_idx].eval()
        self.client_Pu[client_idx].eval()
        if self.has_privacy_projection and not self.use_single_projection and self.client_Pp[client_idx] is not None:
            self.client_Pp[client_idx].eval()
        self.server_classifier.eval()
        self.decoder.eval()

        total_loss, total_correct, total_samples = 0.0, 0, 0
        total_rec_loss, total_adv_loss, total_privacy_penalty = 0.0, 0.0, 0.0
        rec_count, adv_count, privacy_count = 0, 0, 0
        all_y_true, all_y_pred, all_y_prob = [], [], []

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                zu, zp, _ = self._client_forward(client_idx, x)
                logits = self._server_forward(zu)

                loss = self.classification_criterion(logits, y)
                total_loss += loss.item() * x.size(0)

                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0)

                if compute_metrics:
                    all_y_true.extend(y.cpu().numpy())
                    all_y_pred.extend(preds.cpu().numpy())
                    all_y_prob.extend(torch.softmax(logits, dim=1).cpu().numpy())

                if compute_reconstruction:
                    rec_loss = self._compute_reconstruction_loss(zu, x)
                    total_rec_loss += rec_loss.item() * x.size(0)
                    rec_count += x.size(0)

                if self.has_privacy_projection:
                    privacy_penalty = self._compute_privacy_penalty(zp)
                    total_privacy_penalty += privacy_penalty.item() * x.size(0)
                    privacy_count += x.size(0)

        evasion_f1 = 0.0
        if compute_robustness:
            evasion_f1 = self._compute_evasion_f1(client_idx, loader)

        if total_samples == 0:
            return 0.0, 0.0, None, None, None, {}, 0.0

        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        avg_rec_loss = total_rec_loss / rec_count if rec_count > 0 else None
        avg_adv_loss = total_adv_loss / adv_count if adv_count > 0 else None
        avg_privacy_penalty = total_privacy_penalty / privacy_count if privacy_count > 0 else None

        metrics = {}
        if compute_metrics and len(all_y_true) > 0:
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_y_prob = np.array(all_y_prob) if all_y_prob else None

            metrics['accuracy'] = compute_accuracy(all_y_true, all_y_pred)
            metrics['precision'] = compute_precision(all_y_true, all_y_pred, average='weighted')
            metrics['recall'] = compute_recall(all_y_true, all_y_pred, average='weighted')
            metrics['f1'] = compute_f1(all_y_true, all_y_pred, average='weighted')

            # Diagnostic metrics (not reported in paper tables)
            metrics['macro_f1'] = f1_score(all_y_true, all_y_pred, average='macro')
            metrics['weighted_f1'] = f1_score(all_y_true, all_y_pred, average='weighted')
            try:
                metrics['balanced_acc'] = balanced_accuracy_score(all_y_true, all_y_pred)
            except:
                metrics['balanced_acc'] = 0.0
            metrics['per_class_recall'] = recall_score(all_y_true, all_y_pred, average=None, labels=list(range(self.num_classes)))
            metrics['confusion_matrix'] = confusion_matrix(all_y_true, all_y_pred, labels=list(range(self.num_classes)))
            metrics['prediction_distribution'] = dict(Counter(all_y_pred))

            if all_y_prob is not None:
                metrics['auprc'] = compute_auprc(all_y_true, all_y_prob)
            else:
                metrics['auprc'] = 0.0

        return avg_loss, avg_acc, avg_rec_loss, avg_adv_loss, avg_privacy_penalty, metrics, evasion_f1

    def evaluate_client(self, client_idx, loader, compute_reconstruction=False, compute_robustness=False):
        loss, acc, rec_loss, adv_loss, privacy_penalty, _, _ = self.evaluate_client_with_metrics(
            client_idx, loader, compute_reconstruction, compute_robustness, compute_metrics=False
        )
        return loss, acc, rec_loss, adv_loss, privacy_penalty

    def verify_pp_training(self, client_idx: int = 0, local_epochs: int = 1) -> Dict[str, bool]:
        """Manual verification utility - call only when debugging."""
        if not (self.has_privacy_projection and not self.use_single_projection and self.client_Pp[client_idx] is not None):
            print("Pp not available for this ablation. Skipping verification.")
            return {}

        before_state = copy.deepcopy(self.client_Pp[client_idx].state_dict())

        if len(self.valid_train_clients) > 0:
            self.train_client(client_idx, local_epochs=local_epochs, warmup=False)
        else:
            print("No valid training clients. Skipping verification.")
            return {}

        after_state = self.client_Pp[client_idx].state_dict()

        results = {}
        for key in before_state.keys():
            changed = not torch.equal(before_state[key], after_state[key])
            results[key] = changed
            print(f"  {key}: {'CHANGED' if changed else 'UNCHANGED'}")

        any_changed = any(results.values())
        print(f"\nPp parameters changed: {any_changed}")
        if any_changed:
            print("  Pp is training!")
        else:
            print("  WARNING: Pp parameters did not change - gradient path may still have issues.")

        return results

    def verify_gradient_flow(self, client_idx=0, batch_size=4):
        """Manual verification utility - call only when debugging."""
        self.client_backbones[client_idx].train()
        self.client_Pu[client_idx].train()
        if self.has_privacy_projection and not self.use_single_projection and self.client_Pp[client_idx] is not None:
            self.client_Pp[client_idx].train()
        self.server_classifier.train()
        self.decoder.train()

        loader = self.train_loaders[client_idx]
        x, y = next(iter(loader))
        x, y = x[:batch_size].to(self.device), y[:batch_size].to(self.device)

        zu, zp, _ = self._client_forward(client_idx, x)
        logits = self._server_forward(zu)

        ce_loss = self.classification_criterion(logits, y)
        rec_loss = self._compute_reconstruction_loss(zu, x)
        privacy_penalty = self._compute_privacy_penalty(zp)

        if self.has_evasion:
            x_adv = self._generate_pgd_attack(client_idx, x, y)
            adv_zu, _, _ = self._client_forward(client_idx, x_adv)
            adv_logits = self._server_forward(adv_zu)
            adv_ce_loss = self.classification_criterion(adv_logits, y)
        else:
            adv_ce_loss = torch.tensor(0.0, device=self.device)

        total_loss = ce_loss
        if self.has_dual_projection:
            total_loss = total_loss - 1.0 * rec_loss
        if self.has_evasion:
            total_loss = total_loss + 1.0 * adv_ce_loss
        if self.has_privacy_projection:
            total_loss = total_loss + 0.01 * privacy_penalty

        self.client_optimizers[client_idx].zero_grad()
        self.server_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad(set_to_none=True)

        total_loss.backward()

        backbone_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.client_backbones[client_idx].parameters())
        Pu_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.client_Pu[client_idx].parameters())

        Pp_grad = False
        if self.has_privacy_projection and not self.use_single_projection and self.client_Pp[client_idx] is not None:
            Pp_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.client_Pp[client_idx].parameters())

        classifier_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.server_classifier.parameters())

        result = {
            "backbone_grad": backbone_grad,
            "Pu_grad": Pu_grad,
            "Pp_grad": Pp_grad,
            "classifier_grad": classifier_grad,
        }

        print("\nGradient Flow Verification:")
        for name, status in result.items():
            if name == "Pp_grad" and not (self.has_privacy_projection and not self.use_single_projection and self.client_Pp[client_idx] is not None):
                print(f" {name}: SKIPPED (not used in this ablation)")
            else:
                print(f" {name}: {'OK' if status else 'MISSING'}")

        assert backbone_grad, "Backbone gradients missing"
        assert Pu_grad, "Pu gradients missing"
        if self.has_privacy_projection and not self.use_single_projection and self.client_Pp[client_idx] is not None:
            assert Pp_grad, "Pp gradients missing - check backward pass"

        return result

    def train_round(self, round_idx, lambda_1=1.0, lambda_2=1.0, lambda_3=0.01,
                    warmup=False, compute_robustness=False):
        client_backbone_states, client_Pu_states, client_Pp_states = [], [], []
        client_indices = []
        client_train_losses, client_train_accs = [], []
        client_train_rec_losses, client_train_adv_losses, client_train_privacy_penalties = [], [], []
        client_train_precisions, client_train_recalls, client_train_f1s, client_train_auprcs = [], [], [], []
        client_train_evasion_f1s = []

        val_losses, val_accs = [], []
        val_rec_losses, val_adv_losses, val_privacy_penalties = [], [], []
        val_evasion_f1s = []
        val_macro_f1s = []
        val_weighted_f1s = []
        val_balanced_accs = []
        val_per_class_recalls = []
        val_precisions, val_recalls, val_f1s, val_auprcs = [], [], [], []

        # Train clients 
        for client_idx in self.valid_train_clients:
            backbone_state, Pu_state, Pp_state = self.train_client(
                client_idx, 1, lambda_1, lambda_2, lambda_3, warmup
            )
            client_backbone_states.append(backbone_state)
            client_Pu_states.append(Pu_state)
            client_Pp_states.append(Pp_state)
            client_indices.append(client_idx)

            loss, acc, rec_loss, adv_loss, privacy_penalty, metrics, evasion_f1 = self.evaluate_client_with_metrics(
                client_idx, self.train_loaders[client_idx], compute_reconstruction=True,
                compute_robustness=False, compute_metrics=True
            )

            client_train_losses.append(loss)
            client_train_accs.append(acc)
            client_train_evasion_f1s.append(evasion_f1)

            if rec_loss is not None:
                client_train_rec_losses.append(rec_loss)
            if adv_loss is not None:
                client_train_adv_losses.append(adv_loss)
            if privacy_penalty is not None:
                client_train_privacy_penalties.append(privacy_penalty)

            if metrics:
                client_train_precisions.append(metrics.get('precision', 0.0))
                client_train_recalls.append(metrics.get('recall', 0.0))
                client_train_f1s.append(metrics.get('f1', 0.0))
                client_train_auprcs.append(metrics.get('auprc', 0.0))

        # Federated averaging 
        avg_backbone, avg_Pu, avg_Pp = self._federated_average_separate(
            client_backbone_states, client_Pu_states, client_Pp_states, client_indices
        )

        if avg_backbone:
            for client_idx in range(self.num_clients):
                self.client_backbones[client_idx].load_state_dict(avg_backbone)

        if avg_Pu:
            for client_idx in range(self.num_clients):
                self.client_Pu[client_idx].load_state_dict(avg_Pu)

        if avg_Pp and self.has_privacy_projection and not self.use_single_projection:
            for client_idx in range(self.num_clients):
                if self.client_Pp[client_idx] is not None:
                    self.client_Pp[client_idx].load_state_dict(avg_Pp)

        for client_idx in self.valid_val_clients:
            loss, acc, rec_loss, adv_loss, privacy_penalty, metrics, evasion_f1 = self.evaluate_client_with_metrics(
                client_idx, self.val_loaders[client_idx], compute_reconstruction=True,
                compute_robustness=compute_robustness, compute_metrics=True
            )

            val_losses.append(loss)
            val_accs.append(acc)
            val_evasion_f1s.append(evasion_f1)

            if rec_loss is not None:
                val_rec_losses.append(rec_loss)
            if adv_loss is not None:
                val_adv_losses.append(adv_loss)
            if privacy_penalty is not None:
                val_privacy_penalties.append(privacy_penalty)

            if metrics:
                val_precisions.append(metrics.get('precision', 0.0))
                val_recalls.append(metrics.get('recall', 0.0))
                val_f1s.append(metrics.get('f1', 0.0))
                val_auprcs.append(metrics.get('auprc', 0.0))
                val_macro_f1s.append(metrics.get('macro_f1', 0.0))
                val_weighted_f1s.append(metrics.get('weighted_f1', 0.0))
                val_balanced_accs.append(metrics.get('balanced_acc', 0.0))
                per_class_recall = metrics.get('per_class_recall', [0.0] * self.num_classes)
                val_per_class_recalls.append(per_class_recall)

        return {
            "train_loss": np.mean(client_train_losses) if client_train_losses else 0.0,
            "train_acc": np.mean(client_train_accs) if client_train_accs else 0.0,
            "train_rec_loss": np.mean(client_train_rec_losses) if client_train_rec_losses else 0.0,
            "train_adv_loss": np.mean(client_train_adv_losses) if client_train_adv_losses else 0.0,
            "train_privacy_penalty": np.mean(client_train_privacy_penalties) if client_train_privacy_penalties else 0.0,
            "train_precision": np.mean(client_train_precisions) if client_train_precisions else 0.0,
            "train_recall": np.mean(client_train_recalls) if client_train_recalls else 0.0,
            "train_f1": np.mean(client_train_f1s) if client_train_f1s else 0.0,
            "train_auprc": np.mean(client_train_auprcs) if client_train_auprcs else 0.0,
            "train_evasion_f1": np.mean(client_train_evasion_f1s) if client_train_evasion_f1s else 0.0,
            "val_loss": np.mean(val_losses) if val_losses else 0.0,
            "val_acc": np.mean(val_accs) if val_accs else 0.0,
            "val_rec_loss": np.mean(val_rec_losses) if val_rec_losses else 0.0,
            "val_adv_loss": np.mean(val_adv_losses) if val_adv_losses else 0.0,
            "val_privacy_penalty": np.mean(val_privacy_penalties) if val_privacy_penalties else 0.0,
            "val_precision": np.mean(val_precisions) if val_precisions else 0.0,
            "val_recall": np.mean(val_recalls) if val_recalls else 0.0,
            "val_f1": np.mean(val_f1s) if val_f1s else 0.0,
            "val_auprc": np.mean(val_auprcs) if val_auprcs else 0.0,
            "val_evasion_f1": np.mean(val_evasion_f1s) if val_evasion_f1s else 0.0,
            "val_macro_f1": np.mean(val_macro_f1s) if val_macro_f1s else 0.0,
            "val_weighted_f1": np.mean(val_weighted_f1s) if val_weighted_f1s else 0.0,
            "val_balanced_acc": np.mean(val_balanced_accs) if val_balanced_accs else 0.0,
            "val_per_class_recall": np.mean(val_per_class_recalls, axis=0) if val_per_class_recalls else [0.0] * self.num_classes,
        }
    def evaluate_inversion_attack(self):
    
        # Preserve RNG state so inversion evaluation does not
        # perturb the main training trajectory.
        torch_state = torch.get_rng_state()
        np_state = np.random.get_state()
    
        cuda_state = None
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
    
        try:
            all_zu, all_x = [], []
    
            for client_idx in self.valid_val_clients:
                loader = self.val_loaders[client_idx]
    
                self.client_backbones[client_idx].eval()
                self.client_Pu[client_idx].eval()
    
                with torch.no_grad():
                    for x, y in loader:
                        x = x.to(self.device)
                        zu, _, _ = self._client_forward(client_idx, x)
                        all_zu.append(zu.cpu())
                        all_x.append(x.cpu())
    
            if not all_zu:
                return {
                    "mse_gl0": 0.0,
                    "mse_gl1": 0.0,
                    "mse_gl2": 0.0,
                    "mse_gl3": 0.0,
                    "mse_adaptive": 0.0,
                    "best_mse": 0.0,
                    "best_attacker": "None",
                    "inversion_available": False,
                }
    
            zu = torch.cat(all_zu, dim=0)
            x = torch.cat(all_x, dim=0)
    
            split_idx = int(0.8 * len(zu))
    
            results, _, _ = evaluate_trained_inversion_hierarchy(
                cfg=self.cfg,
                zu_train=zu[:split_idx],
                x_train=x[:split_idx],
                zu_test=zu[split_idx:],
                x_test=x[split_idx:],
                device=self.device,
                attack_epochs=50,
                learning_rate=1e-3,
                batch_size=64,
                verbose=False,
                include_adaptive=True,
                seed=self.cfg.seed,
            )
    
            results["inversion_available"] = True
            return results
    
        finally:
            torch.set_rng_state(torch_state)
            np.random.set_state(np_state)
    
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)   

    def verify_reconstruction_hierarchy(self):
        """
        Diagnostic: report MSE per inversion-adversary variant (GL0-GL3 + Adaptive).
    
        Reported privacy MSE = MSE of the strongest attacker (lowest reconstruction error).
        """
        results = self.evaluate_inversion_attack()
    
        if not results.get("inversion_available", True):
            print(" Inversion evaluation not available for this ablation.")
            return {}
    
        mse_values = {
            "GL0": results.get("mse_gl0", 0.0),
            "GL1": results.get("mse_gl1", 0.0),
            "GL2": results.get("mse_gl2", 0.0),
            "GL3": results.get("mse_gl3", 0.0),
            "Adaptive": results.get("mse_adaptive", 0.0),
        }
    
        levels = ["GL0", "GL1", "GL2", "GL3", "Adaptive"]
        strongest = min(mse_values, key=mse_values.get)
        strongest_mse = mse_values[strongest]
        reported_privacy_mse = results.get("best_mse", strongest_mse)
    
        print("\nInversion-Adversary Hierarchy Evaluation:")
        for level in levels:
            marker = "  <-- strongest" if level == strongest else ""
            print(f"  {level:>8}: {mse_values[level]:.4f}{marker}")
        print(f"  Reported Privacy MSE (strongest attacker): {reported_privacy_mse:.4f}")
    
        return mse_values
  
    def _save_csv_log(self):
        """
        Save the complete training log to CSV with timestamp.
        Now writes all rows at once instead of incremental overwrite.
        """
        import time
        output_dir = self.cfg.get_output_path()
        
        # Add timestamp to filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, "train_log.csv")
    
        # Write the entire csv_log to file (overwrites each time with complete data)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "epoch", "train_loss", "train_acc",
                "train_precision", "train_recall", "train_f1", "train_auprc",
                "train_rec_loss", "train_adv_loss", "train_privacy_penalty",
                "train_evasion_f1",
                "val_loss", "val_acc",
                "val_precision", "val_recall", "val_f1", "val_auprc",
                "val_rec_loss", "val_adv_loss", "val_privacy_penalty",
                "val_evasion_f1",
                "val_mse_l0", "val_mse_best",
                "rec_loss_baseline", "rec_loss_warning_count"
            ])
            
            # Write all rows from csv_log
            if self.csv_log:
                writer.writerows(self.csv_log)

    def generate_final_diagnostics(self):
        """Generate comprehensive final diagnostics using ALL validation clients"""
        all_preds = []
        all_labels = []

        self.server_classifier.eval()
        for backbone in self.client_backbones:
            backbone.eval()
        if hasattr(self, 'client_Pu'):
            for pu in self.client_Pu:
                pu.eval()

        with torch.no_grad():
            for client_idx in self.valid_val_clients:
                for x, y in self.val_loaders[client_idx]:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    h = self.client_backbones[client_idx](x)
                    zu = self.client_Pu[client_idx](h)
                    logits = self.server_classifier(zu)

                    preds = logits.argmax(1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(y.cpu().tolist())

        if len(all_preds) == 0:
            return {
                'prediction_distribution': {},
                'label_distribution': {},
                'unique_classes': 0,
                'majority_class': None,
                'majority_fraction': 0.0,
                'prediction_entropy': 0.0,
                'is_collapsed': True,
                'macro_f1': 0.0,
                'weighted_f1': 0.0,
                'balanced_accuracy': 0.0,
                'per_class_recall': [0.0] * self.num_classes,
                'per_class_f1': [0.0] * self.num_classes,
                'confusion_matrix': [[0] * self.num_classes for _ in range(self.num_classes)],
                'class_counts': self.class_counts.tolist() if self.class_counts is not None else [],
                'class_weights': self.class_weights.cpu().tolist() if self.class_weights is not None else None,
                'dataset': self.cfg.dataset,
                'method': getattr(self.cfg, 'method', 'asfl'),
                'ablation': getattr(self, 'ablation', 'full'),
                'best_macro_f1': float(self.best_macro_f1),
                'best_macro_epoch': self.best_macro_epoch,
            }

        pred_counter = Counter(all_preds)
        label_counter = Counter(all_labels)
        unique_classes = len(pred_counter)
        majority_fraction = max(pred_counter.values()) / len(all_preds) if len(all_preds) > 0 else 0
        is_collapsed = (unique_classes == 1 or majority_fraction > 0.95)

        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        try:
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        except:
            balanced_acc = 0.0

        per_class_recall = recall_score(all_labels, all_preds, average=None, labels=list(range(self.num_classes)))
        per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(self.num_classes)))
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.num_classes)))

        probs = np.array(list(pred_counter.values())) / len(all_preds) if len(all_preds) > 0 else np.array([1.0])
        entropy = -(probs * np.log(probs + 1e-12)).sum()

        diagnostics = {
            'prediction_distribution': dict(pred_counter),
            'label_distribution': dict(label_counter),
            'unique_classes': unique_classes,
            'majority_class': max(pred_counter, key=pred_counter.get) if pred_counter else None,
            'majority_fraction': float(majority_fraction),
            'prediction_entropy': float(entropy),
            'is_collapsed': is_collapsed,
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'balanced_accuracy': float(balanced_acc),
            'per_class_recall': per_class_recall.tolist() if hasattr(per_class_recall, 'tolist') else per_class_recall,
            'per_class_f1': per_class_f1.tolist() if hasattr(per_class_f1, 'tolist') else per_class_f1,
            'confusion_matrix': cm.tolist(),
            'class_counts': self.class_counts.tolist() if self.class_counts is not None else [],
            'class_weights': self.class_weights.cpu().tolist() if self.class_weights is not None else None,
            'dataset': self.cfg.dataset,
            'method': getattr(self.cfg, 'method', 'asfl'),
            'ablation': getattr(self, 'ablation', 'full'),
            'best_macro_f1': float(self.best_macro_f1),
            'best_macro_epoch': self.best_macro_epoch,
        }

        return diagnostics

    #  fit() with calibration reset before main training
    def fit(self):
        lambda_1 = self.cfg.lambda_1
        lambda_2 = self.cfg.lambda_2
        lambda_3 = self.cfg.lambda_3

        warmup_epochs = self.cfg.warmup_epochs
        main_epochs = self.cfg.main_epochs

        print("=" * 60)
        print(f"ASFL Training on {self.cfg.dataset}")
        print(f" Ablation: {self.ablation}")
        print(f" Clients: {self.num_clients}")
        print(f" Warmup epochs: {warmup_epochs}")
        print(f" Main epochs: {main_epochs}")
        print(f" λ1: {lambda_1}, λ2: {lambda_2}, λ3: {lambda_3}")
        print("=" * 60)

        print(f"\n--- Warm-up Phase ({warmup_epochs} epochs) ---")
        for warmup_idx in range(warmup_epochs):
            metrics = self.train_round(
                warmup_idx,
                lambda_1=0.0,
                lambda_2=0.0,
                lambda_3=0.0,
                warmup=True,
                compute_robustness=False
            )

            if (warmup_idx + 1) % self.cfg.log_interval == 0:
                print(f"Warmup Epoch {warmup_idx + 1}/{warmup_epochs}: "
                      f"Train Acc: {metrics['train_acc']:.2f}% | "
                      f"Val Acc: {metrics['val_acc']:.2f}%")

        # Reset calibration state before main training 
        # This ensures ASFL and ResSFL both calibrate during the actual training phase
        self.rec_loss_baseline = None
        self.rec_loss_calibration = []
        self.rec_loss_warning_count = 0
        
        if self.verbose:
            print("\n[ASFL] Calibration state reset before main training")

        print(f"\n--- Main Training ({main_epochs} epochs) ---")
        for epoch_idx in range(1, main_epochs + 1):
            round_idx = warmup_epochs + epoch_idx - 1

            lambda_1_eff = lambda_1 if self.has_dual_projection else 0.0
            lambda_2_eff = lambda_2 if self.has_evasion else 0.0
            lambda_3_eff = lambda_3 if self.has_privacy_projection else 0.0

            compute_robustness = (epoch_idx % 10 == 0) or (epoch_idx == main_epochs)

            metrics = self.train_round(
                round_idx,
                lambda_1_eff,
                lambda_2_eff,
                lambda_3_eff,
                warmup=False,
                compute_robustness=compute_robustness
            )

            if (epoch_idx % 10 == 0) or (epoch_idx == main_epochs):
                inv_results = self.evaluate_inversion_attack()
                val_mse_l0 = inv_results.get("mse_gl0", 0.0)
                val_mse_best = inv_results.get("best_mse", 0.0)
            else:
                val_mse_l0 = self.history["val_mse_l0"][-1] if self.history["val_mse_l0"] else 0.0
                val_mse_best = self.history["val_mse_best"][-1] if self.history["val_mse_best"] else 0.0

            # Primary checkpoint: best validation F1 (reported in paper)
            val_f1 = metrics.get("val_f1", 0.0)
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.save_checkpoint("best_f1")

            # Secondary checkpoint: best validation accuracy
            if metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = metrics["val_acc"]
                self.save_checkpoint("best_acc")

            # Diagnostic checkpoint: best macro F1 (not reported in paper)
            val_macro_f1 = metrics.get("val_macro_f1", 0.0)
            if val_macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = val_macro_f1
                self.best_macro_epoch = epoch_idx
                self.save_checkpoint("best_macro")

            # Privacy checkpoint: best privacy MSE
            if val_mse_best > self.best_privacy_mse:
                self.best_privacy_mse = val_mse_best
                self.save_checkpoint("best_privacy")

            # Log learning rate trajectory
            current_lr = self.server_optimizer.param_groups[0]["lr"]
            self.history["lr_history"].append(current_lr)

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
            self.history["train_adv_loss"].append(metrics.get("train_adv_loss", 0.0))
            self.history["val_adv_loss"].append(metrics.get("val_adv_loss", 0.0))
            self.history["train_privacy_penalty"].append(metrics.get("train_privacy_penalty", 0.0))
            self.history["val_privacy_penalty"].append(metrics.get("val_privacy_penalty", 0.0))
            self.history["train_evasion_f1"].append(metrics.get("train_evasion_f1", 0.0))
            self.history["val_evasion_f1"].append(metrics.get("val_evasion_f1", 0.0))
            self.history["val_mse_l0"].append(val_mse_l0)
            self.history["val_mse_best"].append(val_mse_best)

            # Diagnostic metrics (stored but not reported in paper)
            self.history["val_macro_f1"].append(metrics.get("val_macro_f1", 0.0))
            self.history["val_weighted_f1"].append(metrics.get("val_weighted_f1", 0.0))
            self.history["val_balanced_acc"].append(metrics.get("val_balanced_acc", 0.0))
            self.history["val_per_class_recall"].append(metrics.get("val_per_class_recall", [0.0] * self.num_classes))

            self.history["rec_loss_baseline"].append(
                self.rec_loss_baseline if self.rec_loss_baseline is not None else 0.0
            )
            self.history["rec_loss_warning_count"].append(self.rec_loss_warning_count)

            self.csv_log.append([
                epoch_idx,
                metrics["train_loss"], metrics["train_acc"],
                metrics.get("train_precision", 0.0), metrics.get("train_recall", 0.0),
                metrics.get("train_f1", 0.0), metrics.get("train_auprc", 0.0),
                metrics.get("train_rec_loss", 0.0), metrics.get("train_adv_loss", 0.0),
                metrics.get("train_privacy_penalty", 0.0),
                metrics.get("train_evasion_f1", 0.0),
                metrics["val_loss"], metrics["val_acc"],
                metrics.get("val_precision", 0.0), metrics.get("val_recall", 0.0),
                metrics.get("val_f1", 0.0), metrics.get("val_auprc", 0.0),
                metrics.get("val_rec_loss", 0.0), metrics.get("val_adv_loss", 0.0),
                metrics.get("val_privacy_penalty", 0.0),
                metrics.get("val_evasion_f1", 0.0),
                val_mse_l0, val_mse_best,
                self.rec_loss_baseline if self.rec_loss_baseline is not None else 0.0,
                self.rec_loss_warning_count
            ])

            self.save_checkpoint("last")
            self._save_csv_log()

            if epoch_idx % self.cfg.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch_idx}")

            if epoch_idx % self.cfg.log_interval == 0:
                print(f"Epoch {epoch_idx}/{main_epochs}: "
                      f"Train Acc: {metrics['train_acc']:.2f}% | "
                      f"Train F1: {metrics.get('train_f1', 0.0):.4f} | "
                      f"Val Acc: {metrics['val_acc']:.2f}% | "
                      f"Val F1: {metrics.get('val_f1', 0.0):.4f} | "
                      f"Val Evasion F1: {metrics.get('val_evasion_f1', 0.0):.4f} | "
                      f"MSE(Best): {val_mse_best:.4f} | "
                      f"Best F1: {self.best_val_f1:.4f}")

        if len(self.history["val_loss"]) > 0:
            self.final_metrics["final_loss"] = self.history["val_loss"][-1]
            self.final_metrics["final_acc"] = self.history["val_acc"][-1]
            self.final_metrics["final_f1"] = self.history["val_f1"][-1]
            self.final_metrics["final_privacy_mse"] = self.history["val_mse_best"][-1]
            self.final_metrics["final_evasion_f1"] = self.history["val_evasion_f1"][-1]
            self.final_metrics["final_macro_f1"] = self.history["val_macro_f1"][-1] if self.history["val_macro_f1"] else 0.0
            self.final_metrics["final_weighted_f1"] = self.history["val_weighted_f1"][-1] if self.history["val_weighted_f1"] else 0.0
            self.final_metrics["final_balanced_acc"] = self.history["val_balanced_acc"][-1] if self.history["val_balanced_acc"] else 0.0

        self.save_checkpoint("final")
        self.save_metrics()
        self._save_csv_log()

        print("\n" + "=" * 60)
        print("GENERATING FINAL DIAGNOSTICS")

        print("\n" + "=" * 60)
        print("Final Inversion Attack Evaluation")
        print("=" * 60)
        self.verify_reconstruction_hierarchy()
        print("=" * 60)

        final_diagnostics = self.generate_final_diagnostics()

        output_dir = self.cfg.get_output_path()
        diag_path = os.path.join(output_dir, "final_diagnostics.json")
        with open(diag_path, "w") as f:
            json.dump(final_diagnostics, f, indent=2)

        print(f" Saved to: {diag_path}")
        print("\nASFL Divergence Monitoring Summary:")
        if self.rec_loss_baseline is not None:
            print(f" Calibration baseline: {self.rec_loss_baseline:.4f}")
            print(f" Warning count (20x baseline): {self.rec_loss_warning_count}")
            print(f" Calibration steps: {len(self.rec_loss_calibration)}")
        else:
            print(" Calibration did not complete (insufficient batches)")

        print("\n" + "=" * 60)
        print(f"ASFL Training Complete! (Ablation: {self.ablation})")
        print(f" Warmup epochs: {warmup_epochs}")
        print(f" Main epochs: {main_epochs}")
        print(f" Best validation F1 (reported): {self.best_val_f1:.4f}")
        print(f" Best privacy MSE (reported): {self.best_privacy_mse:.4f}")
        print(f" Final epoch F1 (reported): {self.final_metrics['final_f1']:.4f}")
        print(f" Final epoch Privacy MSE (reported): {self.final_metrics['final_privacy_mse']:.4f}")
        print(f" Final epoch Evasion F1 (reported): {self.final_metrics['final_evasion_f1']:.4f}")
        print("=" * 60)

        return self.history

    def save_checkpoint(self, name="last"):
        checkpoint = {
            "main_epoch": len(self.history["train_loss"]),
            "best_val_acc": self.best_val_acc,
            "best_val_f1": self.best_val_f1,
            "best_privacy_mse": self.best_privacy_mse,
            "best_macro_f1": self.best_macro_f1,
            "best_macro_epoch": self.best_macro_epoch,
            "final_metrics": self.final_metrics,
            "ablation": self.ablation,
            "server_classifier": self.server_classifier.state_dict(),
            "server_optimizer": self.server_optimizer.state_dict(),
            "decoder": self.decoder.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),
            "client_backbones": [bb.state_dict() for bb in self.client_backbones],
            "client_Pu": [pu.state_dict() for pu in self.client_Pu],
            "client_optimizers": [opt.state_dict() for opt in self.client_optimizers],
            "history": self.history,
            "cfg": self.cfg.to_dict(),
            "rec_loss_baseline": self.rec_loss_baseline,
            "rec_loss_calibration": self.rec_loss_calibration,
            "rec_loss_warning_count": self.rec_loss_warning_count,
        }

        if self.has_privacy_projection and not self.use_single_projection:
            checkpoint["client_Pp"] = [pp.state_dict() if pp is not None else None for pp in self.client_Pp]

        output_dir = self.cfg.get_output_path()
        checkpoint_path = os.path.join(output_dir, f"checkpoint_asfl_{name}.pth")
        torch.save(checkpoint, checkpoint_path)

    
    def save_metrics(self):
        """Save metrics to JSON file with timestamp."""
        import time
        output_dir = self.cfg.get_output_path()
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(output_dir, f"metrics_asfl_{timestamp}.json")
    
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
            if isinstance(obj, tuple):
                return tuple(convert_to_serializable(v) for v in obj)
            return obj
    
        final_inv_results = self.evaluate_inversion_attack()
    
        metrics = {
            "history": convert_to_serializable(self.history),
            "best_val_acc": float(self.best_val_acc) if isinstance(self.best_val_acc, (np.float32, np.float64)) else self.best_val_acc,
            "best_val_f1": float(self.best_val_f1) if isinstance(self.best_val_f1, (np.float32, np.float64)) else self.best_val_f1,
            "best_privacy_mse": float(self.best_privacy_mse) if isinstance(self.best_privacy_mse, (np.float32, np.float64)) else self.best_privacy_mse,
            "best_macro_f1": float(self.best_macro_f1) if isinstance(self.best_macro_f1, (np.float32, np.float64)) else self.best_macro_f1,
            "best_macro_epoch": int(self.best_macro_epoch),
            "final_metrics": {
                "final_loss": float(self.final_metrics.get("final_loss", 0.0)),
                "final_acc": float(self.final_metrics.get("final_acc", 0.0)),
                "final_f1": float(self.final_metrics.get("final_f1", 0.0)),
                "final_privacy_mse": float(self.final_metrics.get("final_privacy_mse", 0.0)),
                "final_evasion_f1": float(self.final_metrics.get("final_evasion_f1", 0.0)),
                "final_macro_f1": float(self.final_metrics.get("final_macro_f1", 0.0)),
                "final_weighted_f1": float(self.final_metrics.get("final_weighted_f1", 0.0)),
                "final_balanced_acc": float(self.final_metrics.get("final_balanced_acc", 0.0)),
            },
            "ablation": self.ablation,
            "ablation_design": {
                "dual_projection": self.has_dual_projection,
                "privacy_projection": self.has_privacy_projection,
                "evasion": self.has_evasion,
                "single_projection": self.use_single_projection,
            },
            "final_privacy_metrics": {
                "mse_gl0": float(final_inv_results.get("mse_gl0", 0.0)),
                "mse_gl1": float(final_inv_results.get("mse_gl1", 0.0)),
                "mse_gl2": float(final_inv_results.get("mse_gl2", 0.0)),
                "mse_gl3": float(final_inv_results.get("mse_gl3", 0.0)),
                "mse_adaptive": float(final_inv_results.get("mse_adaptive", 0.0)),
                "privacy_mse": float(final_inv_results.get("privacy_mse", 0.0)),
                "strongest_attacker": final_inv_results.get("strongest_attacker", "None"),
            },
            "cfg": convert_to_serializable(self.cfg.to_dict()),
            "lambda_1": float(self.cfg.lambda_1),
            "lambda_2": float(self.cfg.lambda_2),
            "lambda_3": float(self.cfg.lambda_3),
            "warmup_epochs": int(self.cfg.warmup_epochs),
            "main_epochs": int(self.cfg.main_epochs),
            "class_counts": self.class_counts.tolist() if self.class_counts is not None else [],
            "class_weights": self.class_weights.cpu().tolist() if self.class_weights is not None else None,
            "rec_loss_monitoring": {
                "baseline": float(self.rec_loss_baseline) if self.rec_loss_baseline is not None else None,
                "warning_count": int(self.rec_loss_warning_count),
                "calibration_steps": len(self.rec_loss_calibration),
                "divergence_multiplier": 20.0,
            },
        }
    
        with open(metrics_path, "w") as f:
            json.dump(convert_to_serializable(metrics), f, indent=2)
    
        print(f"Metrics saved: {metrics_path}")

def create_asfl_trainer(cfg: ExperimentConfig,
                        train_loaders: List[DataLoader],
                        val_loaders: List[DataLoader],
                        ablation: str = "full") -> ASFLTrainer:
    return ASFLTrainer(cfg, train_loaders, val_loaders, cfg.device, ablation=ablation)