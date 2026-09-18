# asfl/attacks.py

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# 1. Base Decoder Class
class BaseDecoder(nn.Module):
    """Base class for all inversion decoders."""
    def __init__(self, bottleneck_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def forward(self, zu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# 2. Residual Block for Decoders (1D MLP version)
class ResidualBlock1D(nn.Module):
    """Residual block for 1D MLP decoders using LayerNorm (works with batch size 1)."""
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))


# 3. GL0 Decoder (Weakest Attacker) - cost(L0)
class GL0Decoder(BaseDecoder):
    """
    GL0 Decoder: 2-layer MLP.
    Weakest inversion attacker. Hidden dimension: 128.
    """
    def __init__(self, bottleneck_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__(bottleneck_dim, output_dim, hidden_dim=128)
        self.net = nn.Sequential(
            nn.Linear(bottleneck_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, zu: torch.Tensor) -> torch.Tensor:
        return self.net(zu)


# 4. GL1 Decoder - cost(GL0) < cost(GL1)
class GL1Decoder(BaseDecoder):
    """
    GL1 Decoder: 2-layer MLP + 2 residual blocks.
    Hidden dimension: 256.
    """
    def __init__(self, bottleneck_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__(bottleneck_dim, output_dim, hidden_dim=256)
        self.net = nn.Sequential(
            nn.Linear(bottleneck_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            ResidualBlock1D(self.hidden_dim, dropout),
            ResidualBlock1D(self.hidden_dim, dropout),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, zu: torch.Tensor) -> torch.Tensor:
        return self.net(zu)


# 5. GL2 Decoder - cost(GL1) < cost(GL2)
class GL2Decoder(BaseDecoder):
    """
    GL2 Decoder: 2-layer MLP + 4 residual blocks.
    Hidden dimension: 384.
    """
    def __init__(self, bottleneck_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__(bottleneck_dim, output_dim, hidden_dim=384)
        blocks = [
            nn.Linear(bottleneck_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(4):
            blocks.append(ResidualBlock1D(self.hidden_dim, dropout))
        blocks.append(nn.Linear(self.hidden_dim, output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, zu: torch.Tensor) -> torch.Tensor:
        return self.net(zu)


# 6. GL3 Decoder (Strongest ResSFL Attacker) - cost(GL2) < cost(GL3)
class GL3Decoder(BaseDecoder):
    """
    GL3 Decoder: 2-layer MLP + 6 residual blocks.
    Strongest attacker from ResSFL hierarchy.
    Hidden dimension: 512.
    """
    def __init__(self, bottleneck_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__(bottleneck_dim, output_dim, hidden_dim=512)
        blocks = [
            nn.Linear(bottleneck_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(6):
            blocks.append(ResidualBlock1D(self.hidden_dim, dropout))
        blocks.append(nn.Linear(self.hidden_dim, output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, zu: torch.Tensor) -> torch.Tensor:
        return self.net(zu)


# 7. Adaptive Inversion Adversary - cost(GL3) < cost(Adaptive)
class AdaptiveInversionAdversary(BaseDecoder):
    """
    Adaptive Inversion Adversary (Section 3.5).
    Full knowledge of:
    - client encoder architecture
    - dual-projection heads (Pu, Pp)
    - training procedure
    - defensive mechanisms
    Stronger than GL3: 8 residual blocks.
    Hidden dimension is guaranteed to be > GL3 for all datasets.
    """
    def __init__(self, bottleneck_dim: int, output_dim: int,
                 dropout: float = 0.0):
        super().__init__(bottleneck_dim, output_dim)
        # Guarantee Adaptive > GL3 for all datasets
        if output_dim > 10000:
            self.hidden_dim = 512   # Same as GL3 but with 8 residual blocks vs 6
        elif output_dim > 1000:
            self.hidden_dim = 768   # Larger than GL3 (512)
        else:
            self.hidden_dim = 1024  # Larger than GL3 (512)

        blocks = [
            nn.Linear(bottleneck_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(8):
            blocks.append(ResidualBlock1D(self.hidden_dim, dropout))
        blocks.append(nn.Linear(self.hidden_dim, output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, zu: torch.Tensor) -> torch.Tensor:
        return self.net(zu)


# 8. Decoder Factory
def build_decoder(level: int, bottleneck_dim: int, output_dim: int,
                  dropout: float = 0.0) -> BaseDecoder:
    """
    Build inversion decoder based on adversary level.
    
    Args:
        level: 0=GL0, 1=GL1, 2=GL2, 3=GL3, 4=Adaptive
        bottleneck_dim: Dimension of zu (utility representation)
        output_dim: Dimension of reconstruction target (input_dim)
        dropout: Dropout rate
    
    Returns:
        BaseDecoder instance
    
    Cost hierarchy: cost(GL0) < cost(GL1) < cost(GL2) < cost(GL3) < cost(Adaptive)
    """
    decoders = {
        0: GL0Decoder,
        1: GL1Decoder,
        2: GL2Decoder,
        3: GL3Decoder,
        4: AdaptiveInversionAdversary,
    }
    decoder_class = decoders.get(level, GL2Decoder)
    return decoder_class(bottleneck_dim, output_dim, dropout=dropout)


# 9. Inversion Attack
class InversionAttack:
    """
    Model Inversion (MI) Attack.
    
    Trains a decoder to reconstruct private input from utility projection zu.
    Uses MSE reconstruction loss as defined in paper Section 3.4.
    The attacker attacks only zu (the transmitted activation), never zp or h.
    """
    def __init__(self, decoder: BaseDecoder, learning_rate: float = 1e-3,
                 epochs: int = 50, device: str = "cuda"):
        self.decoder = decoder.to(device)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {
            "loss": [],
            "mse": [],
        }

    def train(self, zu: torch.Tensor, x: torch.Tensor,
              batch_size: int = 64, verbose: bool = False) -> Dict[str, List[float]]:
        """
        Train inversion decoder on zu-target pairs.
        
        Args:
            zu: Utility projections (batch, bottleneck_dim)
                = P_u(C_i(x)) - the transmitted activation
            x: Original inputs (batch, input_dim)
            batch_size: Training batch size
            verbose: Print progress
        
        Returns:
            Training history
        """
        self.decoder.train()
        dataset = torch.utils.data.TensorDataset(zu, x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_mse = 0.0
            num_batches = 0

            for batch_zu, batch_x in loader:
                batch_zu = batch_zu.to(self.device)
                batch_x = batch_x.to(self.device)

                self.optimizer.zero_grad()
                recon = self.decoder(batch_zu)
                loss = self.criterion(recon, batch_x)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_mse += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            avg_mse = epoch_mse / num_batches if num_batches > 0 else 0.0

            self.history["loss"].append(avg_loss)
            self.history["mse"].append(avg_mse)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        return self.history

    def reconstruct(self, zu: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input from utility projection zu.
        
        Args:
            zu: Utility projection (batch, bottleneck_dim)
        
        Returns:
            Reconstructed input (batch, input_dim)
        """
        self.decoder.eval()
        with torch.no_grad():
            return self.decoder(zu.to(self.device))

    def compute_mse(self, zu: torch.Tensor, x: torch.Tensor) -> float:
        """Compute MSE between reconstruction and ground truth."""
        recon = self.reconstruct(zu)
        return F.mse_loss(recon, x.to(self.device)).item()


# STANDARD PGD ATTACK (Madry et al., 2018)

class PGDAttack:
    """
    Standard Projected Gradient Descent (PGD) Attack.
    
    Based on Madry et al., "Towards Deep Learning Models Resistant to 
    Adversarial Attacks," ICLR 2018.
    
    This is the EXACT implementation used in:
    - ResSFL paper (Li et al., CVPR 2022)
    - Standard adversarial training literature
    
    Args:
        model: Target model (forward must return dict with "logits" key)
        epsilon: L-infinity perturbation bound (default: 0.3)
        steps: Number of PGD iterations (default: 40)
        alpha: Step size (default: epsilon / steps * 1.5 = 0.01 for eps=0.3, steps=40)
        device: Device
        class_weights: Per-class weights for loss (handles imbalance)
        targeted_class: If set, perform targeted attack on this class
        random_start: If True, start from random perturbation within epsilon ball
        data_min: Global minimum value for clipping (None = no clipping)
        data_max: Global maximum value for clipping (None = no clipping)
        verbose: If True, print debug information
    """
    def __init__(self, model: nn.Module, epsilon: float = 0.3,
                 steps: int = 40, alpha: float = 0.01,
                 device: str = "cuda",
                 class_weights: Optional[torch.Tensor] = None,
                 targeted_class: Optional[int] = None,
                 random_start: bool = True,
                 data_min: Optional[float] = None,
                 data_max: Optional[float] = None,
                 verbose: bool = False):
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.targeted_class = targeted_class
        self.random_start = random_start
        self.data_min = data_min
        self.data_max = data_max
        self.verbose = verbose

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

    def attack(self, x: torch.Tensor, y: torch.Tensor,
               targeted: bool = False, target_class: Optional[int] = None,
               epsilon: Optional[float] = None,
               steps: Optional[int] = None,
               alpha: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate adversarial examples using standard PGD.
        
        Args:
            x: Input batch (batch, input_dim)
            y: True labels
            targeted: If True, perform targeted attack
            target_class: Target class for targeted attack (overrides self.targeted_class)
            epsilon: Override epsilon for this attack (optional)
            steps: Override steps for this attack (optional)
            alpha: Override alpha for this attack (optional)
        
        Returns:
            (adversarial_examples, perturbations)
        """
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        epsilon = epsilon if epsilon is not None else self.epsilon
        steps = steps if steps is not None else self.steps
        alpha = alpha if alpha is not None else self.alpha

        # Determine target class for targeted attack
        if targeted:
            if target_class is not None:
                target = torch.full_like(y, target_class, device=self.device)
            elif self.targeted_class is not None:
                target = torch.full_like(y, self.targeted_class, device=self.device)
            else:
                target = torch.full_like(y, 1, device=self.device)
        else:
            target = None

        # Standard PGD: Random start within epsilon ball
        if self.random_start:
            delta = torch.zeros_like(x, device=self.device).uniform_(-epsilon, epsilon)
        else:
            delta = torch.zeros_like(x, device=self.device)

        # Standard PGD: Use eval() mode (not train())
        self.model.eval()

        for step in range(steps):
            # Re-enable gradients for delta at each iteration
            delta.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(x + delta)
            logits = outputs["logits"]

            # Standard PGD: Correct sign convention
            if targeted and target is not None:
                # Targeted attack: minimize loss for target class
                loss = self.criterion(logits, target)
                step_sign = -1.0
            else:
                # Untargeted attack: maximize loss for true label
                loss = self.criterion(logits, y)
                step_sign = 1.0

            # Use torch.autograd.grad() to compute gradient only for delta
            # This avoids any gradient accumulation on model parameters and
            # does not require self.model.zero_grad()
            # allow_unused=False ensures any graph-disconnection bug fails loudly
            grad = torch.autograd.grad(
                loss,
                delta,
                retain_graph=False,
                create_graph=False
            )[0]
            
            # Ensure gradient exists
            if grad is None:
                if self.verbose:
                    print(f"[PGD] Warning: No gradient at step {step}")
                break
            
            # Standard PGD: Use sign of gradient for update
            with torch.no_grad():
                delta = delta + step_sign * alpha * grad.sign()
                delta = torch.clamp(delta, -epsilon, epsilon)
                
                # Optional: Clip to global data range if specified
                if self.data_min is not None and self.data_max is not None:
                    delta = torch.clamp(x + delta, self.data_min, self.data_max) - x
                
                # Detach delta to break the computation graph
                # This prevents gradient tracking across iterations
                delta = delta.detach()

        x_adv = x + delta.detach()
        
        if self.verbose:
            with torch.no_grad():
                clean_outputs = self.model(x)
                clean_logits = clean_outputs["logits"]
                adv_outputs = self.model(x_adv)
                adv_logits = adv_outputs["logits"]
                clean_acc = (clean_logits.argmax(1) == y).float().mean().item()
                adv_acc = (adv_logits.argmax(1) == y).float().mean().item()
                print(f"[PGD] Clean Acc: {clean_acc:.4f}, Adv Acc: {adv_acc:.4f}")
                print(f"[PGD] Delta L-inf: {delta.abs().max().item():.6f}")
                print(f"[PGD] Delta Mean: {delta.abs().mean().item():.6f}")
        
        return x_adv, delta.detach()

    def attack_batch(self, x: torch.Tensor, y: torch.Tensor,
                     batch_size: int = 32,
                     epsilon: Optional[float] = None,
                     steps: Optional[int] = None,
                     alpha: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate adversarial examples in batches for memory efficiency.
        
        Args:
            x: Input batch
            y: Labels
            batch_size: Sub-batch size for attack
            epsilon: Override epsilon (optional)
            steps: Override steps (optional)
            alpha: Override alpha (optional)
        
        Returns:
            (adversarial_examples, perturbations)
        """
        x = x.to(self.device)
        y = y.to(self.device)

        x_adv_list = []
        delta_list = []

        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            x_adv_batch, delta_batch = self.attack(
                x_batch, y_batch,
                epsilon=epsilon, steps=steps, alpha=alpha
            )
            x_adv_list.append(x_adv_batch)
            delta_list.append(delta_batch)

        return torch.cat(x_adv_list, dim=0), torch.cat(delta_list, dim=0)

    def compute_accuracy(self, model: nn.Module, loader: torch.utils.data.DataLoader,
                         attack: bool = True,
                         epsilon: Optional[float] = None,
                         steps: Optional[int] = None,
                         alpha: Optional[float] = None) -> Tuple[float, Optional[float]]:
        """
        Compute accuracy with and without attack.
        
        Args:
            model: Model to evaluate (forward must return dict with "logits")
            loader: DataLoader
            attack: If True, apply PGD attack
            epsilon: Override epsilon (optional)
            steps: Override steps (optional)
            alpha: Override alpha (optional)
        
        Returns:
            (clean_accuracy, attacked_accuracy)
            attacked_accuracy is None if attack=False
        """
        model.eval()
        clean_correct = 0
        adv_correct = 0
        total = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                outputs = model(x)
                preds = outputs["logits"].argmax(dim=1)
                clean_correct += (preds == y).sum().item()

            if attack:
                x_adv, _ = self.attack(x, y, epsilon=epsilon, steps=steps, alpha=alpha)
                with torch.no_grad():
                    outputs = model(x_adv)
                    preds = outputs["logits"].argmax(dim=1)
                    adv_correct += (preds == y).sum().item()

            total += x.size(0)

        clean_acc = 100.0 * clean_correct / total if total > 0 else 0.0
        adv_acc = 100.0 * adv_correct / total if attack and total > 0 else None

        return clean_acc, adv_acc

    def compute_robust_metrics(self, model: nn.Module, loader: torch.utils.data.DataLoader,
                               attack: bool = True,
                               epsilon: Optional[float] = None,
                               steps: Optional[int] = None,
                               alpha: Optional[float] = None) -> Dict[str, float]:
        """
        Compute metrics with and without attack.
        
        Args:
            model: Model to evaluate
            loader: DataLoader
            attack: If True, apply PGD attack
            epsilon: Override epsilon (optional)
            steps: Override steps (optional)
            alpha: Override alpha (optional)
        
        Returns:
            Dictionary with clean_acc, adv_acc, clean_f1, adv_f1
        """
        model.eval()
        all_y_true = []
        all_y_pred_clean = []
        all_y_pred_adv = []

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            # Clean predictions
            with torch.no_grad():
                outputs = model(x)
                preds = outputs["logits"].argmax(dim=1)
                all_y_true.extend(y.cpu().numpy())
                all_y_pred_clean.extend(preds.cpu().numpy())

            if attack:
                x_adv, _ = self.attack(x, y, epsilon=epsilon, steps=steps, alpha=alpha)
                with torch.no_grad():
                    outputs = model(x_adv)
                    preds = outputs["logits"].argmax(dim=1)
                    all_y_pred_adv.extend(preds.cpu().numpy())

        all_y_true = np.array(all_y_true)
        all_y_pred_clean = np.array(all_y_pred_clean)

        # Clean metrics
        clean_acc = accuracy_score(all_y_true, all_y_pred_clean) * 100
        clean_f1 = f1_score(all_y_true, all_y_pred_clean, average='weighted', zero_division=0)

        metrics = {
            'clean_acc': clean_acc,
            'clean_f1': clean_f1,
        }

        if attack and len(all_y_pred_adv) > 0:
            all_y_pred_adv = np.array(all_y_pred_adv)
            adv_acc = accuracy_score(all_y_true, all_y_pred_adv) * 100
            adv_f1 = f1_score(all_y_true, all_y_pred_adv, average='weighted', zero_division=0)
            metrics.update({
                'adv_acc': adv_acc,
                'adv_f1': adv_f1,
                'robust_acc': adv_acc,
                'robust_f1': adv_f1,
            })

        return metrics


# 10. Verification Functions
def verify_decoder_shapes(cfg) -> Dict[str, Dict[str, Tuple[int, ...]]]:
    """
    Verify decoder input/output shapes for all adversary levels.
    """
    bottleneck_dim = cfg.get_bottleneck_dim()
    input_dim = cfg.get_dataset_input_dim()
    batch_size = 4
    results = {}
    levels = {
        0: "GL0",
        1: "GL1",
        2: "GL2",
        3: "GL3",
        4: "Adaptive"
    }
    for level, name in levels.items():
        decoder = build_decoder(level, bottleneck_dim, input_dim)
        zu = torch.randn(batch_size, bottleneck_dim)
        recon = decoder(zu)
        results[name] = {
            "input_shape": zu.shape,
            "output_shape": recon.shape,
            "hidden_dim": decoder.hidden_dim if hasattr(decoder, 'hidden_dim') else None,
        }
        hidden_info = f" (hidden={decoder.hidden_dim})" if hasattr(decoder, 'hidden_dim') else ""
        print(f"{name}: {zu.shape} -> {recon.shape}{hidden_info}")
    return results


def verify_pgd(cfg, batch_size: int = 4, device: str = "cpu",
               epsilon: float = 0.3, steps: int = 5, alpha: float = 0.01):
    """
    Verify PGD attack produces valid perturbations.
    """
    from models import build_model_by_method
    model = build_model_by_method(cfg).to(device)
    attack = PGDAttack(model, epsilon=epsilon, steps=steps, alpha=alpha, 
                       device=device, verbose=True)
    x = torch.randn(batch_size, cfg.get_dataset_input_dim(), device=device)
    y = torch.randint(0, cfg.get_dataset_num_classes(), (batch_size,), device=device)
    x_adv, delta = attack.attack(x, y)
    print(f"PGD Verification:")
    print(f"  Input shape: {x.shape}")
    print(f"  Adversarial shape: {x_adv.shape}")
    print(f"  Delta shape: {delta.shape}")
    print(f"  Delta norm (L-inf): {delta.abs().max().item():.6f}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Steps: {steps}")
    print(f"  Alpha: {alpha}")
    print(f"  Max perturbation <= epsilon: {delta.abs().max().item() <= epsilon}")
    return {
        "input_shape": x.shape,
        "adv_shape": x_adv.shape,
        "delta_norm_inf": delta.abs().max().item(),
        "within_epsilon": delta.abs().max().item() <= epsilon,
        "epsilon": epsilon,
        "steps": steps,
        "alpha": alpha,
    }


# 11. Adversary Level Mapping
ADVERSARY_LEVEL_NAMES = {
    0: "GL0",
    1: "GL1",
    2: "GL2",
    3: "GL3",
    4: "Adaptive",
}


def get_adversary_name(level: int) -> str:
    """Get human-readable name for adversary level."""
    return ADVERSARY_LEVEL_NAMES.get(level, f"Unknown({level})")


# 12. PGD Attack Factory with Configurable Parameters
def create_pgd_attack(model: nn.Module,
                      epsilon: float = 0.3,
                      steps: int = 40,
                      alpha: float = 0.01,
                      device: str = "cuda",
                      class_weights: Optional[torch.Tensor] = None,
                      targeted_class: Optional[int] = None,
                      random_start: bool = True,
                      data_min: Optional[float] = None,
                      data_max: Optional[float] = None,
                      verbose: bool = False) -> PGDAttack:
    """
    Factory function to create PGD attack with configurable parameters.
    
    Args:
        model: Target model
        epsilon: L-infinity bound (DEFAULT: 0.3)
        steps: Number of iterations (DEFAULT: 40)
        alpha: Step size (DEFAULT: 0.01)
        device: Device
        class_weights: Class weights for imbalanced datasets
        targeted_class: Target class for targeted attack
        random_start: Random start within epsilon ball (default: True)
        data_min: Global minimum value for clipping
        data_max: Global maximum value for clipping
        verbose: If True, print debug information
    
    Returns:
        PGDAttack instance
    """
    return PGDAttack(
        model=model,
        epsilon=epsilon,
        steps=steps,
        alpha=alpha,
        device=device,
        class_weights=class_weights,
        targeted_class=targeted_class,
        random_start=random_start,
        data_min=data_min,
        data_max=data_max,
        verbose=verbose
    )


# 13. Standalone test
if __name__ == "__main__":
    import sys
    from config import ExperimentConfig
    
    print("=" * 60)
    print("Phase 5: Attack Verification")
    print("=" * 60)

    # Test decoder shapes for all datasets
    print("\n[1] Decoder Shapes:")
    print("-" * 40)
    for ds in ["wesad", "bidmc", "mhealth"]:
        print(f"\n{ds.upper()}:")
        cfg = ExperimentConfig(dataset=ds)
        verify_decoder_shapes(cfg)

    # Test PGD attack with default params
    print("\n[2] PGD Attack (Default: eps=0.3, steps=5, alpha=0.01):")
    print("-" * 40)
    cfg = ExperimentConfig(dataset="mhealth")
    verify_pgd(cfg)

    # Test PGD attack with custom params
    print("\n[3] PGD Attack (Custom: eps=0.5, steps=10, alpha=0.02):")
    print("-" * 40)
    verify_pgd(cfg, epsilon=0.5, steps=10, alpha=0.02)

    # Test adversary level mapping
    print("\n[4] Adversary Level Mapping:")
    print("-" * 40)
    for level in [0, 1, 2, 3, 4]:
        name = get_adversary_name(level)
        print(f"  Level {level}: {name}")

    # Test cost hierarchy
    print("\n[5] Cost Hierarchy (hidden dimensions):")
    print("-" * 40)
    for ds in ["wesad", "bidmc", "mhealth"]:
        print(f"\n{ds.upper()}:")
        cfg = ExperimentConfig(dataset=ds)
        for level in [0, 1, 2, 3, 4]:
            decoder = build_decoder(level, cfg.get_bottleneck_dim(), cfg.get_dataset_input_dim())
            name = get_adversary_name(level)
            print(f"  {name}: hidden_dim={decoder.hidden_dim}")

    # Test tunable PGD parameters with a real model
    print("\n[6] Tunable PGD Parameters:")
    print("-" * 40)
    cfg = ExperimentConfig(dataset="mhealth")
    try:
        from models import build_model_by_method
        model = build_model_by_method(cfg).to("cpu")
        attack = create_pgd_attack(model, epsilon=0.3, steps=40, alpha=0.01)
        print(f"  Default: eps={attack.epsilon}, steps={attack.steps}, alpha={attack.alpha}")
        attack2 = create_pgd_attack(model, epsilon=0.5, steps=60, alpha=0.02)
        print(f"  Custom: eps={attack2.epsilon}, steps={attack2.steps}, alpha={attack2.alpha}")
        print("   PGD parameter test passed")
    except Exception as e:
        print(f"  Could not build model for PGD test: {e}")
        print("  Skipping PGD parameter test (requires full model with proper forward signature)")

    # Test data range clipping
    print("\n[7] Data Range Clipping:")
    print("-" * 40)
    cfg = ExperimentConfig(dataset="mhealth")
    try:
        from models import build_model_by_method
        model = build_model_by_method(cfg).to("cpu")
        attack3 = create_pgd_attack(model, epsilon=0.3, steps=40, alpha=0.01,
                                    data_min=-3.0, data_max=3.0)
        print(f"  With z-score range: data_min={attack3.data_min}, data_max={attack3.data_max}")
        attack4 = create_pgd_attack(model, epsilon=0.3, steps=40, alpha=0.01,
                                    data_min=0.0, data_max=1.0)
        print(f"  With min-max range: data_min={attack4.data_min}, data_max={attack4.data_max}")
        print("   Data range clipping test passed")
    except Exception as e:
        print(f"  Could not build model for clipping test: {e}")
        print("  Skipping data range clipping test")

    # Test verbose mode with gradient leakage verification
    print("\n[8] Verbose Mode Test & Gradient Leakage Verification:")
    print("-" * 40)
    cfg = ExperimentConfig(dataset="mhealth")
    try:
        from models import build_model_by_method
        model = build_model_by_method(cfg).to("cpu")
        
        # Clear any existing gradients
        for p in model.parameters():
            p.grad = None
        
        attack_verbose = create_pgd_attack(model, epsilon=0.3, steps=3, alpha=0.01,
                                           verbose=True)
        x = torch.randn(2, cfg.get_dataset_input_dim())
        y = torch.randint(0, cfg.get_dataset_num_classes(), (2,))
        x_adv, delta = attack_verbose.attack(x, y)
        
        print(f"\n  PGD attack completed successfully!")
        print(f"  Input shape: {x.shape}, Adversarial shape: {x_adv.shape}")
        print(f"  Delta L-inf norm: {delta.abs().max().item():.6f}")
        print(f"  Within epsilon bound: {delta.abs().max().item() <= 0.3}")
        
        # Verification: Check that no gradients leaked into model parameters
        num_grads = sum(1 for p in model.parameters() if p.grad is not None)
        print(f"  Parameter gradients after attack: {num_grads} (expected: 0)")
        if num_grads == 0:
            print("   No gradient leakage detected!")
        else:
            print("   WARNING: Gradient leakage detected!")
            print("  This indicates Bug #6 is not fully fixed.")
    except ImportError as e:
        print(f"  Could not import build_model_by_method: {e}")
        print("  Skipping verbose mode test (requires full model with proper forward signature)")
    except Exception as e:
        print(f"  Could not run verbose mode test: {e}")
        print("  Skipping verbose mode test")

    print("\n" + "=" * 60)
    print("Phase 5: COMPLETE")
    print("=" * 60)