# asfl/models.py

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn


class ChannelLayerNorm1d(nn.Module):
    """
    LayerNorm applied to the channel dimension for Conv1d.
    Expects input shape: (N, C, L)
    Normalizes over: (C,)
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L) -> permute to (N, L, C) for LayerNorm
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        return x


class InvertedResidual1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 expand_ratio: int = 6, kernel_size: int = 3):
        super().__init__()
        assert stride in (1, 2)
        
        # Explicitly handle expand_ratio == 1
        hidden_dim = in_ch if expand_ratio == 1 else int(round(in_ch * expand_ratio))
        
        self.use_residual = (stride == 1) and (in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv1d(in_ch, hidden_dim, 1, bias=False),
                ChannelLayerNorm1d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride,
                      padding=kernel_size // 2, groups=hidden_dim, bias=False),
            ChannelLayerNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv1d(hidden_dim, out_ch, 1, bias=False),
            ChannelLayerNorm1d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MobileNetV2Backbone1D(nn.Module):
    _BASE_CFG = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
    ]

    def __init__(self, input_dim: int, cut_layer: int = 3,
                 stem_channels: int = 32, width_mult: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.cut_layer = cut_layer

        # Validate input dimension
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if cut_layer < 1:
            raise ValueError(f"cut_layer must be >= 1, got {cut_layer}")

        n_stages = max(1, min(cut_layer, len(self._BASE_CFG)))
        cfg = self._BASE_CFG[:n_stages]

        stem_stride = 2 if input_dim >= 256 else 1
        stem_kernel = 7 if input_dim >= 256 else 3

        c_in = stem_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c_in, stem_kernel, stride=stem_stride,
                      padding=stem_kernel // 2, bias=False),
            ChannelLayerNorm1d(c_in),
            nn.ReLU6(inplace=True),
        )

        blocks = []
        for expand_ratio, c_out, n_blocks, stride in cfg:
            c_out = int(round(c_out * width_mult))
            for i in range(n_blocks):
                s = stride if i == 0 else 1
                blocks.append(InvertedResidual1D(c_in, c_out, s, expand_ratio))
                c_in = c_out
        self.blocks = nn.Sequential(*blocks)

        self.feature_dim = c_in
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        # Validate dimensions
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        
        hidden_dim = hidden_dim or max(out_dim, in_dim // 2)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
            
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ClientEncoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int, privacy_dim: int,
                 cut_layer: int = 3, stem_channels: int = 32,
                 width_mult: float = 1.0, utility_hidden: Optional[int] = None,
                 privacy_hidden: Optional[int] = None):
        super().__init__()
        # Validate dimensions
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")
        if privacy_dim <= 0:
            raise ValueError(f"privacy_dim must be positive, got {privacy_dim}")
        
        self.backbone = MobileNetV2Backbone1D(
            input_dim,
            cut_layer=cut_layer,
            stem_channels=stem_channels,
            width_mult=width_mult
        )
        h_dim = self.backbone.feature_dim

        self.Pu = ProjectionHead(h_dim, bottleneck_dim, utility_hidden)
        self.Pp = ProjectionHead(h_dim, privacy_dim, privacy_hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        zu = self.Pu(h)
        zp = self.Pp(h)
        return zu, zp, h


class ServerClassifier(nn.Module):
    def __init__(self, bottleneck_dim: int, num_classes: int,
                 hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        # Validate dimensions
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        
        self.net = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, zu: torch.Tensor) -> torch.Tensor:
        return self.net(zu)


class ASFLModel(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int, privacy_dim: int,
                 num_classes: int, cut_layer: int = 3, stem_channels: int = 32,
                 width_mult: float = 1.0, server_hidden_dim: int = 64,
                 utility_hidden_dim: int = 128, privacy_hidden_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.client = ClientEncoder(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            privacy_dim=privacy_dim,
            cut_layer=cut_layer,
            stem_channels=stem_channels,
            width_mult=width_mult,
            utility_hidden=utility_hidden_dim,
            privacy_hidden=privacy_hidden_dim
        )
        self.server = ServerClassifier(
            bottleneck_dim=bottleneck_dim,
            num_classes=num_classes,
            hidden_dim=server_hidden_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        zu, zp, h = self.client(x)
        logits = self.server(zu)
        return {
            "logits": logits,
            "zu": zu,
            "zp": zp,
            "h": h,
        }


class PlainSFLModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, cut_layer: int = 3,
                 stem_channels: int = 32, width_mult: float = 1.0,
                 hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.backbone = MobileNetV2Backbone1D(
            input_dim,
            cut_layer=cut_layer,
            stem_channels=stem_channels,
            width_mult=width_mult
        )
        h_dim = self.backbone.feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        logits = self.classifier(h)
        return {
            "logits": logits,
            "h": h,
        }


class ResSFLModel(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int, num_classes: int,
                 cut_layer: int = 3, stem_channels: int = 32,
                 width_mult: float = 1.0, server_hidden_dim: int = 64,
                 utility_hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.backbone = MobileNetV2Backbone1D(
            input_dim,
            cut_layer=cut_layer,
            stem_channels=stem_channels,
            width_mult=width_mult
        )
        h_dim = self.backbone.feature_dim
        self.bottleneck = ProjectionHead(h_dim, bottleneck_dim, utility_hidden_dim)
        self.classifier = ServerClassifier(
            bottleneck_dim=bottleneck_dim,
            num_classes=num_classes,
            hidden_dim=server_hidden_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        zu = self.bottleneck(h)
        logits = self.classifier(zu)
        return {
            "logits": logits,
            "zu": zu,
            "h": h,
        }


def build_model(cfg) -> ASFLModel:
    model_config = cfg.get_model_config()
    return ASFLModel(
        input_dim=cfg.get_dataset_input_dim(),
        bottleneck_dim=cfg.get_bottleneck_dim(),
        privacy_dim=model_config["privacy_dim"],
        num_classes=cfg.get_dataset_num_classes(),
        cut_layer=model_config["cut_layer"],
        stem_channels=model_config["stem_channels"],
        width_mult=model_config["width_mult"],
        server_hidden_dim=model_config["server_hidden_dim"],
        utility_hidden_dim=model_config["utility_hidden_dim"],
        privacy_hidden_dim=model_config["privacy_hidden_dim"],
        dropout=model_config["dropout"],
    )


def build_plain_model(cfg) -> PlainSFLModel:
    model_config = cfg.get_model_config()
    return PlainSFLModel(
        input_dim=cfg.get_dataset_input_dim(),
        num_classes=cfg.get_dataset_num_classes(),
        cut_layer=model_config["cut_layer"],
        stem_channels=model_config["stem_channels"],
        width_mult=model_config["width_mult"],
        hidden_dim=model_config["server_hidden_dim"],
        dropout=model_config["dropout"],
    )


def build_ressfl_model(cfg) -> ResSFLModel:
    model_config = cfg.get_model_config()
    return ResSFLModel(
        input_dim=cfg.get_dataset_input_dim(),
        bottleneck_dim=cfg.get_bottleneck_dim(),
        num_classes=cfg.get_dataset_num_classes(),
        cut_layer=model_config["cut_layer"],
        stem_channels=model_config["stem_channels"],
        width_mult=model_config["width_mult"],
        server_hidden_dim=model_config["server_hidden_dim"],
        utility_hidden_dim=model_config["utility_hidden_dim"],
        dropout=model_config["dropout"],
    )


def build_model_by_method(cfg) -> nn.Module:
    method_map = {
        "plain": build_plain_model,
        "ressfl": build_ressfl_model,
        "asfl": build_model,
    }
    builder = method_map.get(cfg.method, build_model)
    return builder(cfg)


def verify_forward_backward(cfg, batch_size: int = 4, device: str = "cpu"):
    model = build_model(cfg).to(device)
    model.train()

    x = torch.randn(batch_size, cfg.get_dataset_input_dim(), device=device)
    y = torch.randint(0, cfg.get_dataset_num_classes(), (batch_size,), device=device)

    outputs = model(x)
    logits = outputs["logits"]
    zu = outputs["zu"]
    zp = outputs["zp"]

    assert zu.shape == (batch_size, cfg.get_bottleneck_dim()), \
        f"zu shape mismatch: {zu.shape} vs expected dim {cfg.get_bottleneck_dim()}"
    assert zp.shape == (batch_size, cfg.privacy_dim), \
        f"zp shape mismatch: {zp.shape} vs expected dim {cfg.privacy_dim}"
    assert logits.shape == (batch_size, cfg.get_dataset_num_classes()), \
        f"logits shape mismatch: {logits.shape} vs expected {cfg.get_dataset_num_classes()}"

    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()

    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.client.backbone.parameters())

    print(f"[{cfg.dataset}] forward OK | logits {tuple(logits.shape)} "
          f"| zu {tuple(zu.shape)} | zp {tuple(zp.shape)} "
          f"| loss {loss.item():.4f} | backbone grad flow: {grad_ok}")

    return {
        "logits_shape": tuple(logits.shape),
        "zu_shape": tuple(zu.shape),
        "zp_shape": tuple(zp.shape),
        "loss": loss.item(),
        "grad_ok": grad_ok,
    }


def verify_all_methods(cfg, batch_size: int = 4, device: str = "cpu"):
    """
    Verify all three model variants (Plain, ResSFL, ASFL).
    
    Restores original cfg.method even if verification fails.
    """
    results = {}
    original_method = cfg.method

    try:
        for method in ["plain", "ressfl", "asfl"]:
            cfg.method = method
            model = build_model_by_method(cfg).to(device)
            model.train()

            x = torch.randn(batch_size, cfg.get_dataset_input_dim(), device=device)
            y = torch.randint(0, cfg.get_dataset_num_classes(), (batch_size,), device=device)

            outputs = model(x)
            logits = outputs["logits"]

            assert logits.shape == (batch_size, cfg.get_dataset_num_classes())

            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()

            if method == "asfl":
                backbone = model.client.backbone
            else:
                backbone = model.backbone

            grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in backbone.parameters())

            results[method] = {
                "logits_shape": tuple(logits.shape),
                "loss": loss.item(),
                "grad_ok": grad_ok,
            }

            print(f"  {method.upper()}: loss {loss.item():.4f} | grad flow: {grad_ok}")
    finally:
        # Restore original method once at the end
        cfg.method = original_method

    return results


if __name__ == "__main__":
    from config import ExperimentConfig

    print("=" * 60)
    print("Phase 3.5: Model Verification")
    print("=" * 60)

    for ds in ["wesad", "bidmc", "mhealth"]:
        print(f"\n--- {ds.upper()} ---")
        cfg = ExperimentConfig(dataset=ds, method="asfl")
        verify_forward_backward(cfg)

    # Verify all methods (Plain, ResSFL, ASFL)
    print("\n" + "=" * 60)
    print("Verifying all methods (Plain, ResSFL, ASFL)")
    print("=" * 60)
    cfg = ExperimentConfig(dataset="wesad", method="asfl")
    verify_all_methods(cfg)

    print("\n" + "=" * 60)
    print("Phase 3: COMPLETE")
    print("=" * 60)