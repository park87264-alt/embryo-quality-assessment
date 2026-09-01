from __future__ import annotations

import torch
from torch import nn
import torchvision.models as tv_models

try:
    import timm
except Exception:
    timm = None


def build_backbone(name: str = "resnet18", pretrained: bool = True):
    name = name.lower()
    if name == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim
    if name == "resnet50":
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet50(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim
    if name == "convnext_tiny":
        weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = tv_models.convnext_tiny(weights=weights)
        dim = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
        return model, dim
    if timm is None:
        raise RuntimeError("timm is required for this backbone")
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    return model, model.num_features


class MLPExpert(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualExpertEmbryoModel(nn.Module):
    """Shared image encoder with two task-specific expert branches."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        expert_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.stage_expert = MLPExpert(feat_dim, expert_dim, dropout)
        self.gardner_expert = MLPExpert(feat_dim, expert_dim, dropout)
        self.phase_head = nn.Linear(expert_dim, 16)
        self.exp_head = nn.Linear(expert_dim, 5)
        self.icm_head = nn.Linear(expert_dim, 3)
        self.te_head = nn.Linear(expert_dim, 3)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = images.shape
        feats = self.backbone(images.view(b * v, c, h, w))
        return feats.view(b, v, -1).mean(dim=1)

    def forward(self, images: torch.Tensor, task: str):
        feats = self.encode(images)
        if task == "stage":
            expert = self.stage_expert(feats)
            return {"phase_logits": self.phase_head(expert)}
        if task == "gardner":
            expert = self.gardner_expert(feats)
            return {
                "exp_logits": self.exp_head(expert),
                "icm_logits": self.icm_head(expert),
                "te_logits": self.te_head(expert),
            }
        raise ValueError(f"Unknown task: {task}")
