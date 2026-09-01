from __future__ import annotations

import torch
from torch import nn
import torchvision.models as tv_models

try:
    import timm
except Exception:  # timm is optional for torchvision-only baselines.
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
        raise RuntimeError("timm is required for non-torchvision backbone names")
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    dim = model.num_features
    return model, dim


class OpenBackboneMultiTask(nn.Module):
    """Single-focal or multi-focal late-fusion baseline.

    For multi-focal input, the same ImageNet-pretrained backbone encodes each
    focal plane. Features are fused by mean pooling, a conservative late-fusion
    baseline widely used in multi-view image classification.
    """

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone, dim = build_backbone(backbone_name, pretrained=pretrained)
        self.phase_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 16))
        self.te_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 3))
        self.icm_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 3))

    def forward(self, images: torch.Tensor):
        # images: [B, V, C, H, W]
        b, v, c, h, w = images.shape
        x = images.view(b * v, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, v, -1).mean(dim=1)
        return {
            "phase_logits": self.phase_head(feats),
            "te_logits": self.te_head(feats),
            "icm_logits": self.icm_head(feats),
        }
