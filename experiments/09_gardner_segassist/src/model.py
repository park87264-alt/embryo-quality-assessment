from __future__ import annotations

import torch
from torch import nn
import torchvision.models as tv_models


class ResNet18SegAssistGardner(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(6, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            model.conv1.weight[:, :3] = old_conv.weight
            model.conv1.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        self.backbone = model
        self.exp_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 5))
        self.icm_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 3))
        self.te_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 3))

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return {
            "exp_logits": self.exp_head(feat),
            "icm_logits": self.icm_head(feat),
            "te_logits": self.te_head(feat),
        }
