from __future__ import annotations

import torch
from torch import nn
import torchvision.models as tv_models


def build_resnet18_encoder(pretrained: bool = True):
    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    model = tv_models.resnet18(weights=weights)
    dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, dim


class GardnerROISegAssist(nn.Module):
    """Global + structure-guided ROI branches for Gardner scoring."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3, share_roi_encoder: bool = True):
        super().__init__()
        self.global_encoder, dim = build_resnet18_encoder(pretrained)
        self.share_roi_encoder = share_roi_encoder
        self.icm_encoder, _ = build_resnet18_encoder(pretrained)
        if share_roi_encoder:
            self.te_encoder = self.icm_encoder
        else:
            self.te_encoder, _ = build_resnet18_encoder(pretrained)
        self.exp_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 5))
        self.icm_head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )
        self.te_head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, global_x: torch.Tensor, icm_x: torch.Tensor, te_x: torch.Tensor):
        global_feat = self.global_encoder(global_x)
        icm_feat = self.icm_encoder(icm_x)
        te_feat = self.te_encoder(te_x)
        return {
            "exp_logits": self.exp_head(global_feat),
            "icm_logits": self.icm_head(torch.cat([global_feat, icm_feat], dim=1)),
            "te_logits": self.te_head(torch.cat([global_feat, te_feat], dim=1)),
        }
