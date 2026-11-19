"""Shared ResNet-50 backbone with lithology, mineral, group, and segmentation heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class RockNet(nn.Module):
    """Inference network. Weights are loaded from the round-8 checkpoint; ImageNet weights are not required."""

    def __init__(self, n_lithology: int, n_minerals: int, n_group: int = 3, n_pairs: int = 0):
        super().__init__()
        backbone = resnet50(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.lithology_head = nn.Linear(feat_dim, n_lithology)
        self.mineral_head = nn.Linear(feat_dim, n_minerals)
        self.group_head = nn.Linear(feat_dim, n_group)
        self.pair_head = nn.Linear(feat_dim, n_pairs) if n_pairs > 0 else None
        self.seg_head = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_minerals + 1, kernel_size=1),
        )
        self.n_minerals = n_minerals
        self.n_pairs = n_pairs

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat_map = self.encode(x)
        feat_raw = self.pool(feat_map).flatten(1)
        feat = self.dropout(feat_raw)
        seg = self.seg_head(feat_map)
        seg = F.interpolate(seg, size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = {
            "lithology": self.lithology_head(feat),
            "minerals": self.mineral_head(feat),
            "group": self.group_head(feat),
            "seg": seg,
            "feat": feat_raw,
        }
        if self.pair_head is not None:
            out["pair"] = self.pair_head(feat_raw)
        return out
