"""
Backbone and neck of PedRecNet v2.

The backbone is PP-HGNetV2-B3 (timm ``hgnetv2_b3``, ImageNet SSLD weights): a convolution / BN / ReLU network designed
for GPU latency (used by RT-DETR / D-FINE). Its stride 4, 8, 16 and 32 features are fused top down (FPN style) into
stride 4 features, from which the heatmap heads predict 64 x 48 heatmaps for a 256 x 192 crop, as before.

Compared to the ResNet-50 + three 256 channel transposed convolutions of v1 the network needs about a third of the
multiply-accumulates and has a third of the parameters.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pedrec.configs.pedrec_net_config import PedRecNetConfig


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def init_conv_bn(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class PedRecBackbone(nn.Module):
    def __init__(self, cfg: PedRecNetConfig, pretrained: bool = False):
        """:param pretrained: load the ImageNet weights (training start, needs the Hugging Face hub / cache)"""
        super().__init__()
        import timm
        self.body = timm.create_model(cfg.model.backbone, pretrained=pretrained, features_only=True,
                                      out_indices=(0, 1, 2, 3))
        self.channels: List[int] = self.body.feature_info.channels()
        assert self.body.feature_info.reduction() == [4, 8, 16, 32], self.body.feature_info.reduction()

    def forward(self, x) -> List[torch.Tensor]:
        return self.body(x)


class PedRecNeck(nn.Module):
    """Top down fusion of the stride 32 / 16 / 8 features, then refinement at stride 4 with the stride 4 features."""

    def __init__(self, in_channels: List[int], neck_channels: int, head_channels: int):
        super().__init__()
        c4, c8, c16, c32 = in_channels
        self.lateral_32 = conv_bn_relu(c32, neck_channels)
        self.lateral_16 = conv_bn_relu(c16, neck_channels)
        self.lateral_8 = conv_bn_relu(c8, neck_channels)
        self.fuse_8 = conv_bn_relu(neck_channels, neck_channels, 3)
        self.lateral_4 = conv_bn_relu(c4, neck_channels)
        self.fuse_4 = conv_bn_relu(neck_channels, head_channels, 3)

    @staticmethod
    def _up(x, like):
        return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f4, f8, f16, f32 = features
        p16 = self.lateral_16(f16) + self._up(self.lateral_32(f32), f16)
        p8 = self.fuse_8(self.lateral_8(f8) + self._up(p16, f8))
        return self.fuse_4(self.lateral_4(f4) + self._up(p8, f4))

    def init_weights(self):
        init_conv_bn(self)
