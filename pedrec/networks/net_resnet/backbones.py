"""
Backbones for the PedRecNet.

* ``resnet50``: the original ResNet-50 feature extractor (state dict compatible with all published checkpoints)
* ``timm:<name>``: any timm CNN (``features_only``, last stage) or a plain ViT (ViTPose style: patch tokens reshaped to
  a stride 16 feature map). The deconv decoder is adapted to the backbone stride so that the heatmaps keep 1/4 of the
  input resolution.
"""
import math

import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

from pedrec.models.data_structures import ImageSize
from pedrec.networks.net_resnet.resnet_feature_extractor import ResNetHeadless


class ResNet50Backbone(ResNetHeadless):
    out_channels = 2048
    stride = 32

    def __init__(self, pretrained: bool = False):
        super().__init__(Bottleneck, [3, 4, 6, 3])
        if pretrained:
            from torchvision.models import ResNet50_Weights, resnet50
            weights = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).state_dict()
            self.load_state_dict({k: v for k, v in weights.items() if not k.startswith("fc.")})


class TimmBackbone(nn.Module):
    def __init__(self, name: str, input_size: ImageSize, pretrained: bool = False):
        super().__init__()
        import timm
        self.is_vit = name.startswith(("vit", "deit", "eva", "beit"))
        if self.is_vit:
            self.model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="",
                                           img_size=(input_size.height, input_size.width))
            patch = self.model.patch_embed.patch_size
            self.stride = patch[0] if isinstance(patch, (tuple, list)) else patch
            self.out_channels = self.model.num_features
            self.grid = (input_size.height // self.stride, input_size.width // self.stride)
        else:
            self.model = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(-1,))
            self.stride = self.model.feature_info.reduction()[-1]
            self.out_channels = self.model.feature_info.channels()[-1]

    def forward(self, x):
        if not self.is_vit:
            return self.model(x)[-1]
        tokens = self.model.forward_features(x)
        tokens = tokens[:, getattr(self.model, "num_prefix_tokens", 1):]
        batch, _, channels = tokens.shape
        return tokens.transpose(1, 2).reshape(batch, channels, *self.grid)


def build_backbone(name: str, input_size: ImageSize, pretrained: bool = False) -> nn.Module:
    if name == "resnet50":
        return ResNet50Backbone(pretrained)
    if name.startswith("timm:"):
        return TimmBackbone(name[len("timm:"):], input_size, pretrained)
    raise ValueError(f"Unknown backbone '{name}'")


def num_deconv_layers_for_stride(stride: int) -> int:
    """Number of stride 2 deconvolutions to get from the backbone stride to heatmaps with stride 4."""
    layers = int(round(math.log2(stride / 4)))
    if layers < 1 or 4 * 2 ** layers != stride:
        raise ValueError(f"Unsupported backbone stride {stride} (8, 16, 32 or 64 expected)")
    return layers
