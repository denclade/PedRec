from dataclasses import dataclass, field
from typing import List

from torch import nn
from torchvision.models.resnet import Bottleneck

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.models.data_structures import ImageSize


@dataclass
class PedRecNetModelConfig(object):
    input_size: ImageSize
    heatmap_size: ImageSize
    num_joints: int
    final_conv_kernel: int
    deconv_with_bias: bool
    num_deconv_layers: int
    num_deconv_filters: List[int]
    num_deconv_kernels: List[int]
    num_pose_3d_deconv_layers: int
    num_pose_3d_deconv_filters: List[int]
    num_pose_3d_deconv_kernels: List[int]


@dataclass
class PedRecNetLayerConfig:
    layers: List[int]
    block: nn.Module


@dataclass
class PedRecNetTestConfig:
    post_process: bool


@dataclass
class PedRecNetTrainConfig:
    loss_use_target_weight: bool


@dataclass
class PedRecArchConfig:
    """
    Architecture variant of the PedRecNet. ``v1`` (default) is the published network; the other options require
    (re-)training of the affected parts, see ``get_arch_preset`` and doc/review/architecture_and_inference_review.md.
    """
    name: str = "v1"
    backbone: str = "resnet50"  # "resnet50" or "timm:<model name>", e.g. timm:convnext_tiny, timm:vit_base_patch16_224
    backbone_pretrained: bool = False  # ImageNet weights for the backbone (when not initialized from a checkpoint)
    orientation_head: str = "softargmax"  # "softargmax" (360 / 180 bins) or "biternion" (cos / sin, circular)
    orientation_pose_features: str = "deconv1d"  # features of the 3D pose: "deconv1d" (original) or "mlp"
    conf_head: str = "fc"  # "fc" (original) or "heatmap" (per joint heatmap statistics, input size independent)
    shared_pose_deconv: bool = False  # 2D and 3D heads share the last deconv stage
    mtl_weighting: str = "pedrec"  # "pedrec" (original sigma weighting) or "kendall" (log variance weighting)
    udp: bool = False  # unbiased data processing (Huang et al. 2020) for the affine transforms / coordinates

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(values: dict) -> "PedRecArchConfig":
        known = {k: v for k, v in values.items() if k in PedRecArchConfig.__dataclass_fields__}
        return PedRecArchConfig(**known)


ARCH_PRESETS = {
    "v1": PedRecArchConfig(),
    # v2: same backbone / pose heads as v1 (initializable from the v1 checkpoints), circular orientation head,
    # heatmap based joint confidence, original Kendall MTL weighting and UDP
    "v2": PedRecArchConfig(name="v2", orientation_head="biternion", orientation_pose_features="mlp",
                           conf_head="heatmap", mtl_weighting="kendall", udp=True),
}


def get_arch_preset(name: str, backbone: str = None) -> PedRecArchConfig:
    if name not in ARCH_PRESETS:
        raise KeyError(f"Unknown architecture preset '{name}', available: {', '.join(ARCH_PRESETS)}")
    arch = PedRecArchConfig(**ARCH_PRESETS[name].to_dict())
    if backbone is not None and backbone != arch.backbone:
        arch.backbone = backbone
        arch.backbone_pretrained = True
        arch.name = f"{arch.name}-{backbone.split(':')[-1]}"
    return arch


@dataclass
class PedRecNetConfig:
    layer: PedRecNetLayerConfig
    model: PedRecNetModelConfig = field(default_factory=lambda: PedRecNetModelConfig(
        input_size=ImageSize(width=192, height=256),
        heatmap_size=ImageSize(width=48, height=64),
        num_joints=len(SKELETON_PEDREC_JOINTS),
        final_conv_kernel=1,
        deconv_with_bias=False,
        num_deconv_layers=3,
        num_deconv_filters=[256, 256, 256],
        num_deconv_kernels=[4, 4, 4],
        num_pose_3d_deconv_layers=3,
        num_pose_3d_deconv_filters=[256, 256, 256],
        num_pose_3d_deconv_kernels=[4, 4, 4]
    ))

    train: PedRecNetTrainConfig = field(default_factory=lambda: PedRecNetTrainConfig(loss_use_target_weight=True))
    test: PedRecNetTestConfig = field(default_factory=lambda: PedRecNetTestConfig(post_process=True))
    arch: PedRecArchConfig = field(default_factory=PedRecArchConfig)


@dataclass
class PedRecNet50Config(PedRecNetConfig):
    layer: PedRecNetLayerConfig = field(default_factory=lambda: PedRecNetLayerConfig(
        layers=[3, 4, 6, 3],
        block=Bottleneck
    ))
