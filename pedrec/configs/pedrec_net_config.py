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


@dataclass
class PedRecNet50Config(PedRecNetConfig):
    layer: PedRecNetLayerConfig = field(default_factory=lambda: PedRecNetLayerConfig(
        layers=[3, 4, 6, 3],
        block=Bottleneck
    ))
