from dataclasses import dataclass, field

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.models.data_structures import ImageSize


@dataclass
class PedRecNetModelConfig:
    input_size: ImageSize = field(default_factory=lambda: ImageSize(width=192, height=256))
    heatmap_size: ImageSize = field(default_factory=lambda: ImageSize(width=48, height=64))  # stride 4
    num_joints: int = len(SKELETON_PEDREC_JOINTS)
    backbone: str = "hgnetv2_b3"  # timm model name (PP-HGNetV2-B3, GPU latency oriented, ImageNet SSLD weights)
    neck_channels: int = 128  # top down fusion of the stride 8 / 16 / 32 features
    head_channels: int = 64  # stride 4 features of the 2D / 3D heatmap heads


@dataclass
class PedRecNetConfig:
    model: PedRecNetModelConfig = field(default_factory=PedRecNetModelConfig)
