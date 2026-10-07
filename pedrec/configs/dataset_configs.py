from dataclasses import dataclass, field

from pedrec.models.constants.sample_method import SAMPLE_METHOD
from pedrec.models.data_structures import ImageSize

@dataclass
class CocoDatasetConfig:
    flip: bool
    scale_factor: float
    rotation_factor: int
    use_mebow_orientation: bool

@dataclass
class TudDatasetConfig:
    flip: bool
    scale_factor: float
    subsample: int
    subsampling_strategy: SAMPLE_METHOD


@dataclass
class VideoActionDatasetConfig:
    skeleton_3d_range: int
    flip: bool
    subsample: int
    subsampling_strategy: SAMPLE_METHOD
    use_unit_skeleton: bool
    min_joint_score: float
    add_2d: bool
    temporal_field: ImageSize = field(default_factory=lambda: ImageSize(32, 32))
    frame_sampling: int = 1


@dataclass
class PedRecTemporalDatasetConfig:
    skeleton_3d_range: int  # range of the 3d skeleton in all 3 dimensions, e.g arm to arm reach etc. usually 3000mm
    flip: bool
    scale_factor: float
    rotation_factor: int
    img_pattern: str
    subsample: int
    subsampling_strategy: SAMPLE_METHOD
    use_unit_skeleton: bool
    min_joint_score: float
    add_2d: bool
    temporal_field: ImageSize = field(default_factory=lambda: ImageSize(32, 32))
    gt_result_ratio: float = 1.0
    frame_sampling: int = 1


@dataclass
class PedRecDatasetConfig:
    skeleton_3d_range: int  # range of the 3d skeleton in all 3 dimensions, e.g arm to arm reach etc. usually 3000mm
    flip: bool
    scale_factor: float
    rotation_factor: int
    img_pattern: str
    subsample: int
    subsampling_strategy: SAMPLE_METHOD
    gt_result_ratio: float = 1.0


def get_coco_dataset_cfg_default() -> CocoDatasetConfig:
    return CocoDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=30,
        use_mebow_orientation=False
    )


def get_tud_dataset_cfg_default() -> TudDatasetConfig:
    return TudDatasetConfig(
        flip=True,
        scale_factor=0.25,
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC
    )


def get_sim_dataset_cfg_default() -> PedRecDatasetConfig:
    return PedRecDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC
    )

def get_video_action_dataset_cfg_default() -> VideoActionDatasetConfig:
    return VideoActionDatasetConfig(
        flip=True,
        skeleton_3d_range=3000,
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC
    )

def get_sim_val_dataset_cfg_default() -> PedRecDatasetConfig:
    cfg = get_sim_dataset_cfg_default()
    cfg.subsample = 10
    return cfg



def get_h36m_dataset_cfg_default() -> PedRecDatasetConfig:
    return PedRecDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="img_{id}.{type}",
        subsample=10,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC
    )


def get_h36m_val_dataset_cfg_default() -> PedRecDatasetConfig:
    cfg = get_h36m_dataset_cfg_default()
    cfg.subsample = 64
    return cfg
