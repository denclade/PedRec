from dataclasses import dataclass, field
from typing import Optional, Tuple

from pedrec.models.constants.sample_method import SAMPLE_METHOD
from pedrec.models.data_structures import ImageSize

@dataclass
class CocoDatasetConfig:
    flip: bool
    scale_factor: float
    rotation_factor: int
    use_mebow_orientation: bool
    half_body_prob: float = 0.3  # training: probability to crop to the upper or lower body
    random_erasing_prob: float = 0.3  # training: probability to erase a random rectangle (occlusion)
    color_jitter: float = 0.2  # training: max relative brightness / contrast / saturation change

@dataclass
class TudDatasetConfig:
    flip: bool
    scale_factor: float
    subsample: int
    subsampling_strategy: SAMPLE_METHOD
    half_body_prob: float = 0.3  # training: probability to crop to the upper or lower body
    random_erasing_prob: float = 0.3  # training: probability to erase a random rectangle (occlusion)
    color_jitter: float = 0.2  # training: max relative brightness / contrast / saturation change


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
    half_body_prob: float = 0.3  # training: probability to crop to the upper or lower body
    random_erasing_prob: float = 0.3  # training: probability to erase a random rectangle (occlusion)
    color_jitter: float = 0.2  # training: max relative brightness / contrast / saturation change
    subjects_include: Optional[Tuple[str, ...]] = None  # only these subject ids (None = all)
    subjects_exclude: Optional[Tuple[str, ...]] = None  # without these subject ids


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

# SIM validation: avatars of rt_validate_3d.pkl which are held out of the PedRecNet training. There are only 15 SIM
# avatars, 4 of them in rt_rom_01b.pkl; these 4 of the other 11 cover both genders, skin colors 0 / 1 / 5, sizes 1 / 2,
# BMI 0 - 3 and ages 1 / 2. Before, the complete rt_validate_3d.pkl was trained and validated (SIM results = training
# data). SIM-C01 uses the same avatars (not part of the PedRecNet training).
SIM_VAL_HOLDOUT_SUBJECTS: Tuple[str, ...] = (
    "2be5712e-adc4-4d3e-913f-735c650e2177",
    "3bc845aa-b5c8-4d09-a789-45c7e842e8ef",
    "7a6c0822-e163-4349-ae3b-4d17ec21087c",
    "a780b53f-6cbf-4131-9005-ba245d52c811",
)


def get_sim_train_dataset_cfg_default() -> PedRecDatasetConfig:
    cfg = get_sim_dataset_cfg_default()
    cfg.subjects_exclude = SIM_VAL_HOLDOUT_SUBJECTS
    return cfg


def get_sim_val_dataset_cfg_default() -> PedRecDatasetConfig:
    cfg = get_sim_dataset_cfg_default()
    cfg.subsample = 4
    cfg.subjects_include = SIM_VAL_HOLDOUT_SUBJECTS
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
