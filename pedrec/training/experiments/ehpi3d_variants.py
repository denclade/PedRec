"""
Registry of the EHPI3D (action recognition) training variants on the SIM-C01 dataset.

A variant is the combination of a *source* (which skeletons are used to build the EHPI images) and a *temporal*
setting (frame rate / temporal field). Variant names are ``<source>`` or ``<source>_<temporal>``, e.g.
``gt_pred_64frames`` (the published model).
"""
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from pedrec.configs.dataset_configs import PedRecTemporalDatasetConfig, VideoActionDatasetConfig
from pedrec.models.constants.sample_method import SAMPLE_METHOD
from pedrec.models.data_structures import ImageSize


@dataclass(frozen=True)
class Ehpi3DVariant:
    name: str
    description: str
    gt_result_ratio: float  # 1 = ground truth skeletons only, 0 = PedRecNet predictions only, else mixing ratio
    use_unit_skeleton: bool = True
    min_joint_score: float = 0.0
    use_ehpi_videos: bool = False  # additionally train on the real EHPI video dataset
    frame_sampling: int = 1  # 2 = every second frame (15 fps instead of 30 fps)
    temporal_field: Tuple[int, int] = (32, 32)  # (number of frames, number of joints)

    @property
    def experiment_name(self) -> str:
        return f"ehpi_3d_sim_c01_actionrec_{self.name}"

    def get_pedrec_cfg(self) -> PedRecTemporalDatasetConfig:
        return PedRecTemporalDatasetConfig(
            flip=True,
            scale_factor=0.25,
            rotation_factor=0,
            skeleton_3d_range=3000,
            img_pattern="view_{cam_name}-frame_{id}.{type}",
            subsample=1,
            subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
            gt_result_ratio=self.gt_result_ratio,
            use_unit_skeleton=self.use_unit_skeleton,
            min_joint_score=self.min_joint_score,
            add_2d=False,
            frame_sampling=self.frame_sampling,
            temporal_field=ImageSize(*self.temporal_field)
        )

    def get_vid_cfg(self) -> Optional[VideoActionDatasetConfig]:
        if not self.use_ehpi_videos:
            return None
        return VideoActionDatasetConfig(
            flip=True,
            skeleton_3d_range=3000,
            subsample=1,
            subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
            use_unit_skeleton=self.use_unit_skeleton,
            min_joint_score=self.min_joint_score,
            add_2d=False,
            frame_sampling=self.frame_sampling,
            temporal_field=ImageSize(*self.temporal_field)
        )


_SOURCES: List[Ehpi3DVariant] = [
    Ehpi3DVariant("gt", "Ground truth skeletons only.", gt_result_ratio=1.0),
    Ehpi3DVariant("pred", "PedRecNet predicted skeletons only.", gt_result_ratio=0.0),
    Ehpi3DVariant("gt_pred", "65% ground truth / 35% predicted skeletons.", gt_result_ratio=0.65),
    Ehpi3DVariant("gt_pred_ehpi2dvids", "gt_pred + real EHPI video dataset.", gt_result_ratio=0.65,
                  use_ehpi_videos=True),
    Ehpi3DVariant("gt_pred_no_unit_skeleton", "gt_pred without unit skeleton normalization.", gt_result_ratio=0.65,
                  use_unit_skeleton=False),
    Ehpi3DVariant("gt_pred_zero_by_score", "gt_pred, joints with score < 0.4 are zeroed.", gt_result_ratio=0.65,
                  min_joint_score=0.4),
]

_TEMPORAL = {
    "": {},
    "15fps": {"frame_sampling": 2},
    "64frames": {"temporal_field": (64, 32)},
}


def _build_variants() -> Dict[str, Ehpi3DVariant]:
    variants: Dict[str, Ehpi3DVariant] = {}
    for source in _SOURCES:
        for suffix, overrides in _TEMPORAL.items():
            name = source.name if suffix == "" else f"{source.name}_{suffix}"
            description = source.description if suffix == "" else f"{source.description} ({suffix})"
            variants[name] = replace(source, name=name, description=description, **overrides)
    return variants


VARIANTS: Dict[str, Ehpi3DVariant] = _build_variants()

DEFAULT_VARIANT = "gt_pred_64frames"  # the published EHPI3D model


def get_variant(name: str) -> Ehpi3DVariant:
    key = name[len("ehpi_3d_sim_c01_actionrec_"):] if name.startswith("ehpi_3d_sim_c01_actionrec_") else name
    if key not in VARIANTS:
        raise KeyError(f"Unknown EHPI3D variant '{name}'. Available: {', '.join(VARIANTS.keys())}")
    return VARIANTS[key]


def format_variant_table() -> str:
    lines = [f"{'variant':36} {'gt ratio':8} {'frames':6} {'fps':4} {'unit':5} {'min score':9} description", "-" * 110]
    for variant in VARIANTS.values():
        fps = 30 // variant.frame_sampling
        lines.append(f"{variant.name:36} {variant.gt_result_ratio:<8} {variant.temporal_field[0]:<6} {fps:<4} "
                     f"{str(variant.use_unit_skeleton):5} {variant.min_joint_score:<9} {variant.description}")
    return "\n".join(lines)
