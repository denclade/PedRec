import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

from pedrec.configs.dataset_configs import PedRecDatasetConfig
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS


@dataclass
class PedRecDatasetEntryAnnotations(object):
    img_path: str = None
    bb: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.float32))
    skeleton_2d: np.ndarray = field(default_factory=lambda: np.zeros((len(SKELETON_PEDREC_JOINTS), 4), dtype=np.float32))
    skeleton_3d: np.ndarray = field(default_factory=lambda: np.zeros((len(SKELETON_PEDREC_JOINTS), 5), dtype=np.float32))
    center: np.ndarray = field(default_factory=lambda: np.zeros((2,), dtype=np.float32))
    scale: np.ndarray = field(default_factory=lambda: np.zeros((2,), dtype=np.float32))
    env_position: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
    body_orientation: np.ndarray = field(default_factory=lambda: np.zeros((4,), dtype=np.float32))
    head_orientation: np.ndarray = field(default_factory=lambda: np.zeros((4,), dtype=np.float32))
    scene_id: int = 0
    scene_start: int = 0
    scene_end: int = 0
    actions: List[List[int]] = None
    movement: int = 0
    movement_speed: int = 0
    frame_nr_local: int = 0
    frame_nr_global: int = 0
    is_real_img: bool = True
    skeleton_2d_results: np.ndarray = field(default_factory=lambda: np.zeros((len(SKELETON_PEDREC_JOINTS), 4), dtype=np.float32))
    skeleton_3d_results: np.ndarray = field(default_factory=lambda: np.zeros((len(SKELETON_PEDREC_JOINTS), 5), dtype=np.float32))


@dataclass()
class PedRecDatasetEntries(object):
    img_ids: np.ndarray
    img_dirs: np.ndarray
    img_types: np.ndarray
    scene_ids: np.ndarray
    scene_starts: np.ndarray
    scene_ends: np.ndarray
    frame_nr_globals: np.ndarray
    frame_nr_locals: np.ndarray
    subject_ids: np.ndarray
    genders: np.ndarray
    skin_colors: np.ndarray
    sizes: np.ndarray
    bmis: np.ndarray
    ages: np.ndarray
    movements: np.ndarray
    movement_speeds: np.ndarray
    is_real_imgs: np.ndarray
    actions: np.ndarray
    bbs: np.ndarray
    env_positions: np.ndarray
    body_orientations: np.ndarray
    head_orientations: np.ndarray
    skeleton2ds: np.ndarray
    skeleton3ds: np.ndarray
    skeleton2ds_results: np.ndarray
    skeleton3ds_results: np.ndarray
    centers: np.ndarray = None
    scales: np.ndarray = None


    def get_entry(self, idx: int, cfg: PedRecDatasetConfig, gt_pose: bool = True) -> PedRecDatasetEntryAnnotations:
        img_dir = self.img_dirs[idx]
        cam_name = os.path.basename(os.path.normpath(img_dir))
        img_path = os.path.join(img_dir, cfg.img_pattern.format(id=str(self.img_ids[idx]).zfill(5),
                                                                type=self.img_types[idx],
                                                                cam_name=cam_name))
        entry = PedRecDatasetEntryAnnotations(
            img_path=img_path,
            bb=self.bbs[idx],
            skeleton_2d=self.skeleton2ds[idx],
            skeleton_3d=self.skeleton3ds[idx],
            center=self.centers[idx],
            scale=self.scales[idx],
            env_position=self.env_positions[idx],
            body_orientation=self.body_orientations[idx],
            head_orientation=self.head_orientations[idx],
            scene_id=self.scene_ids[idx],
            scene_start=self.scene_starts[idx],
            scene_end=self.scene_ends[idx],
            actions=self.actions[idx],
            movement=self.movements[idx],
            movement_speed=self.movement_speeds[idx],
            frame_nr_local=self.frame_nr_locals[idx],
            frame_nr_global=self.frame_nr_globals[idx],
            is_real_img=self.is_real_imgs[idx],
        )
        if self.skeleton2ds_results is not None:
            entry.skeleton_2d_results = self.skeleton2ds_results[idx]
        if self.skeleton3ds_results is not None:
            entry.skeleton_3d_results = self.skeleton3ds_results[idx]
        return entry
    #
    # return [
    #     "img_path",
    #     "center",
    #     "scale",
    #     "dataset_type",
    #     "scene_id",
    #     "frame_nr_global",
    #     "frame_nr_local",
    #     "subject_id",
    #     "gender",
    #     "skin_color",
    #     "size",
    #     "bmi",
    #     "age",
    #     "movement",
    #     "movement_speed",
    #     "is_real_img",
    #     "actions",
    #     "bb",
    #     "env_position",
    #     "body_orientation",
    #     "head_orientation",
    #     "skeleton2d",
    #     "skeleton3d",
    # ]
