import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS


@dataclass
class VideoActionDatasetEntryAnnotations(object):
    img_path: str = None
    skeleton_2d: np.ndarray = field(default_factory=lambda: np.zeros((len(SKELETON_PEDREC_JOINTS), 4), dtype=np.float32))
    skeleton_3d: np.ndarray = field(default_factory=lambda: np.zeros((len(SKELETON_PEDREC_JOINTS), 5), dtype=np.float32))
    body_orientation: np.ndarray = field(default_factory=lambda: np.zeros((4,), dtype=np.float32))
    head_orientation: np.ndarray = field(default_factory=lambda: np.zeros((4,), dtype=np.float32))
    scene_start: int = 0
    scene_end: int = 0
    actions: List[List[int]] = None
    frame_nr_local: int = 0

@dataclass()
class VideoActionDatasetEntries(object):
    img_dirs: np.ndarray
    scene_starts: np.ndarray
    scene_ends: np.ndarray
    frame_nr_locals: np.ndarray
    actions: np.ndarray
    body_orientations: np.ndarray
    head_orientations: np.ndarray
    skeleton2ds: np.ndarray
    skeleton3ds: np.ndarray


    def get_entry(self, idx: int) -> VideoActionDatasetEntryAnnotations:
        img_path = os.path.join(self.img_dirs[idx], f"{str(self.frame_nr_locals[idx]).zfill(5)}.jpg")
        return VideoActionDatasetEntryAnnotations(
            img_path=img_path,
            skeleton_2d=self.skeleton2ds[idx],
            skeleton_3d=self.skeleton3ds[idx],
            body_orientation=self.body_orientations[idx],
            head_orientation=self.head_orientations[idx],
            scene_start=self.scene_starts[idx],
            scene_end=self.scene_ends[idx],
            actions=[self.actions[idx]],
            frame_nr_local=self.frame_nr_locals[idx],
        )