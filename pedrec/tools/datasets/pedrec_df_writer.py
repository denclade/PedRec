"""
Writes PedRec dataframes (the format of the published SIM / Human3.6m dataframes) from converted 3D pose datasets.

Every converter produces two files per split:

* ``<name>_<split>_pedrec.pkl``: frames with an extracted image (``images/<sequence>/img_00001.jpg``) for the
  PedRecNet training, usually every n-th frame
* ``<name>_<split>_seq.pkl``: all frames of the sequences resampled to ~30 fps, no images needed, for the temporal
  3D lifter

Conventions (as in the Human3.6m generator): 2D joints in image pixels; 3D joints in mm in camera coordinates
(x right, y up, z forward), relative to the hip center; per joint flags score, visible, supported (joints that the
dataset does not provide are not supported and ignored by the losses).
"""
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT, SKELETON_PEDREC_JOINTS
from pedrec.tools.datasets.dataset_generator_helper import get_column_names

NUM_JOINTS = len(SKELETON_PEDREC_JOINTS)
J = SKELETON_PEDREC_JOINT

# a mapping value is a source joint index or a function source joints (N x K x 3) -> N x 3
JointSource = Union[int, Callable[[np.ndarray], np.ndarray]]


def mean_of(*indices: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda joints: joints[:, list(indices)].mean(axis=1)


def farther_from(anchor: int, *candidates: int) -> Callable[[np.ndarray], np.ndarray]:
    """The candidate joint farthest from the anchor (e.g. toe tip vs. heel next to the ankle)."""
    def select(joints: np.ndarray) -> np.ndarray:
        options = joints[:, list(candidates)]
        distances = np.linalg.norm(options - joints[:, [anchor]], axis=-1)
        return options[np.arange(len(joints)), distances.argmax(axis=1)]
    return select


def to_pedrec_joints(joints: np.ndarray, mapping: Dict[J, JointSource]) -> (np.ndarray, np.ndarray):
    """
    :param joints: N x K x D source joints
    :return: N x 26 x D joints and the 26 supported flags
    """
    result = np.zeros((joints.shape[0], NUM_JOINTS, joints.shape[2]), dtype=np.float64)
    supported = np.zeros(NUM_JOINTS, dtype=bool)
    for joint, source in mapping.items():
        result[:, joint.value] = joints[:, source] if isinstance(source, (int, np.integer)) else source(joints)
        supported[joint.value] = True
    return result, supported


def complete_derived_joints(joints: np.ndarray, supported: np.ndarray) -> np.ndarray:
    """Hip center / neck / spine center from the hips and shoulders if the dataset does not provide them."""
    joints = joints.copy()
    if not supported[J.hip_center.value] and supported[J.left_hip.value] and supported[J.right_hip.value]:
        joints[:, J.hip_center.value] = joints[:, [J.left_hip.value, J.right_hip.value]].mean(axis=1)
        supported[J.hip_center.value] = True
    if not supported[J.neck.value] and supported[J.left_shoulder.value] and supported[J.right_shoulder.value]:
        joints[:, J.neck.value] = joints[:, [J.left_shoulder.value, J.right_shoulder.value]].mean(axis=1)
        supported[J.neck.value] = True
    if not supported[J.spine_center.value] and supported[J.hip_center.value] and supported[J.neck.value]:
        joints[:, J.spine_center.value] = joints[:, [J.hip_center.value, J.neck.value]].mean(axis=1)
        supported[J.spine_center.value] = True
    return joints


def to_mm(joints: np.ndarray) -> np.ndarray:
    """Detects meters / centimeters / millimeters from the shoulder to hip distance (~0.5 m) and returns mm."""
    torso = np.nanmedian(np.linalg.norm(joints[:, J.left_shoulder.value] - joints[:, J.left_hip.value], axis=-1))
    if torso < 5:  # meters
        return joints * 1000.0
    if torso < 150:  # centimeters
        return joints * 10.0
    return joints


def project(points_cam: np.ndarray, focal: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Pinhole projection of N x K x 3 camera coordinates (x right, y down, z forward) to pixels."""
    z = np.clip(points_cam[..., 2:3], 1e-6, None)
    return points_cam[..., :2] / z * focal + center


def bbs_from_joints(skeleton_2d: np.ndarray, supported: np.ndarray, img_width: int, img_height: int,
                    expand: float = 0.15) -> np.ndarray:
    """N x 6 center bbs (x, y, w, h, score, class) around the supported 2D joints, expanded and clipped."""
    xy = skeleton_2d[:, supported, :2]
    min_xy, max_xy = xy.min(axis=1), xy.max(axis=1)
    size = max_xy - min_xy
    min_xy = np.clip(min_xy - expand * size, 0, [img_width, img_height])
    max_xy = np.clip(max_xy + expand * size, 0, [img_width, img_height])
    size = max_xy - min_xy
    center = min_xy + size / 2
    return np.concatenate([center, size, np.ones((len(xy), 1)), np.zeros((len(xy), 1))], axis=1).astype(np.float32)


@dataclass
class Sequence:
    """One person in one camera: joints of all frames (native frame rate)."""
    dataset: str
    name: str  # unique sequence name, used as image directory
    subject_id: str
    joints_2d: np.ndarray  # N x 26 x 2 pixels
    joints_3d_cam: np.ndarray  # N x 26 x 3 mm, camera coordinates with y down (OpenCV)
    supported: np.ndarray  # 26 bools
    img_width: int
    img_height: int
    fps: float
    frame_ids: Optional[np.ndarray] = None  # original frame numbers (default 0..N-1)
    valid: Optional[np.ndarray] = None  # frames with a valid annotation
    extra: dict = field(default_factory=dict)


class PedRecDfWriter:
    def __init__(self, dataset_type: int = 0):
        self.dataset_type = dataset_type
        self.rows_images: List[list] = []
        self.rows_sequences: List[list] = []
        self.scene_id = 0

    def _rows(self, seq: Sequence, frames: np.ndarray, img_dir: str, img_ids: np.ndarray) -> List[list]:
        n = len(frames)
        supported = seq.supported.astype(np.float32)
        joints_2d = seq.joints_2d[frames]
        inside = (joints_2d[..., 0] >= 0) & (joints_2d[..., 0] < seq.img_width) & \
                 (joints_2d[..., 1] >= 0) & (joints_2d[..., 1] < seq.img_height)
        visible = inside * supported
        skeleton_2d = np.zeros((n, NUM_JOINTS, 5), dtype=np.float32)
        skeleton_2d[..., :2] = joints_2d * supported[None, :, None]
        skeleton_2d[..., 2] = visible
        skeleton_2d[..., 3] = visible
        skeleton_2d[..., 4] = supported
        joints_3d = seq.joints_3d_cam[frames].copy()
        joints_3d[..., 1] *= -1  # y up
        hip = joints_3d[:, J.hip_center.value].copy()
        skeleton_3d = np.zeros((n, NUM_JOINTS, 6), dtype=np.float32)
        skeleton_3d[..., :3] = (joints_3d - hip[:, None]) * supported[None, :, None]
        skeleton_3d[..., 3] = supported
        skeleton_3d[..., 4] = supported
        skeleton_3d[..., 5] = supported
        bbs = bbs_from_joints(joints_2d, seq.supported, seq.img_width, seq.img_height)
        rows = []
        for i in range(n):
            meta = [seq.dataset, self.dataset_type, self.scene_id, 0, 0, int(i + 1), int(i), img_dir,
                    int(img_ids[i]), "jpg", seq.subject_id, -1, -1, -1, -1, -1, -1, -1, True, -1]
            orientations = [0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0]  # not provided (not supported)
            rows.append(meta + bbs[i].tolist() + hip[i].tolist() + orientations +
                        skeleton_2d[i].reshape(-1).tolist() + skeleton_3d[i].reshape(-1).tolist())
        return rows

    def add(self, seq: Sequence, image_frames: Optional[np.ndarray] = None, sequence_fps: float = 30.0):
        """
        :param image_frames: frame indices that have an extracted image (images/<name>/img_<frame + 1>.jpg)
        """
        valid = seq.valid if seq.valid is not None else np.ones(len(seq.joints_2d), dtype=bool)
        valid = valid & np.isfinite(seq.joints_3d_cam).all(axis=(1, 2)) & (seq.joints_3d_cam[:, :, 2] > 0).any(axis=1)
        if image_frames is not None:
            frames = np.array([f for f in image_frames if valid[f]], dtype=int)
            if len(frames):
                self.rows_images += self._rows(seq, frames, os.path.join("images", seq.name), frames + 1)
        step = max(1, int(round(seq.fps / sequence_fps)))
        frames = np.flatnonzero(valid)[::1]
        frames = frames[frames % step == 0]
        if len(frames):
            self.rows_sequences += self._rows(seq, frames, os.path.join("images", seq.name), frames + 1)
        self.scene_id += 1

    @staticmethod
    def _to_df(rows: List[list]) -> pd.DataFrame:
        df = pd.DataFrame(data=rows, columns=get_column_names())
        df = df.astype({c: "float32" for c in df.columns if c.startswith(("skeleton", "bb_center", "bb_width",
                                                                          "bb_height", "bb_score", "env_position",
                                                                          "body_orientation", "head_orientation"))})
        # scene start / end (row indices of each scene, used by the temporal datasets)
        df["scene_start"] = df.groupby("scene_id").cumcount().pipe(lambda c: df.index - c)
        df["scene_end"] = df["scene_start"] + df.groupby("scene_id")["scene_id"].transform("size") - 1
        for column in ("dataset", "img_dir", "img_type", "subject_id"):
            df[column] = df[column].astype("category")
        return df

    def save(self, image_path: Optional[str], sequence_path: Optional[str]):
        for rows, path in ((self.rows_images, image_path), (self.rows_sequences, sequence_path)):
            if path is None:
                continue
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._to_df(rows).to_pickle(path)
            print(f"Wrote {len(rows)} frames to {path}")


def extract_frames(video_path: str, output_dir: str, frames: np.ndarray, quality: int = 92) -> np.ndarray:
    """Writes the given frames of a video as output_dir/img_<frame + 1>.jpg; returns the frames that exist."""
    os.makedirs(output_dir, exist_ok=True)
    wanted = set(int(f) for f in frames)
    missing = {f for f in wanted if not os.path.isfile(os.path.join(output_dir, f"img_{f + 1:05d}.jpg"))}
    if missing:
        cap = cv2.VideoCapture(video_path)
        index, last = 0, max(missing)
        while index <= last:
            ok = cap.grab()
            if not ok:
                break
            if index in missing:
                ok, frame = cap.retrieve()
                if ok:
                    cv2.imwrite(os.path.join(output_dir, f"img_{index + 1:05d}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, quality])
            index += 1
        cap.release()
    return np.array(sorted(f for f in wanted if os.path.isfile(os.path.join(output_dir, f"img_{f + 1:05d}.jpg"))),
                    dtype=int)
