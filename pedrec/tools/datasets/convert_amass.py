"""
AMASS (Mahmood et al., ICCV 2019; https://amass.is.tue.mpg.de) -> PedRec sequence dataframes for the temporal 3D
lifter (motion capture only, no images).

AMASS unifies many mocap datasets as SMPL-H parameters, among them ``MPI_Limits`` (PosePrior, Akhter & Black 2015:
the joint limits / range of motion of the human body) and large everyday motion collections (CMU, BMLmovi, KIT, ...).
The joints are computed with the SMPL-H body model (``smplx`` package, model files from https://mano.is.tue.mpg.de,
"Extended SMPL+H model", or the smplx layout) and seen by a virtual camera per sequence (random distance / height / direction, looking at
the person) to get the 2D inputs of the lifter.

    mise run data:convert:amass  # --root data/datasets/AMASS --body-models data/models/body_models
"""
import sys

sys.path.append('.')

import argparse
import glob
import os
import zlib
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from pedrec.configs.default_paths import get_data_root
from pedrec.tools.datasets.pedrec_df_writer import J, PedRecDfWriter, Sequence, to_pedrec_joints

IMG_WIDTH, IMG_HEIGHT = 1920, 1080
# SMPL-H body joints 0-21, followed by the mesh vertices below (indices 22-31 of the combined array)
VERTICES = ["nose", "leye", "reye", "lear", "rear", "LBigToe", "RBigToe", "lmiddle", "rmiddle"]
HEAD_TOP_VERTEX = 411  # top of the head of the SMPL mesh
MAPPING = {
    J.hip_center: 0, J.left_hip: 1, J.right_hip: 2, J.left_knee: 4, J.right_knee: 5, J.spine_center: 6,
    J.left_ankle: 7, J.right_ankle: 8, J.neck: 12, J.head_lower: 15, J.left_shoulder: 16, J.right_shoulder: 17,
    J.left_elbow: 18, J.right_elbow: 19, J.left_wrist: 20, J.right_wrist: 21,
    J.nose: 22, J.left_eye: 23, J.right_eye: 24, J.left_ear: 25, J.right_ear: 26, J.left_foot_end: 27,
    J.right_foot_end: 28, J.left_hand_end: 29, J.right_hand_end: 30, J.head_upper: 31,
}


def look_at_camera(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation world (z up) -> OpenCV camera (x right, y down, z forward)."""
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def virtual_camera(root_positions: np.ndarray, rng: np.random.Generator):
    center = root_positions.mean(axis=0)
    extent = np.ptp(root_positions[:, :2], axis=0).max()
    azimuth = rng.uniform(0, 2 * np.pi)
    distance = rng.uniform(4.0, 8.0) + extent
    height = rng.uniform(0.8, 2.0)
    position = np.array([center[0] + distance * np.cos(azimuth), center[1] + distance * np.sin(azimuth), height])
    target = np.array([center[0], center[1], rng.uniform(0.7, 1.1)])
    rotation = look_at_camera(position, target)
    focal = rng.uniform(1100.0, 1700.0)
    return rotation, position, focal


class BodyModels:
    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir, self.device = model_dir, device
        self.models: Dict[str, object] = {}

    def model_file(self, gender: str) -> str:
        """smplx layout (smplh/SMPLH_MALE.npz) or the AMASS "Extended SMPL+H" layout (smplh/male/model.npz)."""
        for g in (gender, "neutral"):
            for candidate in (os.path.join(self.model_dir, "smplh", f"SMPLH_{g.upper()}.npz"),
                              os.path.join(self.model_dir, "smplh", g, "model.npz"),
                              os.path.join(self.model_dir, g, "model.npz")):
                if os.path.isfile(candidate):
                    return candidate
        raise FileNotFoundError(f"No SMPL-H model for '{gender}' below {self.model_dir} (expected "
                                f"smplh/SMPLH_{gender.upper()}.npz or smplh/{gender}/model.npz)")

    def _load(self, gender: str):
        import smplx
        from smplx.utils import Struct
        path = self.model_file(gender)
        data = dict(np.load(path, allow_pickle=True))
        for side in "lr":  # the AMASS model files have no hand PCA, AMASS poses use the full hand pose anyway
            data.setdefault(f"hands_components{side}", np.eye(45, dtype=np.float32))
            data.setdefault(f"hands_mean{side}", np.zeros(45, dtype=np.float32))
        return smplx.SMPLH(model_path=path, data_struct=Struct(**data), gender=gender, use_pca=False,
                           flat_hand_mean=True, num_betas=16, ext="npz")

    def joints(self, data, step: int, max_frames: int, chunk: int = 1000) -> np.ndarray:
        import torch
        from smplx.vertex_ids import vertex_ids
        gender = str(np.asarray(data["gender"]).astype(str)).replace("b'", "").replace("'", "")
        gender = gender if gender in ("male", "female") else "neutral"
        if gender not in self.models:
            self.models[gender] = self._load(gender).to(self.device).eval()
        model = self.models[gender]
        poses = np.asarray(data["poses"], dtype=np.float32)[::step][:max_frames]
        trans = np.asarray(data["trans"], dtype=np.float32)[::step][:max_frames]
        # smplx limits num_betas (e.g. to 10 for models with less than 300 shape components), use what it supports
        betas = np.zeros(model.num_betas, dtype=np.float32)
        source_betas = np.asarray(data["betas"], dtype=np.float32)[:model.num_betas]
        betas[:len(source_betas)] = source_betas
        ids = [vertex_ids["smplh"][name] for name in VERTICES] + [HEAD_TOP_VERTEX]
        out = []
        with torch.no_grad():
            for start in range(0, len(poses), chunk):
                p = torch.from_numpy(poses[start:start + chunk]).to(self.device)
                n = len(p)
                result = model(global_orient=p[:, :3], body_pose=p[:, 3:66], left_hand_pose=p[:, 66:111],
                               right_hand_pose=p[:, 111:156], transl=torch.from_numpy(trans[start:start + n]),
                               betas=torch.from_numpy(betas).expand(n, -1), return_verts=True)
                out.append(torch.cat([result.joints[:, :22], result.vertices[:, ids]], dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 32, 3))


def convert(root: str, output_dir: str, body_models: str, datasets: Optional[List[str]], fps: float = 30.0,
            max_frames: int = 3000, val_every: int = 20, seed: int = 0, device: str = "cpu"):
    rng = np.random.default_rng(seed)
    models = BodyModels(body_models, device)
    files = sorted(glob.glob(os.path.join(root, "**", "*_poses.npz"), recursive=True))
    if datasets:
        files = [f for f in files if os.path.relpath(f, root).split(os.sep)[0] in datasets]
    writers = {"train": PedRecDfWriter(0), "val": PedRecDfWriter(1)}
    for path in tqdm(files, desc="AMASS", unit="seq", dynamic_ncols=True):
        data = np.load(path)
        if "poses" not in data or len(data["poses"]) < 10:
            continue
        source_fps = float(data["mocap_framerate"]) if "mocap_framerate" in data else 120.0
        step = max(1, int(round(source_fps / fps)))
        joints_world = models.joints(data, step, max_frames)
        if len(joints_world) < 10:
            continue
        rotation, position, focal = virtual_camera(joints_world[:, 0], rng)
        joints_cam = np.matmul(joints_world - position, rotation.T) * 1000.0  # mm, OpenCV camera
        joints_2d = joints_cam[..., :2] / np.clip(joints_cam[..., 2:3], 1e-3, None) * focal + \
            np.array([IMG_WIDTH / 2, IMG_HEIGHT / 2])
        p2d, supported = to_pedrec_joints(joints_2d, MAPPING)
        p3d, _ = to_pedrec_joints(joints_cam, MAPPING)
        name = os.path.splitext(os.path.relpath(path, root))[0].replace(os.sep, "_")
        dataset = os.path.relpath(path, root).split(os.sep)[0]
        # stable split, independent of which AMASS subsets are converted
        split = "val" if zlib.crc32(name.encode()) % val_every == 0 else "train"
        writers[split].add(Sequence("AMASS", name, dataset, p2d, p3d, supported, IMG_WIDTH, IMG_HEIGHT,
                                    source_fps / step), image_frames=None, sequence_fps=fps)
    for split, writer in writers.items():
        writer.save(None, os.path.join(output_dir, f"amass_{split}_seq.pkl"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(get_data_root(), "datasets", "AMASS"),
                        help="Directory with the extracted AMASS datasets.")
    parser.add_argument("--body-models", default=os.path.join(get_data_root(), "models", "body_models"),
                        help="Directory with smplh/{male,female,neutral}/model.npz (AMASS) or smplh/SMPLH_MALE.npz (smplx).")
    parser.add_argument("--output-dir", default=None, help="Default: --root")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of the AMASS datasets, e.g. MPI_Limits CMU BMLmovi (default: all found).")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=3000, help="Frames per sequence (after resampling).")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    convert(args.root, args.output_dir or args.root, args.body_models, args.datasets, args.fps, args.max_frames,
            device=args.device)


if __name__ == "__main__":
    main()
