"""
Fit3D (Fieraru et al., AAAI 2021; https://fit3d.imar.ro), also HumanSC3D / CHI3D (same layout) -> PedRec dataframes.

Fitness exercises with a large range of motion (squats, lunges, push-ups, jumping jacks, ...): 4 cameras, mocap
ground truth (``joints3d_25``, Human3.6m joint order + 2 points per foot and hand). Layout as provided by IMAR
(https://github.com/sminchisescu-research/imar_vision_datasets_tools):

    <root>/train/<subject>/videos/<camera>/<action>.mp4
    <root>/train/<subject>/joints3d_25/<action>.json          {"joints3d_25": frames x 25 x 3, world coordinates}
    <root>/train/<subject>/camera_parameters/<camera>/<action>.json
        {"extrinsics": {"R", "T"}, "intrinsics_w_distortion": {"f", "c", "k", "p"}, ...}

Only the train split has public ground truth; the last subject(s) are used for validation (``--val-subjects``).

    python pedrec/tools/datasets/convert_fit3d.py --root data/datasets/Fit3D
"""
import sys

sys.path.append('.')

import argparse
import json
import os
from typing import List, Optional

import numpy as np

from pedrec.configs.default_paths import get_data_root
from pedrec.tools.datasets.pedrec_df_writer import (J, FrameExtractor, PedRecDfWriter, Sequence, farther_from,
                                                    image_scale,
                                                    to_mm, to_pedrec_joints)

# Human3.6m order: 0 pelvis, 1 r_hip, 2 r_knee, 3 r_ankle, 4 l_hip, 5 l_knee, 6 l_ankle, 7 spine, 8 thorax, 9 neck /
# nose, 10 head top, 11 l_shoulder, 12 l_elbow, 13 l_wrist, 14 r_shoulder, 15 r_elbow, 16 r_wrist;
# 17 / 18 right foot, 19 / 20 left foot, 21 / 22 left hand, 23 / 24 right hand
MAPPING_25 = {
    J.hip_center: 0, J.right_hip: 1, J.right_knee: 2, J.right_ankle: 3, J.left_hip: 4, J.left_knee: 5,
    J.left_ankle: 6, J.spine_center: 7, J.neck: 8, J.head_lower: 9, J.head_upper: 10, J.left_shoulder: 11,
    J.left_elbow: 12, J.left_wrist: 13, J.right_shoulder: 14, J.right_elbow: 15, J.right_wrist: 16,
    J.right_foot_end: farther_from(3, 17, 18), J.left_foot_end: farther_from(6, 19, 20),
    J.left_hand_end: farther_from(13, 21, 22), J.right_hand_end: farther_from(16, 23, 24),
}
FPS = 50.0


def world_to_camera(joints_world: np.ndarray, camera: dict) -> np.ndarray:
    rotation = np.asarray(camera["extrinsics"]["R"], dtype=np.float64)
    translation = np.asarray(camera["extrinsics"]["T"], dtype=np.float64).reshape(1, 1, 3)
    return np.matmul(joints_world - translation, rotation.T)


def project_with_distortion(points_cam: np.ndarray, intrinsics: dict) -> np.ndarray:
    """Projection with radial / tangential distortion as in the IMAR toolkit (project_3d_to_2d)."""
    f = np.asarray(intrinsics["f"], dtype=np.float64).reshape(-1)
    c = np.asarray(intrinsics["c"], dtype=np.float64).reshape(-1)
    shape = points_cam.shape
    points = points_cam.reshape(-1, 3)
    x = points[:, :2] / np.clip(points[:, 2:3], 1e-6, None)
    if "k" in intrinsics and "p" in intrinsics:
        k = np.asarray(intrinsics["k"], dtype=np.float64).reshape(-1)
        p = np.asarray(intrinsics["p"], dtype=np.float64).reshape(-1)[[1, 0]]
        r2 = np.sum(x ** 2, axis=1)
        radial = 1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3
        tan = x @ p
        x = x * (tan + radial)[:, None] + r2[:, None] * p[None]
    return (f * x + c).reshape(*shape[:-1], 2)


def _video_size(path: str):
    import cv2
    cap = cv2.VideoCapture(path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 900, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 900)
    cap.release()
    return size


def convert(root: str, output_dir: str, image_step: int, val_subjects: List[str], dataset_name: str = "Fit3D",
            max_image_size: Optional[int] = None, workers: Optional[int] = None):
    train_dir = os.path.join(root, "train")
    subjects = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    writers = {"train": PedRecDfWriter(0), "val": PedRecDfWriter(1)}
    total = sum(len(os.listdir(os.path.join(train_dir, s, "joints3d_25"))) *
                len(os.listdir(os.path.join(train_dir, s, "camera_parameters"))) for s in subjects)
    with FrameExtractor(workers, total=total, desc=dataset_name) as extractor:
        for subject in subjects:
            split = "val" if subject in val_subjects else "train"
            subject_dir = os.path.join(train_dir, subject)
            for joints_file in sorted(os.listdir(os.path.join(subject_dir, "joints3d_25"))):
                action = os.path.splitext(joints_file)[0]
                with open(os.path.join(subject_dir, "joints3d_25", joints_file)) as f:
                    joints_all = np.asarray(json.load(f)["joints3d_25"], dtype=np.float64)
                persons = joints_all if joints_all.ndim == 4 else joints_all[None]  # CHI3D: 2 persons per sequence
                for camera_name in sorted(os.listdir(os.path.join(subject_dir, "camera_parameters"))):
                    camera_file = os.path.join(subject_dir, "camera_parameters", camera_name, f"{action}.json")
                    if not os.path.isfile(camera_file):
                        extractor.progress.update(1)
                        continue
                    with open(camera_file) as f:
                        camera = json.load(f)
                    video = os.path.join(subject_dir, "videos", camera_name, f"{action}.mp4")
                    width, height = _video_size(video) if os.path.isfile(video) else (900, 900)
                    scale = image_scale(width, height, max_image_size)
                    sequences = []
                    for person, joints_world in enumerate(persons):
                        joints_cam = world_to_camera(joints_world, camera)
                        intrinsics = camera.get("intrinsics_w_distortion") or camera["intrinsics_wo_distortion"]
                        joints_2d = project_with_distortion(joints_cam, intrinsics) * scale
                        p2d, supported = to_pedrec_joints(joints_2d, MAPPING_25)
                        p3d, _ = to_pedrec_joints(joints_cam, MAPPING_25)
                        p3d = to_mm(p3d)
                        sequences.append(Sequence(dataset_name, f"{subject}_{action.replace(' ', '_')}_{camera_name}",
                                                  subject, p2d, p3d, supported, int(round(width * scale)),
                                                  int(round(height * scale)), FPS,
                                                  extra={"person": person} if len(persons) > 1 else {}))
                    name = sequences[0].name

                    def add(frames, sequences=sequences, writer=writers[split]):
                        for sequence in sequences:  # all persons share the images of the camera
                            writer.add(sequence, frames)
                    extractor.submit(video, os.path.join(output_dir, "images", name),
                                     np.arange(0, len(sequences[0].joints_2d), image_step), add, scale)
    prefix = dataset_name.lower()
    for split, writer in writers.items():
        writer.save(os.path.join(output_dir, f"{prefix}_{split}_pedrec.pkl"),
                    os.path.join(output_dir, f"{prefix}_{split}_seq.pkl"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(get_data_root(), "datasets", "Fit3D"))
    parser.add_argument("--output-dir", default=None, help="Default: --root")
    parser.add_argument("--image-step", type=int, default=10, help="Extract every n-th frame as image.")
    parser.add_argument("--val-subjects", nargs="*", default=["s11"], help="Subjects used for validation.")
    parser.add_argument("--name", default="Fit3D", help="Dataset name (Fit3D, HumanSC3D, CHI3D).")
    parser.add_argument("--max-image-size", type=int, default=0, help="Downscale the images (longer side), 0 = off.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel video decoders (default: CPUs, max 8).")
    args = parser.parse_args(argv)
    convert(args.root, args.output_dir or args.root, args.image_step, args.val_subjects, args.name,
            args.max_image_size, args.workers)


if __name__ == "__main__":
    main()
