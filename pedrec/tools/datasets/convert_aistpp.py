"""
AIST++ (Li et al., ICCV 2021; https://google.github.io/aistplusplus_dataset) -> PedRec dataframes.

Street / jazz / break dance etc. of 30 dancers in 9 calibrated views, i.e. a very large range of motion (jumps, spins,
floor moves). Annotations (``mise run download:datasets:aistpp``, GitHub release v1.0 of google/aistplusplus_dataset):
``keypoints3d/<seq>.pkl`` (COCO 17 joints, world coordinates in cm, 60 fps),
``cameras/<env>.json`` (9 views: name, size, matrix, rotation (Rodrigues), translation, distortions),
``cameras/mapping.txt`` (sequence -> environment), ``splits/`` and ``ignore_list.txt``. Videos (60 fps) are named
like the sequence with ``cAll`` replaced by the view (``c01`` ... ``c09``).

    python pedrec/tools/datasets/convert_aistpp.py --root data/datasets/AIST++
"""
import sys

sys.path.append('.')

import argparse
import json
import os
import pickle
from typing import Optional

import cv2
import numpy as np

from pedrec.configs.default_paths import get_data_root
from pedrec.tools.datasets.pedrec_df_writer import (J, FrameExtractor, PedRecDfWriter, Sequence,
                                                    complete_derived_joints, image_scale, to_mm, to_pedrec_joints)

# COCO order
MAPPING_COCO = {
    J.nose: 0, J.left_eye: 1, J.right_eye: 2, J.left_ear: 3, J.right_ear: 4, J.left_shoulder: 5,
    J.right_shoulder: 6, J.left_elbow: 7, J.right_elbow: 8, J.left_wrist: 9, J.right_wrist: 10, J.left_hip: 11,
    J.right_hip: 12, J.left_knee: 13, J.right_knee: 14, J.left_ankle: 15, J.right_ankle: 16,
}
VIEWS = [f"c{i:02d}" for i in range(1, 10)]
FPS = 60.0


def _read_list(path: str):
    return [line.strip() for line in open(path) if line.strip()] if os.path.isfile(path) else []


def convert(root: str, output_dir: str, image_step: int, views=VIEWS, max_image_size: Optional[int] = None,
            workers: Optional[int] = None):
    annotations = root if os.path.isdir(os.path.join(root, "keypoints3d")) else os.path.join(root, "annotations")
    video_dir = os.path.join(root, "videos")
    mapping = dict(line.split() for line in _read_list(os.path.join(annotations, "cameras", "mapping.txt")))
    ignore = set(_read_list(os.path.join(annotations, "ignore_list.txt")))
    val = set(_read_list(os.path.join(annotations, "splits", "pose_val.txt"))) | \
        set(_read_list(os.path.join(annotations, "splits", "pose_test.txt")))
    writers = {"train": PedRecDfWriter(0), "val": PedRecDfWriter(1)}
    files = [f for f in sorted(os.listdir(os.path.join(annotations, "keypoints3d")))
             if os.path.splitext(f)[0] not in ignore and os.path.splitext(f)[0] in mapping]
    with FrameExtractor(workers, total=len(files) * len(views), desc="AIST++") as extractor:
        for file in files:
            seq = os.path.splitext(file)[0]
            with open(os.path.join(annotations, "keypoints3d", file), "rb") as f:
                data = pickle.load(f)
            joints_world = np.asarray(data.get("keypoints3d_optim", data.get("keypoints3d")), dtype=np.float64)
            with open(os.path.join(annotations, "cameras", f"{mapping[seq]}.json")) as f:
                cameras = {camera["name"]: camera for camera in json.load(f)}
            split = "val" if seq in val else "train"
            for view in views:
                if view not in cameras:
                    extractor.progress.update(1)
                    continue
                camera = cameras[view]
                rotation, _ = cv2.Rodrigues(np.asarray(camera["rotation"], dtype=np.float64))
                translation = np.asarray(camera["translation"], dtype=np.float64).reshape(1, 1, 3)
                joints_cam = np.matmul(joints_world, rotation.T) + translation
                points = np.nan_to_num(joints_world.reshape(-1, 3))
                joints_2d, _ = cv2.projectPoints(points, np.asarray(camera["rotation"], dtype=np.float64),
                                                 translation.reshape(3), np.asarray(camera["matrix"], dtype=np.float64),
                                                 np.asarray(camera["distortions"], dtype=np.float64))
                width, height = camera.get("size", [1920, 1080])
                scale = image_scale(width, height, max_image_size)
                joints_2d = joints_2d.reshape(joints_world.shape[0], -1, 2) * scale
                p2d, supported = to_pedrec_joints(joints_2d, MAPPING_COCO)
                p3d, _ = to_pedrec_joints(joints_cam, MAPPING_COCO)
                p3d = to_mm(p3d)
                p2d = complete_derived_joints(p2d, supported.copy())
                p3d = complete_derived_joints(p3d, supported)
                video_name = seq.replace("cAll", view)
                valid = np.isfinite(joints_world).all(axis=(1, 2))
                sequence = Sequence("AIST++", video_name, seq.split("_")[3] if seq.count("_") >= 3 else seq,
                                    p2d, p3d, supported, int(round(width * scale)), int(round(height * scale)), FPS,
                                    valid=valid)
                extractor.submit(os.path.join(video_dir, f"{video_name}.mp4"),
                                 os.path.join(output_dir, "images", video_name), np.arange(0, len(p2d), image_step),
                                 lambda frames, sequence=sequence, writer=writers[split]: writer.add(sequence, frames),
                                 scale)
    for split, writer in writers.items():
        writer.save(os.path.join(output_dir, f"aistpp_{split}_pedrec.pkl"),
                    os.path.join(output_dir, f"aistpp_{split}_seq.pkl"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(get_data_root(), "datasets", "AIST++"),
                        help="Directory with the annotations (keypoints3d/, cameras/, ...) and videos/.")
    parser.add_argument("--output-dir", default=None, help="Default: --root")
    parser.add_argument("--image-step", type=int, default=20, help="Extract every n-th frame (60 fps) as image.")
    parser.add_argument("--views", nargs="*", default=VIEWS)
    parser.add_argument("--max-image-size", type=int, default=0, help="Downscale the images (longer side), 0 = off.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel video decoders (default: CPUs, max 8).")
    args = parser.parse_args(argv)
    convert(args.root, args.output_dir or args.root, args.image_step, args.views, args.max_image_size, args.workers)


if __name__ == "__main__":
    main()
