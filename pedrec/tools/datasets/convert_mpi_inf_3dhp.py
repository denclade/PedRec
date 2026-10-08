"""
MPI-INF-3DHP (Mehta et al., 3DV 2017; https://vcai.mpi-inf.mpg.de/3dhp-dataset/) -> PedRec dataframes.

Training set: subjects S1-S8, sequences Seq1 / Seq2, the 8 chest height cameras 0, 1, 2, 4, 5, 6, 7, 8 (studio with
green screen, 25 / 50 fps). Per sequence ``annot.mat`` with ``annot2`` (2D, per camera) and ``annot3`` (3D in camera
coordinates, mm) of 28 joints; images from ``imageSequence/video_<cam>.avi``.
Test set (validation): TS1-TS6 (studio and outdoor), ``annot_data.mat`` (17 joints, ``valid_frame``) and
``imageSequence/img_<frame>.jpg``.

    python pedrec/tools/datasets/convert_mpi_inf_3dhp.py --root data/datasets/MPI-INF-3DHP
"""
import sys

sys.path.append('.')

import argparse
import os
import shutil
from typing import Optional

import numpy as np
from tqdm import tqdm

from pedrec.configs.default_paths import get_data_root
from pedrec.tools.datasets.pedrec_df_writer import (J, FrameExtractor, PedRecDfWriter, Sequence, _check_scale,
                                                    image_scale, to_pedrec_joints)

TRAIN_CAMERAS = [0, 1, 2, 4, 5, 6, 7, 8]
# 28 joints: 0 spine3, 1 spine4, 2 spine2, 3 spine, 4 pelvis, 5 neck, 6 head, 7 head_top, 8 l_clavicle, 9 l_shoulder,
# 10 l_elbow, 11 l_wrist, 12 l_hand, 13 r_clavicle, 14 r_shoulder, 15 r_elbow, 16 r_wrist, 17 r_hand, 18 l_hip,
# 19 l_knee, 20 l_ankle, 21 l_foot, 22 l_toe, 23 r_hip, 24 r_knee, 25 r_ankle, 26 r_foot, 27 r_toe
MAPPING_28 = {
    J.left_shoulder: 9, J.right_shoulder: 14, J.left_elbow: 10, J.right_elbow: 15, J.left_wrist: 11,
    J.right_wrist: 16, J.left_hip: 18, J.right_hip: 23, J.left_knee: 19, J.right_knee: 24, J.left_ankle: 20,
    J.right_ankle: 25, J.hip_center: 4, J.spine_center: 0, J.neck: 5, J.head_lower: 6, J.head_upper: 7,
    J.left_foot_end: 22, J.right_foot_end: 27, J.left_hand_end: 12, J.right_hand_end: 17,
}
# test set, 17 joints: head_top, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist, r_hip, r_knee,
# r_ankle, l_hip, l_knee, l_ankle, pelvis, spine, head
MAPPING_17 = {
    J.head_upper: 0, J.neck: 1, J.right_shoulder: 2, J.right_elbow: 3, J.right_wrist: 4, J.left_shoulder: 5,
    J.left_elbow: 6, J.left_wrist: 7, J.right_hip: 8, J.right_knee: 9, J.right_ankle: 10, J.left_hip: 11,
    J.left_knee: 12, J.left_ankle: 13, J.hip_center: 14, J.spine_center: 15, J.head_lower: 16,
}


def _video_info(path: str):
    import cv2
    cap = cv2.VideoCapture(path)
    info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return info


def _image_size(path: str, default):
    import cv2
    img = cv2.imread(path) if os.path.isfile(path) else None
    return (img.shape[1], img.shape[0]) if img is not None else default


def convert_train(root: str, output_dir: str, image_step: int, subjects=range(1, 9), cameras=TRAIN_CAMERAS,
                  max_image_size: Optional[int] = 1024, workers: Optional[int] = None):
    from scipy.io import loadmat
    writer = PedRecDfWriter(dataset_type=0)
    sequences = [(subject, seq) for subject in subjects for seq in (1, 2)]
    with FrameExtractor(workers, total=len(sequences) * len(cameras), desc="MPI-INF-3DHP train") as extractor:
        for subject, seq in sequences:
            seq_dir = os.path.join(root, f"S{subject}", f"Seq{seq}")
            annot_path = os.path.join(seq_dir, "annot.mat")
            if not os.path.isfile(annot_path):
                extractor.progress.write(f"missing {annot_path}, skipped")
                extractor.progress.update(len(cameras))
                continue
            annot = loadmat(annot_path)
            for cam in cameras:
                video = os.path.join(seq_dir, "imageSequence", f"video_{cam}.avi")
                joints_2d = np.asarray(annot["annot2"][cam][0], dtype=np.float64).reshape(-1, 28, 2)
                joints_3d = np.asarray(annot["annot3"][cam][0], dtype=np.float64).reshape(-1, 28, 3)
                width, height, fps = _video_info(video) if os.path.isfile(video) else (2048, 2048, 25.0)
                scale = image_scale(width, height, max_image_size)
                p2d, supported = to_pedrec_joints(joints_2d * scale, MAPPING_28)
                p3d, _ = to_pedrec_joints(joints_3d, MAPPING_28)
                name = f"S{subject}_Seq{seq}_cam{cam}"
                sequence = Sequence("MPI-INF-3DHP", name, f"S{subject}", p2d, p3d, supported,
                                    int(round(width * scale)), int(round(height * scale)), fps)
                extractor.submit(video, os.path.join(output_dir, "images", name), np.arange(0, len(p2d), image_step),
                                 lambda frames, sequence=sequence: writer.add(sequence, frames), scale)
    writer.save(os.path.join(output_dir, "mpi_inf_3dhp_train_pedrec.pkl"),
                os.path.join(output_dir, "mpi_inf_3dhp_train_seq.pkl"))


def convert_test(root: str, output_dir: str, image_step: int, max_image_size: Optional[int] = 1024):
    import h5py
    writer = PedRecDfWriter(dataset_type=1)
    for subject in range(1, 7):
        ts_dir = os.path.join(root, "mpi_inf_3dhp_test_set", f"TS{subject}")
        annot_path = os.path.join(ts_dir, "annot_data.mat")
        if not os.path.isfile(annot_path):
            print(f"missing {annot_path}, skipped")
            continue
        with h5py.File(annot_path, "r") as f:
            joints_2d = np.asarray(f["annot2"]).reshape(-1, 17, 2)
            joints_3d = np.asarray(f["annot3"]).reshape(-1, 17, 3)
            valid = np.asarray(f["valid_frame"]).reshape(-1).astype(bool)
        p2d, supported = to_pedrec_joints(joints_2d, MAPPING_17)
        p3d, _ = to_pedrec_joints(joints_3d, MAPPING_17)
        name = f"TS{subject}"
        width, height = _image_size(os.path.join(ts_dir, "imageSequence", "img_000001.jpg"),
                                    (2048, 2048) if subject <= 4 else (1920, 1080))
        scale = image_scale(width, height, max_image_size)
        image_dir = os.path.join(output_dir, "images", name)
        os.makedirs(image_dir, exist_ok=True)
        _check_scale(image_dir, scale)
        frames = [f for f in range(0, len(p2d), image_step) if valid[f]]
        for f in tqdm(frames, desc=f"MPI-INF-3DHP {name}", unit="img", dynamic_ncols=True):
            # images are provided as files: copy / resize into the common layout
            source = os.path.join(ts_dir, "imageSequence", f"img_{f + 1:06d}.jpg")
            target = os.path.join(image_dir, f"img_{f + 1:05d}.jpg")
            if not os.path.isfile(source) or os.path.isfile(target):
                continue
            if scale == 1.0:
                shutil.copyfile(source, target)
            else:
                import cv2
                img = cv2.imread(source)
                img = cv2.resize(img, (int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))),
                                 interpolation=cv2.INTER_AREA)
                cv2.imwrite(target, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        image_frames = np.array([f for f in frames if os.path.isfile(os.path.join(image_dir, f"img_{f + 1:05d}.jpg"))])
        fps = 50.0 if subject <= 4 else 25.0
        writer.add(Sequence("MPI-INF-3DHP", name, name, p2d * scale, p3d, supported, int(round(width * scale)),
                            int(round(height * scale)), fps, valid=valid), image_frames)
    writer.save(os.path.join(output_dir, "mpi_inf_3dhp_val_pedrec.pkl"),
                os.path.join(output_dir, "mpi_inf_3dhp_val_seq.pkl"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(get_data_root(), "datasets", "MPI-INF-3DHP"),
                        help="Directory with S1..S8 / test set.")
    parser.add_argument("--output-dir", default=None, help="Default: --root")
    parser.add_argument("--image-step", type=int, default=10, help="Extract every n-th frame as image.")
    parser.add_argument("--max-image-size", type=int, default=1024,
                        help="Downscale the 2048x2048 images to this size (longer side), 0 = original size.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel video decoders (default: CPUs, max 8).")
    parser.add_argument("--cameras", nargs="*", type=int, default=TRAIN_CAMERAS)
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.root
    convert_train(args.root, output_dir, args.image_step, cameras=args.cameras, max_image_size=args.max_image_size,
                  workers=args.workers)
    convert_test(args.root, output_dir, args.image_step, max_image_size=args.max_image_size)


if __name__ == "__main__":
    main()
