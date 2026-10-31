"""Generate canonical 2D/3D PedRecNet inputs for every image-backed lifter dataset.

Only existing images with valid person boxes are processed. Sparse result rows
carry explicit original_index into the full sequence dataframe. No detector,
GT convention adapter, image augmentation or synthetic predictions are applied.
"""
import argparse
import json
import logging
import os
import sys

sys.path.append(".")

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.datasets.lifter_dataset_specs import dataset_specs, dataframe_signature, file_sha256
from pedrec.evaluations.eval_helper import get_total_coords
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.networks.net_pedrec.pedrec_net_factory import load_pedrec_net
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.train_pedrec import IMAGENET_TRANSFORM
from pedrec.utils.augmentation_helper import get_affine_transform
from pedrec.utils.bb_helper import bbs_to_centers_scales
from pedrec.utils.log_helper import configure_logger
from pedrec.utils.pandas_helper import read_pedrec_df
from pedrec.utils.pedrec_dataset_helper import get_filter_bb, get_filter_skeleton2d, get_filter_skeleton3d

logger = logging.getLogger(__name__)
N_JOINTS = len(SKELETON_PEDREC_JOINTS)


class SequenceImages(Dataset):
    """Read available sequence images directly; preserve full positional row IDs."""

    def __init__(self, df, root, pattern, input_size, max_frames=None):
        self.input_size = input_size
        boxes = df[get_filter_bb(df)].to_numpy(np.float32)
        visible = df[[c for c in get_filter_skeleton2d(df) if c.endswith("_visible")]].to_numpy().sum(axis=1)
        valid = np.isfinite(boxes).all(axis=1) & (boxes[:, 4] >= 1) & (visible >= 3)
        valid &= (boxes[:, 2] > 1) & (boxes[:, 3] > 1)
        self.rows, self.paths = [], []
        # List each directory once, rather than stat millions of absent frames.
        candidates = df.loc[valid, ["img_dir", "img_id", "img_type"]]
        for directory, group in tqdm(candidates.groupby("img_dir", observed=True, sort=False),
                                     desc="Find sequence images", leave=False):
            path = os.path.join(root, str(directory))
            if not os.path.isdir(path):
                continue
            filenames = set(os.listdir(path))
            camera = os.path.basename(os.path.normpath(str(directory)))
            for row, image_id, image_type in zip(group.index, group.img_id, group.img_type):
                filename = pattern.format(id=f"{int(image_id):05d}", type=image_type, cam_name=camera)
                if filename in filenames:
                    self.rows.append(int(row))
                    self.paths.append(os.path.join(path, filename))
                    if max_frames is not None and len(self.rows) >= max_frames:
                        break
            if max_frames is not None and len(self.rows) >= max_frames:
                break
        self.rows = np.asarray(self.rows, dtype=np.int64)
        self.centers, self.scales = bbs_to_centers_scales(boxes[self.rows], input_size)
        logger.info("Available images: %d/%d valid boxes (%d sequence rows)", len(self.rows), valid.sum(), len(df))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image = cv2.imread(self.paths[index])
        if image is None:
            raise OSError(f"Cannot read image: {self.paths[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        trans = get_affine_transform(self.centers[index], self.scales[index], 0, self.input_size)
        crop = cv2.warpAffine(image, trans, (self.input_size.width, self.input_size.height), flags=cv2.INTER_LINEAR)
        return IMAGENET_TRANSFORM(crop), self.rows[index], self.centers[index], self.scales[index]


@torch.inference_mode()
def write_predictions(spec, model, device, weights_hash, batch_size=64, num_workers=4, max_frames=None):
    # Positional indices, never dataframe labels, are shared with PoseSequenceDataset.
    df = read_pedrec_df(spec.dataframe).reset_index(drop=True)
    size = PedRecNetConfig().model.input_size
    dataset = SequenceImages(df, os.path.dirname(spec.dataframe), spec.image_pattern, size, max_frames)
    p2_columns, p3_columns = get_filter_skeleton2d(df), get_filter_skeleton3d(df)
    orientation_columns = [f"{part}_orientation_{value}" for part in ("body", "head")
                           for value in ("theta", "phi", "score", "theta_supported", "phi_supported")]
    columns = p2_columns + p3_columns + orientation_columns
    values = np.zeros((len(df) if spec.dense_results else len(dataset), len(columns)), dtype=np.float32)
    # Read the small flag blocks once; retaining all GT coordinates is unnecessary.
    flags2 = df[[c for c in p2_columns if c.endswith(("_visible", "_supported"))]].to_numpy(np.float32)
    flags3 = df[[c for c in p3_columns if c.endswith(("_visible", "_supported"))]].to_numpy(np.float32)
    model.eval()
    offset = 0
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                        pin_memory=device.type == "cuda")
    for images, rows, centers, scales in tqdm(loader, desc=f"{spec.title} {spec.split}"):
        outputs = model(images.to(device, non_blocking=True))
        p2, p3, orientation = [o.float().cpu().numpy() for o in outputs[:3]]
        rows = rows.numpy()
        p2 = get_total_coords(p2, size, centers.numpy(), scales.numpy(), np.zeros(len(rows)))
        p3[..., :3] = (p3[..., :3] - 0.5) * 3000.0
        p2 = np.concatenate([p2[..., :3], flags2[rows].reshape(-1, N_JOINTS, 2)], axis=-1)
        p3 = np.concatenate([p3[..., :4], flags3[rows].reshape(-1, N_JOINTS, 2)], axis=-1)
        ori = np.zeros((len(rows), 2, 5), np.float32)
        ori[..., :2] = orientation
        ori[..., 2] = 1  # point predictions, not calibrated orientation confidence
        block = np.concatenate([p2.reshape(len(rows), -1), p3.reshape(len(rows), -1), ori.reshape(len(rows), -1)], axis=1)
        if not np.isfinite(block).all():
            raise ValueError(f"Non-finite predictions in {spec.title} {spec.split}; results not written")
        destination = rows if spec.dense_results else slice(offset, offset + len(rows))
        values[destination] = block
        offset += len(rows)
    result = pd.DataFrame(values, columns=columns)
    result.insert(0, "original_index", np.arange(len(df)) if spec.dense_results else dataset.rows)
    os.makedirs(os.path.dirname(spec.results), exist_ok=True)
    temporary = spec.results + ".tmp"
    result.to_pickle(temporary)
    os.replace(temporary, spec.results)
    manifest = {"version": 1, "canonical": True, "pose_weights_sha256": weights_hash,
                "dataframe_signature": dataframe_signature(spec.dataframe), "rows": len(df),
                "predicted_rows": len(dataset), "dataset": spec.name, "split": spec.split}
    with open(spec.results + ".json", "w") as stream:
        json.dump(manifest, stream, indent=2)
    logger.info("Wrote %d predictions -> %s", len(dataset), spec.results)
    return len(dataset)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--weights", default=None, help="Pose MTL checkpoint; default final p2d3d_c_o checkpoint")
    parser.add_argument("--datasets", nargs="+", default=None, help="h36m sim_rom sim_circle sim_c01 mpi_inf_3dhp fit3d aistpp")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke test: cap available images per dataset/split")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    configure_logger()
    args = parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    paths = get_experiment_paths(args.data_dir)
    specs = dataset_specs(paths)
    if args.datasets:
        known = {"h36m", "sim_rom", "sim_circle", "sim_c01", "mpi_inf_3dhp", "fit3d", "aistpp", "amass"}
        unknown = set(args.datasets) - known
        if unknown:
            raise ValueError(f"Unknown datasets: {sorted(unknown)}")
        specs = [s for s in specs if s.name in args.datasets]
    specs = [s for s in specs if args.split == "all" or s.split == args.split]
    for spec in specs:
        if not spec.has_images:
            logger.info("%s %s: no RGB images; GT/noise inputs only", spec.title, spec.split)
    specs = [s for s in specs if s.has_images]
    if not specs:
        raise FileNotFoundError("No image-backed sequence dataframes found for this selection")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("Run data generation on the GPU host, or explicitly select --cpu for a small smoke test")
    device = torch.device("cpu" if args.cpu else "cuda")
    weights = args.weights or paths.get_stage_checkpoint_path("p2d3d_c_o")
    model = load_pedrec_net(weights, device)
    digest = file_sha256(weights)
    for spec in specs:
        write_predictions(spec, model, device, digest, args.batch_size, args.num_workers, args.max_frames)


if __name__ == "__main__":
    main()
