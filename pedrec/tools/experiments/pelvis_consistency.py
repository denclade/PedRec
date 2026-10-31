"""Numerical diagnostics for notebooks/pelvis_consistency.ipynb.

Measure raw canonical predictions, never the dataset adapter output. Internal
pelvis consistency needs no GT: hip_center must equal the mean of the two hips
in the canonical SIM convention. A consistent skeleton can still be inaccurate.
"""
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT as J

ROOT, LEFT, RIGHT = J.hip_center.value, J.left_hip.value, J.right_hip.value


def pelvis_delta(pose: np.ndarray) -> np.ndarray:
    """Root minus hip mean, for B x J x D coordinates in any common units."""
    return pose[:, ROOT] - (pose[:, LEFT] + pose[:, RIGHT]) / 2


def valid_3d(target: np.ndarray) -> np.ndarray:
    """Same visible / supported flags as the training 3D evaluation."""
    return (target[..., 3] == 1) & (target[..., 4] == 1) & np.isfinite(target[..., :3]).all(axis=-1)


def masked_joint_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = mask.sum(axis=1)
    return np.divide(np.where(mask, values, 0).sum(axis=1), count,
                     out=np.full(len(values), np.nan), where=count > 0)


def measure_batch(pose_2d: np.ndarray, pose_3d: np.ndarray, target_3d: np.ndarray,
                  input_size, range_mm: float, orientation=None, target_orientation=None) -> pd.DataFrame:
    """Per image measurements; input poses / targets use the network normalization.

    MPJPE columns are per-image means over visible supported joints EXCLUDING
    hip_center. They are diagnostic, not the joint-macro training MPJPE.
    Hip-relative MPJPE centers prediction and GT on their OWN hip means; it is
    not a correction of the deployed prediction and does not fix other joints'
    convention differences. Never pool these errors as a benchmark result.
    """
    if range_mm <= 0:
        raise ValueError("range_mm must be positive")
    p3 = (np.asarray(pose_3d)[..., :3].astype(np.float64) - 0.5) * range_mm
    gt = (np.asarray(target_3d)[..., :3].astype(np.float64) - 0.5) * range_mm
    p2 = np.asarray(pose_2d)[..., :2] * np.array([input_size.width - 1, input_size.height - 1])
    delta = pelvis_delta(p3)
    delta2 = pelvis_delta(p2)
    gt_mask = valid_3d(target_3d)
    gt_pelvis_valid = gt_mask[:, [ROOT, LEFT, RIGHT]].all(axis=1)
    gt_delta = pelvis_delta(gt)
    gt_delta[~gt_pelvis_valid] = np.nan
    # Excluding the root also makes the centering diagnostics use the same joint mask.
    error_mask = gt_mask.copy() & np.isfinite(p3).all(axis=-1)
    error_mask[:, ROOT] = False
    hip_p, hip_gt = p3[:, [LEFT, RIGHT]].mean(axis=1), gt[:, [LEFT, RIGHT]].mean(axis=1)
    root_p, root_gt = p3[:, ROOT], gt[:, ROOT]
    root_error = np.linalg.norm((p3 - root_p[:, None]) - (gt - root_gt[:, None]), axis=-1)
    hip_error = np.linalg.norm((p3 - hip_p[:, None]) - (gt - hip_gt[:, None]), axis=-1)
    root_mean = masked_joint_mean(root_error, error_mask)
    root_mean[~gt_mask[:, ROOT]] = np.nan
    hip_mean = masked_joint_mean(hip_error, error_mask)
    hip_mean[~gt_mask[:, [LEFT, RIGHT]].all(axis=1)] = np.nan
    data = {
        "pelvis_3d_mm": np.linalg.norm(delta, axis=1),
        "pelvis_dx_mm": delta[:, 0], "pelvis_dy_mm": delta[:, 1], "pelvis_dz_mm": delta[:, 2],
        "pelvis_2d_px": np.linalg.norm(delta2, axis=1),
        "gt_pelvis_3d_mm": np.linalg.norm(gt_delta, axis=1),
        "root_relative_error_mm": root_mean, "hip_relative_error_mm": hip_mean,
        "valid_gt_joints": error_mask.sum(axis=1),
    }
    for i, part in enumerate(("body", "head")):
        error = np.full(len(p3), np.nan)
        angles = np.full(len(p3), np.nan)
        if orientation is not None and target_orientation is not None:
            pred = orientation[:, i, 1]
            target = target_orientation[:, i, 1]
            supported = (target_orientation[:, i, 4] == 1) & np.isfinite(pred) & np.isfinite(target)
            error[supported] = np.abs((pred[supported] - target[supported] + 0.5) % 1 - 0.5) * 360
            angles[supported] = target[supported] * 360
        data[f"{part}_phi_error_deg"] = error
        data[f"{part}_phi_gt_deg"] = angles
    return pd.DataFrame(data)


def collect_samples(model, dataset, indices: Sequence[int], device, input_size, range_mm: float,
                    batch_size: int = 32, num_workers: int = 0, keep_examples: int = 6):
    """Run a bare PedRecNet on an explicit ordered subset, keeping only worst examples.

    Pass wrapper.model, not the training wrapper. Old v2 checkpoints can use the
    current bare network: no adapter mask is applied and no loss is evaluated.
    """
    if not len(indices):
        raise ValueError("Empty sample selection")
    if keep_examples < 0:
        raise ValueError("keep_examples must be nonnegative")
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    model.eval()
    frames, examples, offset = [], [], 0
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Canonical pelvis", leave=False):
            outputs = model(images.to(device, non_blocking=True))
            p2, p3, ori = [o.detach().float().cpu().numpy() for o in outputs[:3]]
            gt = labels["skeleton_3d"].numpy()
            measured = measure_batch(p2, p3, gt, input_size, range_mm, ori, labels["orientation"].numpy())
            count = len(measured)
            measured["sample_index"] = list(indices[offset:offset + count])
            measured["image_path"] = list(labels.get("img_path", [""] * count))
            frames.append(measured)
            if keep_examples:
                # At most K examples from a batch can belong to the overall top K.
                values = measured.pelvis_3d_mm.to_numpy()
                finite = np.flatnonzero(np.isfinite(values))
                candidates = finite[np.argsort(values[finite])[-keep_examples:]]
                for i in candidates:
                    if not np.isfinite(measured.iloc[i].pelvis_3d_mm):
                        continue
                    examples.append({
                        "sample_index": int(indices[offset + i]),
                        "pelvis_3d_mm": float(measured.iloc[i].pelvis_3d_mm),
                        "image_path": measured.iloc[i].image_path,
                        "crop": images[i].numpy().copy(),
                        "pose_2d_px": p2[i, :, :2] * [input_size.width - 1, input_size.height - 1],
                        "pose_3d_mm": (p3[i, :, :3] - 0.5) * range_mm,
                        "target_3d_mm": (gt[i, :, :3] - 0.5) * range_mm,
                        "target_valid": valid_3d(gt[i:i + 1])[0],
                    })
                examples = sorted(examples, key=lambda e: e["pelvis_3d_mm"], reverse=True)[:keep_examples]
            offset += count
    return pd.concat(frames, ignore_index=True), examples


def summarize_pelvis(samples: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics per checkpoint and dataset; thresholds are diagnostic only."""
    records = []
    for (checkpoint, dataset), group in samples.groupby(["checkpoint", "dataset"], sort=False):
        values = group.pelvis_3d_mm.to_numpy()
        finite = values[np.isfinite(values)]
        record = {"checkpoint": checkpoint, "dataset": dataset, "n": len(group), "n_finite": len(finite)}
        for name, function in (("mean_mm", np.mean), ("median_mm", np.median),
                               ("p95_mm", lambda a: np.quantile(a, 0.95))):
            record[name] = function(finite) if len(finite) else np.nan
        for threshold in (10, 30, 100):
            record[f"over_{threshold}mm_pct"] = 100 * np.mean(finite > threshold) if len(finite) else np.nan
        for column in ("pelvis_dx_mm", "pelvis_dy_mm", "pelvis_dz_mm", "pelvis_2d_px", "gt_pelvis_3d_mm",
                       "root_relative_error_mm", "hip_relative_error_mm"):
            record[f"{column}_median"] = group[column].median()
        records.append(record)
    return pd.DataFrame(records)


def summarize_orientation(samples: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (checkpoint, dataset), group in samples.groupby(["checkpoint", "dataset"], sort=False):
        for part in ("body", "head"):
            errors = group[f"{part}_phi_error_deg"].dropna().to_numpy()
            if not len(errors):
                continue
            records.append({"checkpoint": checkpoint, "dataset": dataset, "part": part, "n": len(errors),
                            "mean_deg": errors.mean(), "median_deg": np.median(errors),
                            "p95_deg": np.quantile(errors, 0.95),
                            "acc22_5_pct": 100 * np.mean(errors <= 22.5),
                            "over90_pct": 100 * np.mean(errors > 90)})
    return pd.DataFrame(records)
