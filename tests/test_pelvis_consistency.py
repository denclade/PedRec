"""Geometry, masking, units and paired sample selection of the notebook diagnostics."""
import numpy as np
import torch
from torch.utils.data import Dataset

from pedrec.models.data_structures import ImageSize
from pedrec.tools.experiments.pelvis_consistency import (
    ROOT, LEFT, RIGHT, collect_samples, measure_batch, pelvis_delta,
    summarize_orientation, summarize_pelvis,
)


def _poses(count=1):
    p3 = np.full((count, 26, 3), 0.5)
    p3[:, LEFT, 0], p3[:, RIGHT, 0] = 0.46, 0.54
    p2 = np.full((count, 26, 2), 0.5)
    gt = np.concatenate([p3.copy(), np.ones((count, 26, 2))], axis=-1)
    return p2, p3, gt


def test_pelvis_metric_is_translation_invariant_and_uses_mm():
    p2, p3, gt = _poses()
    p3[:, ROOT, 1] += 0.1
    result = measure_batch(p2, p3, gt, ImageSize(192, 256), 3000)
    np.testing.assert_allclose(result.pelvis_3d_mm, [300])
    np.testing.assert_allclose(result.pelvis_dy_mm, [300])
    np.testing.assert_allclose(result.hip_relative_error_mm, [0])
    np.testing.assert_allclose(result.root_relative_error_mm, [300])
    shift = np.array([0.7, -0.4, 0.2])
    np.testing.assert_allclose(pelvis_delta(p3), pelvis_delta(p3 + shift), atol=1e-12)


def test_udp_pixels_and_camera_y_sign():
    p2, p3, gt = _poses()
    p2[:, ROOT] += [1 / 191, 2 / 255]
    result = measure_batch(p2, p3, gt, ImageSize(192, 256), 3000)
    np.testing.assert_allclose(result.pelvis_2d_px, [np.sqrt(5)])


def test_gt_and_centering_need_visible_supported_hips():
    p2, p3, gt = _poses(3)
    gt[0, LEFT, 4] = 0  # unsupported
    gt[1, RIGHT, 3] = 0  # invisible
    gt[2, :, 3:] = 0  # e.g. COCO has no 3D GT
    result = measure_batch(p2, p3, gt, ImageSize(192, 256), 3000)
    assert result.gt_pelvis_3d_mm.isna().all()
    assert result.hip_relative_error_mm.isna().all()
    assert np.isnan(result.root_relative_error_mm.iloc[2])
    assert result.valid_gt_joints.iloc[2] == 0
    np.testing.assert_allclose(result.pelvis_3d_mm, 0, atol=1e-9)


def test_nonfinite_prediction_is_reported_not_hidden():
    p2, p3, gt = _poses(2)
    p3[0, ROOT, 0] = np.nan
    result = measure_batch(p2, p3, gt, ImageSize(192, 256), 3000)
    result = result.assign(checkpoint="new", dataset="SIM")
    summary = summarize_pelvis(result).iloc[0]
    assert summary.n == 2 and summary.n_finite == 1
    assert abs(summary.median_mm) < 1e-9


def test_circular_phi_error_wraps_and_missing_labels_are_nan():
    p2, p3, gt = _poses(2)
    ori = np.zeros((2, 2, 2))
    target = np.zeros((2, 2, 5))
    ori[:, 0, 1] = [359 / 360, 0]
    target[:, 0, 1] = [1 / 360, 0.5]
    target[:, 0, 4] = 1
    result = measure_batch(p2, p3, gt, ImageSize(192, 256), 3000, ori, target)
    np.testing.assert_allclose(result.body_phi_error_deg, [2, 180])
    assert result.head_phi_error_deg.isna().all()
    summary = summarize_orientation(result.assign(checkpoint="new", dataset="COCO"))
    assert len(summary) == 1 and summary.iloc[0].part == "body"
    assert summary.iloc[0].over90_pct == 50


class _Samples(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        _, _, gt = _poses()
        return torch.full((3, 8, 6), float(index)), {
            "skeleton_3d": torch.tensor(gt[0]),
            "orientation": torch.zeros(2, 5), "img_path": f"frame_{index}.jpg",
        }


class _CanonicalModel(torch.nn.Module):
    def forward(self, images):
        p2, p3, _ = _poses(len(images))
        p3[:, ROOT, 1] += images[:, 0, 0, 0].numpy() / 3000
        return torch.tensor(p2), torch.tensor(p3), torch.zeros(len(images), 2, 2)


def test_collector_preserves_subset_order_and_worst_example_identity():
    frame, examples = collect_samples(_CanonicalModel(), _Samples(), [4, 1, 3], torch.device("cpu"),
                                      ImageSize(6, 8), 3000, batch_size=2, keep_examples=2)
    assert list(frame.sample_index) == [4, 1, 3]
    assert list(frame.image_path) == ["frame_4.jpg", "frame_1.jpg", "frame_3.jpg"]
    np.testing.assert_allclose(frame.pelvis_3d_mm, [4, 1, 3], atol=1e-3)  # collector uses fp32 predictions
    assert [e["sample_index"] for e in examples] == [4, 3]
    assert examples[0]["crop"][0, 0, 0] == 4
