"""Real-image export, sparse alignment, source mixture and frozen-convention loss."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from pedrec.datasets.lifter_dataset_specs import LifterDatasetSpec, dataset_specs
from pedrec.datasets.pose_sequence_dataset import (PoseSequenceDataset, LifterTrainingMixture,
                                                  PREDICTION, NOISY_GT, CLEAN_GT)
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT as J
from pedrec.networks.net_pedrec.pedrec_convention_adapter import ConventionAdapter
from pedrec.networks.net_pedrec.pose_lifter import lifter_features
from pedrec.tools.resultwriters.lifter_to_results import write_predictions
from pedrec.training.train_lifter import adapt_for_loss, check_prediction_manifest, train_lifter
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.train_stepper import TrainingOptions
from pedrec.utils.pandas_helper import read_pedrec_df
from tests.test_lifter import GT, RESULTS


class _Pose(torch.nn.Module):
    def forward(self, images):
        p2 = torch.full((len(images), 26, 3), 0.5, device=images.device)
        p3 = torch.full((len(images), 26, 4), 0.5, device=images.device)
        p2[..., 2] = 1
        p3[..., 0], p3[..., 3] = 0.52, 1
        return p2, p3, torch.zeros(len(images), 2, 2, device=images.device)


@pytest.mark.parametrize("dense", [False, True])
def test_export_existing_images_aligned_to_sequence_rows(tmp_path, dense):
    df = read_pedrec_df(GT)
    df.index += 100  # alignment is positional, not pandas labels
    source = tmp_path / "sequence.pkl"
    df.to_pickle(source)
    for position in (0, 2, 5):
        row = df.iloc[position]
        image = tmp_path / row.img_dir / f"img_{int(row.img_id):05d}.{row.img_type}"
        image.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image), np.full((48, 32, 3), 100, np.uint8))
    spec = LifterDatasetSpec("fixture", "fixture", "train", str(source), str(tmp_path / "pred.pkl"),
                             dense_results=dense)
    assert write_predictions(spec, _Pose(), torch.device("cpu"), "hash", batch_size=2, num_workers=0) == 3
    result = read_pedrec_df(spec.results)
    assert len(result) == (len(df) if dense else 3)
    assert list(result.original_index) == (list(range(len(df))) if dense else [0, 2, 5])
    assert result.original_index.dtype.kind == "i"
    np.testing.assert_allclose(result.set_index("original_index").loc[[0, 2, 5], "skeleton3d_nose_x"], 60, atol=1e-3)
    # x=y=.5 in the crop must invert to the annotated crop center.
    np.testing.assert_allclose(result.set_index("original_index").loc[[0, 2, 5], "skeleton2d_nose_x"],
                               df.iloc[[0, 2, 5]].bb_center_x, atol=1e-3)
    dataset = PoseSequenceDataset(str(source), spec.results, train=False, source=PREDICTION)
    assert len(dataset) == 3
    assert dataset.valid_result.sum() == 3
    sample, _, _ = dataset[1]
    np.testing.assert_allclose(sample[-1, :, 3], 0.02, atol=1e-6)
    check_prediction_manifest(spec, "hash")
    with pytest.raises(ValueError, match="different pose weights"):
        check_prediction_manifest(spec, "changed")
    metadata = json.loads((tmp_path / "pred.pkl.json").read_text())
    assert metadata["predicted_rows"] == 3 and metadata["canonical"]


def test_source_modes_keep_clean_gt_unchanged_and_map_only_predictions():
    dataset = PoseSequenceDataset(GT, RESULTS, train=False, return_metadata=True, convention=2)
    idx = len(dataset) - 1
    clean, target, _, clean_meta = dataset.get_sample(idx, CLEAN_GT)
    np.testing.assert_allclose(clean[-1, :, 3:6], target)
    position = dataset.targets[idx]
    rows = dataset._window_rows(position, 1, dataset.valid_gt)
    np.testing.assert_allclose(clean, lifter_features(dataset.skeleton_2d[rows], dataset.skeleton_3d[rows], dataset.bbs[rows]))
    noisy, _, _, noisy_meta = dataset.get_sample(idx, NOISY_GT)
    assert not np.array_equal(clean, noisy)
    _, _, _, prediction_meta = dataset.get_sample(idx, PREDICTION)
    assert clean_meta["convention"] == noisy_meta["convention"] == -1
    assert prediction_meta["convention"] == 2


def test_sparse_predictions_reject_duplicate_or_missing_row_ids(tmp_path):
    results = read_pedrec_df(RESULTS).iloc[[3, 1]].copy()
    results["original_index"] = [3, 1]
    path = tmp_path / "sparse.pkl"
    results.to_pickle(path)
    dataset = PoseSequenceDataset(GT, str(path), train=False, source=PREDICTION)
    assert list(dataset.order[dataset.targets]) == [1, 3]
    assert dataset.result_positions[1] == 1 and dataset.result_positions[3] == 0
    results["original_index"] = [1, 1]
    results.to_pickle(path)
    with pytest.raises(ValueError, match="duplicate"):
        PoseSequenceDataset(GT, str(path))


def test_missing_frames_repeat_previous_pose_without_lookahead():
    dataset = PoseSequenceDataset(GT, train=False)
    valid = np.zeros(len(dataset.order), bool)
    valid[[0, 2, 7]] = True
    rows = dataset._window_rows(7, 1, valid)
    assert list(rows[-8:]) == [0, 0, 2, 2, 2, 2, 2, 7]


def test_frozen_convention_loss_keeps_gradient_to_lifter():
    adapter = ConventionAdapter().requires_grad_(False)
    with torch.no_grad():
        adapter.offsets[2, J.hip_center.value, J.spine_center.value] = 0.1
    pred = torch.randn(2, 26, 3, requires_grad=True)
    mapped = adapt_for_loss(pred, torch.tensor([2, -1]), adapter)
    assert not torch.allclose(mapped[0], pred[0])
    assert torch.equal(mapped[1], pred[1])
    mapped.square().sum().backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    assert adapter.offsets.grad is None
    with pytest.raises(ValueError, match="matching frozen"):
        adapt_for_loss(pred, torch.tensor([2, -1]), None)


class _Pool:
    def __init__(self, length, predicted, name):
        self.order = self.targets = np.arange(length)
        self.valid_result = np.full(length, predicted)
        self.name = name
    def __len__(self):
        return len(self.targets)
    def get_sample(self, target, source):
        return self.name, source


def test_global_source_mixture_is_not_diluted_by_large_gt_datasets():
    torch.manual_seed(1234)
    mixture = LifterTrainingMixture([_Pool(2, True, "images"), _Pool(10000, False, "amass")],
                                    prediction_ratio=0.5, clean_gt_ratio=0.1, samples_per_epoch=1000)
    samples = [mixture[i] for i in range(len(mixture))]
    counts = np.bincount([s[1] for s in samples], minlength=3) / len(samples)
    np.testing.assert_allclose(counts, [0.5, 0.4, 0.1], atol=0.05)
    assert all(name == "images" for name, source in samples if source == PREDICTION)
    gt = [name for name, source in samples if source != PREDICTION]
    assert 0.4 < gt.count("amass") / len(gt) < 0.6
    with pytest.raises(ValueError, match="No real prediction"):
        LifterTrainingMixture([_Pool(20, False, "amass")])


def test_specs_share_prediction_paths_and_skip_amass_images(tmp_path):
    paths = get_experiment_paths(str(tmp_path))
    for directory, filename in [(paths.h36m_train_dir, paths.h36m_train_filename),
                                 (paths.sim_c01_val_dir, paths.sim_c01_val_filename),
                                 (str(tmp_path / "datasets/AMASS"), "amass_train_seq.pkl")]:
        path = Path(directory) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    specs = {s.name: s for s in dataset_specs(paths)}
    assert specs["h36m"].convention == 2 and specs["h36m"].results.endswith("_predictions.pkl")
    assert specs["sim_c01"].dense_results and specs["sim_c01"].results.endswith(paths.sim_c01_val_results_filename)
    assert not specs["amass"].has_images


def test_empty_validation_is_skipped_and_cannot_select_zero_error(tmp_path):
    df = read_pedrec_df(GT)
    for c in df.columns:
        if c.startswith("skeleton3d") and c.endswith("_supported"):
            df[c] = 0
    empty = tmp_path / "no_targets.pkl"
    df.to_pickle(empty)
    output = str(tmp_path / "lifter.pth")
    result = train_lifter([("fixture", GT, RESULTS)],
                          [("empty", str(empty), None), ("fixture", GT, RESULTS)],
                          torch.device("cpu"), output, epochs=1, batch_size=4, num_workers=0,
                          samples_per_epoch=8, options=TrainingOptions(ema_decay=0))
    assert "empty" not in result and "fixture" in result
    metadata = json.loads((tmp_path / "lifter.pth.json").read_text())
    assert metadata["selection_sets"] == ["fixture"]


def test_training_with_sparse_native_gt_and_frozen_adapter(tmp_path):
    result = read_pedrec_df(RESULTS).iloc[[0, 2, 5]].copy()
    result["original_index"] = [0, 2, 5]
    path = tmp_path / "sparse.pkl"
    result.to_pickle(path)
    spec = LifterDatasetSpec("h36m", "H36M", "train", GT, str(path), convention=2)
    adapter = ConventionAdapter().requires_grad_(False)
    with torch.no_grad():
        adapter.offsets[2, J.hip_center.value, J.spine_center.value] = 0.1
    before = adapter.offsets.clone()
    output = str(tmp_path / "lifter.pth")
    metrics = train_lifter([spec], [spec], torch.device("cpu"), output, epochs=1, batch_size=4,
                           num_workers=0, samples_per_epoch=12, pose_adapter=adapter,
                           options=TrainingOptions(ema_decay=0))
    assert np.isfinite(metrics["H36M"]["lifted_mm"])
    assert adapter.offsets.grad is None and torch.equal(before, adapter.offsets)
    metadata = json.loads((tmp_path / "lifter.pth.json").read_text())
    assert metadata["frozen_adapter_for_prediction_loss"]
    assert sum(metadata["input_counts"].values()) == 12
