"""
Pose sequences for the temporal 3D lifter (``pedrec.networks.net_pedrec.pose_lifter``) from the PedRec dataframes
(Human3.6m, SIM-ROM, SIM-C01, ...). No images are needed.

Each sample is the window of the last ``window`` frames (same scene and camera) ending at a frame with a 3D ground
truth. The lifter input is either the PedRecNet prediction of these frames (result dataframe with the same index, e.g.
the ``*_allframes.pkl`` files of the SIM-C01 result writer) or the ground truth with noise that imitates the per frame
prediction errors.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.networks.net_pedrec.pose_lifter import lifter_features, SKELETON_3D_RANGE, WINDOW
from pedrec.utils.pandas_helper import read_pedrec_df
from pedrec.utils.pedrec_dataset_helper import get_filter_skeleton2d, get_filter_skeleton3d, get_filter_bb

logger = logging.getLogger(__name__)
NUM_JOINTS = len(SKELETON_PEDREC_JOINTS)
PREDICTION, NOISY_GT, CLEAN_GT = 0, 1, 2
SOURCE_NAMES = ("prediction", "noisy_gt", "clean_gt")


@dataclass
class SequenceNoiseConfig:
    """Noise added to ground truth inputs (training only), in the lifter feature units."""
    noise_2d: float = 0.01  # std of the 2D positions (fraction of the bb size)
    noise_3d: float = 0.012  # std of the per frame 3D positions (x 3000 mm = 36 mm)
    bias_3d: float = 0.01  # std of an offset per joint that is constant over the window (correlated errors)
    outlier_prob: float = 0.03  # probability per joint and frame of a large error with a low confidence
    outlier_scale: float = 0.08


class PoseSequenceDataset(Dataset):
    def __init__(self, df_path: str, results_df_path: Optional[str] = None, window: int = WINDOW, train: bool = True,
                 gt_result_ratio: float = 0.5, max_stride: int = 2, target_step: int = 1,
                 noise: SequenceNoiseConfig = None, seed: int = 0, convention: int = -1,
                 return_metadata: bool = False, source: Optional[int] = None):
        """
        :param results_df_path: PedRecNet predictions with the same index as ``df_path`` (rows of invalid frames 0)
        :param gt_result_ratio: training: probability to use the (noisy) ground truth instead of the predictions
        :param max_stride: training: random temporal stride in [1, max_stride] (robustness to the frame rate)
        :param target_step: use every n-th valid frame as target (smaller epochs for large datasets)
        """
        self.window = window
        self.train = train
        self.gt_result_ratio = gt_result_ratio if results_df_path is not None else 1.0
        self.max_stride = max_stride if train else 1
        self.noise = noise or SequenceNoiseConfig()
        self.seed = seed
        self.convention = convention
        self.return_metadata = return_metadata
        self.source = source
        self._rng = None
        self._rng_owner = None

        df = read_pedrec_df(df_path)
        self.skeleton_2d = df[get_filter_skeleton2d(df)].to_numpy(np.float32).reshape(len(df), NUM_JOINTS, 5)
        self.skeleton_3d = df[get_filter_skeleton3d(df)].to_numpy(np.float32).reshape(len(df), NUM_JOINTS, 6)
        self.bbs = df[get_filter_bb(df)].to_numpy(np.float32)
        visible_2d = self.skeleton_2d[:, :, 3].sum(axis=1)
        supported_3d = self.skeleton_3d[:, :, 5].sum(axis=1)
        self.valid_gt = (visible_2d >= 3) & (supported_3d >= NUM_JOINTS // 2) & (self.bbs[:, 2] > 1) & (self.bbs[:, 3] > 1)
        self.valid_gt &= np.isfinite(self.skeleton_3d[:, :, :3]).all(axis=(1, 2))
        self.valid_gt &= np.isfinite(self.bbs).all(axis=1) & (self.bbs[:, 4] >= 1)

        self.result_2d = self.result_3d = None
        self.valid_result = np.zeros(len(df), dtype=bool)
        self.result_positions = np.full(len(df), -1, dtype=np.int64)
        if results_df_path is not None:
            results = read_pedrec_df(results_df_path)
            if "original_index" in results:
                rows = results["original_index"].to_numpy()
                if not np.isfinite(rows).all() or not np.equal(rows, rows.astype(np.int64)).all():
                    raise ValueError(f"{results_df_path}: original_index must contain integer row positions")
                rows = rows.astype(np.int64)
                if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(df)):
                    raise ValueError(f"{results_df_path}: duplicate or out-of-range original_index")
            elif len(results) == len(df):
                rows = np.arange(len(df))
            else:
                raise ValueError(f"{results_df_path} has {len(results)} rows, {df_path} {len(df)}; the result "
                                 "dataframe needs explicit original_index for sparse predictions")
            self.result_2d = results[get_filter_skeleton2d(results)].to_numpy(np.float32).reshape(len(results), NUM_JOINTS, 5)
            self.result_3d = results[get_filter_skeleton3d(results)].to_numpy(np.float32).reshape(len(results), NUM_JOINTS, 6)
            self.result_positions[rows] = np.arange(len(results))
            finite = np.isfinite(self.result_2d).all(axis=(1, 2)) & np.isfinite(self.result_3d).all(axis=(1, 2))
            self.valid_result[rows] = finite & (np.abs(self.result_3d[:, :, :3]).sum(axis=(1, 2)) > 0)

        # sequences: rows of the same scene and camera, ordered by frame number
        order = np.lexsort((df["frame_nr_local"].to_numpy(), df["img_dir"].astype(str).to_numpy(),
                            df["scene_id"].to_numpy()))
        keys = (df["scene_id"].astype(str) + "|" + df["img_dir"].astype(str)).to_numpy()[order]
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        self.order = order  # position -> df row
        self.sequence_start = np.repeat(starts, np.diff(np.r_[starts, len(order)]))  # position -> first position
        targets = np.flatnonzero(self.valid_gt[order])
        self.targets = targets[::max(1, target_step)]
        if source == PREDICTION:
            self.targets = self.targets[self.valid_result[order[self.targets]]]
        logger.info(f"{df_path}: {len(starts)} sequences, {len(self.targets)} targets"
                    + (f", predictions for {self.valid_result.mean():.0%} of the frames" if results_df_path else ""))

    def __len__(self):
        return len(self.targets)

    @property
    def rng(self) -> np.random.Generator:
        # one generator per DataLoader worker process (and epoch), deterministic given the torch seed
        owner = torch.initial_seed()
        if self._rng is None or self._rng_owner != owner:
            self._rng = np.random.default_rng([self.seed, owner % (2 ** 32)])
            self._rng_owner = owner
        return self._rng

    def _window_rows(self, position: int, stride: int, valid: np.ndarray) -> np.ndarray:
        positions = position - stride * np.arange(self.window - 1, -1, -1)
        positions = np.maximum(positions, self.sequence_start[position])  # pad with the first frame of the sequence
        rows = self.order[positions]
        # Match causal inference: fill internal gaps with the PREVIOUS known pose.
        known = np.flatnonzero(valid[rows])
        if not len(known):
            raise ValueError("Sequence window contains no valid input")
        last = rows[known[0]]  # leading padding repeats the first available pose
        for i in range(len(rows)):
            if valid[rows[i]]:
                last = rows[i]
            else:
                rows[i] = last
        return rows

    def _noisy_gt(self, rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Ground truth with noise (also for validation, with a fixed seed per sample)."""
        skeleton_2d = self.skeleton_2d[rows].copy()
        skeleton_2d[:, :, 2] = skeleton_2d[:, :, 3] * rng.uniform(0.75, 1.0, size=skeleton_2d.shape[:2])
        features = lifter_features(skeleton_2d, self.skeleton_3d[rows], self.bbs[rows])
        n = self.noise
        frames, joints = features.shape[:2]
        features[:, :, 0:2] += rng.normal(0, n.noise_2d, size=(frames, joints, 2))
        features[:, :, 3:6] += rng.normal(0, n.noise_3d, size=(frames, joints, 3))
        features[:, :, 3:6] += rng.normal(0, n.bias_3d, size=(1, joints, 3))
        outliers = rng.random((frames, joints)) < n.outlier_prob
        features[outliers, 0:2] += rng.normal(0, n.outlier_scale, size=(outliers.sum(), 2))
        features[outliers, 3:6] += rng.normal(0, n.outlier_scale, size=(outliers.sum(), 3))
        features[outliers, 2] *= rng.uniform(0, 0.5, size=outliers.sum())
        return features.astype(np.float32)

    def get_sample(self, idx, source=None):
        position = self.targets[idx]
        target_row = self.order[position]
        rng = self.rng if self.train else np.random.default_rng([self.seed, int(idx)])
        stride = int(rng.integers(1, self.max_stride + 1)) if self.max_stride > 1 else 1
        if source is None:
            source = self.source
        if source is None:
            use_result = self.result_3d is not None and self.valid_result[target_row] and \
                (not self.train or rng.random() >= self.gt_result_ratio)
            source = PREDICTION if use_result else NOISY_GT
        if source == PREDICTION:
            if not self.valid_result[target_row]:
                raise ValueError("Prediction source requested for a target without predictions")
            rows = self._window_rows(position, stride, self.valid_result)
            positions = self.result_positions[rows]
            features = lifter_features(self.result_2d[positions], self.result_3d[positions], self.bbs[rows])
        else:
            rows = self._window_rows(position, stride, self.valid_gt)
            if source == CLEAN_GT:
                features = lifter_features(self.skeleton_2d[rows], self.skeleton_3d[rows], self.bbs[rows])
            elif source == NOISY_GT:
                features = self._noisy_gt(rows, rng)
            else:
                raise ValueError(f"Unknown lifter input source: {source}")
        target = self.skeleton_3d[target_row, :, :3] / SKELETON_3D_RANGE
        mask = self.skeleton_3d[target_row, :, 5].astype(np.float32)
        result = features, target.astype(np.float32), mask
        if self.return_metadata:
            # GT/noise are native-convention denoising/identity pairs; only real
            # canonical predictions need conversion for comparison with native GT.
            return (*result, {"source": source, "convention": self.convention if source == PREDICTION else -1})
        return result

    def __getitem__(self, idx):
        return self.get_sample(idx)


class LifterTrainingMixture(Dataset):
    """Source-first sampling so abundant AMASS/GT frames cannot drown real predictions.

    Within a source, sample datasets uniformly (then a target uniformly). Base
    sequence arrays are shared; no threefold copy of the large datasets or
    per-target sampler weights is required. Ratios are global probabilities.
    """

    def __init__(self, datasets, prediction_ratio=0.5, clean_gt_ratio=0.1, samples_per_epoch=None):
        self.datasets = datasets
        ratios = np.array([prediction_ratio, clean_gt_ratio])
        if not np.isfinite(ratios).all() or (ratios < 0).any() or ratios.sum() > 1 + 1e-12:
            raise ValueError("prediction_ratio and clean_gt_ratio must be nonnegative and sum to at most 1")
        self.probabilities = np.array([prediction_ratio, max(0, 1 - ratios.sum()), clean_gt_ratio])
        self.probabilities /= self.probabilities.sum()
        self.prediction_targets = [np.flatnonzero(d.valid_result[d.order[d.targets]]) for d in datasets]
        self.prediction_sets = [i for i, targets in enumerate(self.prediction_targets) if len(targets)]
        self.gt_sets = [i for i, dataset in enumerate(datasets) if len(dataset)]
        if not self.gt_sets:
            raise ValueError("No valid training targets")
        if prediction_ratio > 0 and not self.prediction_sets:
            raise ValueError("No real prediction targets; run mise run train:lifter:data first, or set --prediction-ratio 0")
        self.length = sum(len(d) for d in datasets) if samples_per_epoch is None else samples_per_epoch
        if self.length < 1:
            raise ValueError("samples_per_epoch must be positive")
        self._rng, self._owner = None, None

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        owner = torch.initial_seed()
        if self._rng is None or owner != self._owner:
            self._rng, self._owner = np.random.default_rng(owner), owner
        rng = self._rng
        source = int(rng.choice(3, p=self.probabilities))
        dataset_id = int(rng.choice(self.prediction_sets if source == PREDICTION else self.gt_sets))
        dataset = self.datasets[dataset_id]
        if source == PREDICTION:
            target = int(rng.choice(self.prediction_targets[dataset_id]))
        else:
            target = int(rng.integers(len(dataset)))
        return dataset.get_sample(target, source)
