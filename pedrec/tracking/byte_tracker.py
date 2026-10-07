"""
Multi person tracker following ByteTrack (Zhang et al., ECCV 2022): a constant velocity Kalman filter per track and
two association stages with the Hungarian algorithm, first with the high score detections, then the remaining tracks
with the low score detections (which recovers partially occluded persons instead of dropping them).

Compared to the legacy tracker (optical flow on the joints + pose similarity merging) this needs no second PedRecNet
pass for undetected persons, keeps ids through short occlusions and does not create duplicate persons.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------------------------------------------------------- Kalman filter


class KalmanFilterXYAH:
    """
    Kalman filter on (center x, center y, aspect ratio w/h, height) and their velocities, as in SORT / DeepSORT /
    ByteTrack. Process and measurement noise scale with the bb height.
    """
    std_weight_position = 1.0 / 20
    std_weight_velocity = 1.0 / 160

    def __init__(self):
        ndim = 4
        self.motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self.motion_mat[i, ndim + i] = 1.0
        self.update_mat = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros_like(measurement)]
        h = measurement[3]
        std = [2 * self.std_weight_position * h, 2 * self.std_weight_position * h, 1e-2,
               2 * self.std_weight_position * h,
               10 * self.std_weight_velocity * h, 10 * self.std_weight_velocity * h, 1e-5,
               10 * self.std_weight_velocity * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self.std_weight_position * h, self.std_weight_position * h, 1e-2, self.std_weight_position * h,
               self.std_weight_velocity * h, self.std_weight_velocity * h, 1e-5, self.std_weight_velocity * h]
        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + np.diag(np.square(std))
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self.std_weight_position * h, self.std_weight_position * h, 1e-1, self.std_weight_position * h]
        projected_mean = self.update_mat @ mean
        projected_cov = self.update_mat @ covariance @ self.update_mat.T + np.diag(np.square(std))
        kalman_gain = np.linalg.solve(projected_cov, (covariance @ self.update_mat.T).T).T
        mean = mean + kalman_gain @ (measurement - projected_mean)
        covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return mean, covariance


# --------------------------------------------------------------------------------------------------- helpers


def center_bb_to_xyah(bb: np.ndarray) -> np.ndarray:
    """(center x, center y, width, height, ...) -> (center x, center y, w / h, h)"""
    width, height = float(bb[2]), max(float(bb[3]), 1e-6)
    return np.array([bb[0], bb[1], width / height, height], dtype=np.float64)


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    w = xyah[2] * xyah[3]
    h = xyah[3]
    return np.array([xyah[0] - w / 2, xyah[1] - h / 2, xyah[0] + w / 2, xyah[1] + h / 2])


def center_bbs_to_xyxy(bbs: np.ndarray) -> np.ndarray:
    bbs = np.asarray(bbs, dtype=np.float64)
    if len(bbs) == 0:
        return np.zeros((0, 4))
    return np.stack((bbs[:, 0] - bbs[:, 2] / 2, bbs[:, 1] - bbs[:, 3] / 2,
                     bbs[:, 0] + bbs[:, 2] / 2, bbs[:, 1] + bbs[:, 3] / 2), axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes a (N x 4) and b (M x 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def linear_assignment(cost: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    return matches, [r for r in range(cost.shape[0]) if r not in matched_rows], \
        [c for c in range(cost.shape[1]) if c not in matched_cols]


# --------------------------------------------------------------------------------------------------- tracker


@dataclass
class Track:
    uid: int
    mean: np.ndarray
    covariance: np.ndarray
    score: float
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    payload: object = None  # e.g. the Human of the last matched detection
    extra: Dict = field(default_factory=dict)

    @property
    def xyxy(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])


@dataclass
class TrackerResult:
    """detection index -> track uid for all detections which belong to a (confirmed or new) track"""
    assignments: Dict[int, int]
    lost_uids: List[int]
    removed_uids: List[int]


class ByteTracker:
    def __init__(self, high_thresh: float = 0.6, low_thresh: float = 0.1, new_track_thresh: float = 0.7,
                 match_thresh: float = 0.8, low_match_thresh: float = 0.5, max_time_lost: int = 30):
        """
        :param high_thresh: detections with a score >= high_thresh are used in the first association stage
        :param low_thresh: detections between low_thresh and high_thresh are only used to continue existing tracks
        :param new_track_thresh: minimum score to start a new track from an unmatched detection
        :param match_thresh: maximum cost (1 - IoU) for a match in the first stage
        :param low_match_thresh: maximum cost (1 - IoU) for a match in the second stage
        :param max_time_lost: frames a lost track is kept (can be re-identified) before it is removed
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.low_match_thresh = low_match_thresh
        self.max_time_lost = max_time_lost
        self.kalman = KalmanFilterXYAH()
        self.tracks: List[Track] = []  # active (updated in the last frame)
        self.lost_tracks: List[Track] = []
        self.next_uid = 0

    def reset(self):
        self.tracks = []
        self.lost_tracks = []
        self.next_uid = 0

    def _predict(self, tracks: List[Track]):
        for track in tracks:
            if track.time_since_update > 0:
                track.mean[7] = 0  # height velocity of lost tracks is not reliable
            track.mean, track.covariance = self.kalman.predict(track.mean, track.covariance)
            track.age += 1
            track.time_since_update += 1

    def _update(self, track: Track, bb: np.ndarray, score: float, payload):
        track.mean, track.covariance = self.kalman.update(track.mean, track.covariance, center_bb_to_xyah(bb))
        track.score = score
        track.hits += 1
        track.time_since_update = 0
        track.payload = payload

    def update(self, bbs: np.ndarray, scores: np.ndarray, payloads: Optional[List] = None) -> TrackerResult:
        """
        :param bbs: N x >=4 center bbs (center x, center y, width, height) of the current frame
        :param scores: N detection scores
        :param payloads: optional per detection objects stored with the track
        """
        bbs = np.asarray(bbs, dtype=np.float64).reshape(len(scores), -1) if len(scores) else np.zeros((0, 4))
        scores = np.asarray(scores, dtype=np.float64)
        payloads = payloads if payloads is not None else [None] * len(scores)

        pool = self.tracks + self.lost_tracks
        self._predict(pool)

        high = [i for i, s in enumerate(scores) if s >= self.high_thresh]
        low = [i for i, s in enumerate(scores) if self.low_thresh <= s < self.high_thresh]
        det_xyxy = center_bbs_to_xyxy(bbs) if len(bbs) else np.zeros((0, 4))
        assignments: Dict[int, int] = {}

        # stage 1: all tracks (active + lost) <-> high score detections, IoU fused with the detection score
        track_xyxy = np.array([t.xyxy for t in pool]).reshape(-1, 4)
        iou = iou_matrix(track_xyxy, det_xyxy[high])
        cost = 1 - iou * scores[high][None, :] if len(high) else np.zeros((len(pool), 0))
        matches, unmatched_tracks, unmatched_high = linear_assignment(cost, self.match_thresh)
        for t, d in matches:
            det_idx = high[d]
            self._update(pool[t], bbs[det_idx], scores[det_idx], payloads[det_idx])
            assignments[det_idx] = pool[t].uid

        # stage 2: remaining recently active tracks <-> low score detections
        remaining = [pool[i] for i in unmatched_tracks if pool[i].time_since_update == 1]
        iou = iou_matrix(np.array([t.xyxy for t in remaining]).reshape(-1, 4), det_xyxy[low])
        matches, _, _ = linear_assignment(1 - iou, self.low_match_thresh)
        for t, d in matches:
            det_idx = low[d]
            self._update(remaining[t], bbs[det_idx], scores[det_idx], payloads[det_idx])
            assignments[det_idx] = remaining[t].uid

        # new tracks from unmatched high score detections
        for d in unmatched_high:
            det_idx = high[d]
            if scores[det_idx] < self.new_track_thresh:
                continue
            mean, covariance = self.kalman.initiate(center_bb_to_xyah(bbs[det_idx]))
            track = Track(uid=self.next_uid, mean=mean, covariance=covariance, score=scores[det_idx],
                          payload=payloads[det_idx])
            self.next_uid += 1
            pool.append(track)
            assignments[det_idx] = track.uid

        # bookkeeping
        self.tracks = [t for t in pool if t.time_since_update == 0]
        lost = [t for t in pool if t.time_since_update > 0]
        removed = [t.uid for t in lost if t.time_since_update > self.max_time_lost]
        self.lost_tracks = [t for t in lost if t.time_since_update <= self.max_time_lost]
        return TrackerResult(assignments=assignments, lost_uids=[t.uid for t in self.lost_tracks],
                             removed_uids=removed)
