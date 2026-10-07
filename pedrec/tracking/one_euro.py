"""
One Euro filter (Casiez, Roussel, Vogel, CHI 2012) for low latency smoothing of noisy per frame estimates.

The filter adapts its cutoff frequency to the speed of the signal: slow movements are smoothed strongly (less jitter),
fast movements only slightly (less lag). Used to smooth the 3D pose and the orientations of tracked humans.
"""
import math
from typing import Optional

import numpy as np


def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        """
        :param min_cutoff: minimum cutoff frequency in Hz (lower = smoother when the signal is slow)
        :param beta: speed coefficient (higher = less lag on fast movements); depends on the unit of the signal
        :param d_cutoff: cutoff frequency of the derivative in Hz
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = None

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None or self.x_prev.shape != x.shape or dt <= 0:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()
        dx = (x - self.x_prev) / dt
        a_d = _alpha(np.full_like(x, self.d_cutoff), dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat.copy()


class HumanStateSmoother:
    """
    Smooths the 3D skeleton (mm) and the body / head orientation (theta, phi in radians) of one tracked human.
    Angles are filtered on the unit circle (cos, sin), so the wrap around at 0 / 2*pi is handled correctly.
    """

    def __init__(self, skeleton_min_cutoff: float = 1.5, skeleton_beta: float = 0.005,
                 orientation_min_cutoff: float = 1.0, orientation_beta: float = 0.3):
        self.skeleton_filter = OneEuroFilter(skeleton_min_cutoff, skeleton_beta)
        self.orientation_filter = OneEuroFilter(orientation_min_cutoff, orientation_beta)

    def smooth_skeleton_3d(self, skeleton_3d: np.ndarray, dt: float) -> np.ndarray:
        result = skeleton_3d.copy()
        result[:, :3] = self.skeleton_filter(skeleton_3d[:, :3], dt).astype(skeleton_3d.dtype)
        return result

    def smooth_orientation(self, orientation: np.ndarray, dt: float) -> np.ndarray:
        """orientation: 2 x 2 (body / head) x (theta, phi) in radians"""
        unit_circle = np.stack((np.cos(orientation), np.sin(orientation)), axis=-1)
        smoothed = self.orientation_filter(unit_circle, dt)
        angles = np.arctan2(smoothed[..., 1], smoothed[..., 0])
        angles[..., 0] = np.clip(angles[..., 0], 0, np.pi)  # theta in [0, pi]
        angles[..., 1] = np.mod(angles[..., 1], 2 * np.pi)  # phi in [0, 2pi)
        return angles.astype(orientation.dtype)
