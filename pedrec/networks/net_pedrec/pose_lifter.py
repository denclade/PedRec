"""
Temporal 3D pose lifting: refines the 3D pose of the current frame from the PedRecNet outputs of the last frames of
the same track (online / causal, no look-ahead, so it adds no latency).

Input per frame and joint (6 values, see ``lifter_features``):
    2D position relative to the person bb (normalized by the bb size), 2D joint confidence and the per frame 3D pose
    of PedRecNet (hip relative, mm / SKELETON_3D_RANGE).
Output: the refined 3D pose of the last frame (same units), predicted as a residual to the per frame 3D pose.

The network is a causal temporal convolution network in the style of VideoPose3D (Pavllo et al., CVPR 2019) with
dilations 1, 3, 9 (receptive field 27 frames, ~0.9 s at 30 fps). It has ~0.6M parameters and runs on all tracks of a
frame in one batch, i.e. well below a millisecond on a GPU.
"""
import numpy as np
import torch
import torch.nn as nn

SKELETON_3D_RANGE = 3000.0  # mm, PedRecNet predicts the 3D pose normalized to a cube of this size around the hip
NUM_FEATURES = 6
WINDOW = 27


def lifter_features(skeleton_2d: np.ndarray, skeleton_3d_mm: np.ndarray, bb: np.ndarray) -> np.ndarray:
    """
    :param skeleton_2d: ... x J x >=3 (x, y in image pixels, confidence)
    :param skeleton_3d_mm: ... x J x >=3 (x, y, z in mm relative to the hip)
    :param bb: ... x >=4 (center x, center y, width, height in pixels) of the person crop
    :return: ... x J x 6 float32
    """
    bb = np.asarray(bb, dtype=np.float32)
    center = bb[..., None, 0:2]
    size = np.maximum(np.maximum(bb[..., 2], bb[..., 3]), 1.0)[..., None, None]
    xy = (skeleton_2d[..., :2] - center) / size
    conf = skeleton_2d[..., 2:3]
    xyz = skeleton_3d_mm[..., :3] / SKELETON_3D_RANGE
    return np.concatenate((xy, conf, xyz), axis=-1).astype(np.float32)


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.dilation = dilation
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )

    def forward(self, x):
        # valid convolution: the output belongs to the last frames, the residual is cropped at the start (causal)
        return x[:, :, 2 * self.dilation:] + self.layers(x)


class TemporalPoseLifter(nn.Module):
    def __init__(self, num_joints: int, channels: int = 256, dilations=(3, 9), dropout: float = 0.1):
        super().__init__()
        self.num_joints = num_joints
        self.window = 3 + 2 * sum(dilations)
        self.expand = nn.Sequential(
            nn.Conv1d(num_joints * NUM_FEATURES, channels, kernel_size=3, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[_TemporalBlock(channels, d, dropout) for d in dilations])
        self.shrink = nn.Conv1d(channels, num_joints * 3, kernel_size=1)
        nn.init.zeros_(self.shrink.weight)  # start as identity (per frame 3D pose)
        nn.init.zeros_(self.shrink.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        :param features: B x T x J x 6 with T == self.window, the last frame is the current one
        :return: B x J x 3 refined 3D pose of the last frame (mm / SKELETON_3D_RANGE, hip relative)
        """
        batch, frames, joints, _ = features.shape
        assert frames == self.window, f"expected {self.window} frames, got {frames}"
        x = features.reshape(batch, frames, joints * NUM_FEATURES).transpose(1, 2)
        x = self.blocks(self.expand(x))
        delta = self.shrink(x)[:, :, -1].reshape(batch, joints, 3)
        return features[:, -1, :, 3:6] + delta
