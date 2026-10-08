import torch
import torch.nn as nn

from pedrec.networks.net_pedrec.pedrec_backbone import conv_bn_relu, init_conv_bn
from pedrec.utils.torch_utils.torch_modules import SoftArgmax2d, DepthRegression


def _init_heatmap_conv(conv: nn.Conv2d):
    nn.init.normal_(conv.weight, std=0.001)
    nn.init.zeros_(conv.bias)


class PedRecPose2DHead(nn.Module):
    """2D heatmaps (B x J x 64 x 48 logits) and their soft-argmax (x, y normalized to [0, 1], UDP)."""

    def __init__(self, channels: int, num_joints: int):
        super().__init__()
        self.refine = conv_bn_relu(channels, channels, 3)
        self.heatmap = nn.Conv2d(channels, num_joints, kernel_size=1)
        self.soft_argmax = SoftArgmax2d()

    def forward(self, x):
        pose_map = self.heatmap(self.refine(x))
        _, pose_coords = self.soft_argmax(pose_map)
        return pose_coords, pose_map

    def init_weights(self):
        init_conv_bn(self.refine)
        _init_heatmap_conv(self.heatmap)


class PedRecPose3DHead(nn.Module):
    """
    3D pose from an x / y heatmap (soft-argmax) and a per joint depth map weighted with the heatmap, normalized to the
    3D range around the hip (x, y, z in [0, 1]).
    """

    def __init__(self, channels: int, num_joints: int):
        super().__init__()
        self.refine = conv_bn_relu(channels, channels, 3)
        self.heatmap = nn.Conv2d(channels, num_joints, kernel_size=1)
        self.depth_map = nn.Conv2d(channels, num_joints, kernel_size=1)
        self.soft_argmax = SoftArgmax2d()
        self.depth = DepthRegression()

    def forward(self, x):
        x = self.refine(x)
        pose_map = self.heatmap(x)
        pose_map_softmax, pose_coords = self.soft_argmax(pose_map)
        depth = self.depth(self.depth_map(x), pose_map_softmax)
        pose_coords = torch.cat([pose_coords, depth], dim=2)
        pose_coords[:, :, 1] = 1 - pose_coords[:, :, 1]
        return pose_coords, pose_map

    def init_weights(self):
        init_conv_bn(self.refine)
        _init_heatmap_conv(self.heatmap)
        _init_heatmap_conv(self.depth_map)
