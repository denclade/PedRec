import math

import torch
import torch.nn as nn

from pedrec.configs.pedrec_net_config import PedRecNetConfig


class PedRecOrientationsHead(nn.Module):
    """
    Body and head orientation from the pooled stride 32 backbone features and the (detached) 3D pose.

    Each orientation is regressed as (cos, sin) of theta and phi (biternion, continuous at 0 / 360 degrees). Returns
    B x 2 (body, head) x 2 (theta / pi, phi / 2pi) and the raw theta / phi vectors (B x 2 x 2) for the loss.
    """

    def __init__(self, cfg: PedRecNetConfig, in_channels: int, pose_size: int = 256):
        super(PedRecOrientationsHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pose_mlp = nn.Sequential(nn.Linear(cfg.model.num_joints * 3, pose_size), nn.ReLU(inplace=True),
                                      nn.Linear(pose_size, pose_size), nn.ReLU(inplace=True))
        self.body_orientation = BiternionOrientationHead(in_channels + pose_size)
        self.head_orientation = BiternionOrientationHead(in_channels + pose_size)

    def forward(self, x, x_pose):
        x = torch.flatten(self.avgpool(x), 1)
        pose_features = self.pose_mlp(torch.flatten(x_pose.detach()[:, :, :3], 1))
        x = torch.cat([x, pose_features], dim=1)
        body, body_theta, body_phi = self.body_orientation(x)
        head, head_theta, head_phi = self.head_orientation(x)
        orientation = torch.stack((body, head), dim=1)
        theta_vectors = torch.stack((body_theta, head_theta), dim=1)
        phi_vectors = torch.stack((body_phi, head_phi), dim=1)
        return orientation, theta_vectors, phi_vectors

    def init_weights(self):
        for m in self.pose_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        self.body_orientation.init_weights()
        self.head_orientation.init_weights()


class BiternionOrientationHead(nn.Module):
    """
    Regresses (cos, sin) of theta and phi (Beyer et al., "Biternion Nets", GCPR 2015). Unlike the expected value over
    angle bins this has no discontinuity at 0 / 360 degrees.
    """

    def __init__(self, in_features: int):
        super(BiternionOrientationHead, self).__init__()
        self.vectors = nn.Linear(in_features, 4)  # theta (cos, sin), phi (cos, sin)

    def forward(self, x):
        vectors = self.vectors(x.float())
        theta_vec, phi_vec = vectors[:, 0:2], vectors[:, 2:4]
        theta = torch.atan2(theta_vec[:, 1], theta_vec[:, 0]).clamp(0, math.pi) / math.pi
        phi = torch.remainder(torch.atan2(phi_vec[:, 1], phi_vec[:, 0]), 2 * math.pi) / (2 * math.pi)
        orientation = torch.stack((theta, phi), dim=1)
        return orientation, theta_vec, phi_vec

    def init_weights(self):
        nn.init.xavier_normal_(self.vectors.weight.data, gain=0.1)
        # start with theta = 90 degrees (upright) and phi = 0
        with torch.no_grad():
            self.vectors.bias.copy_(torch.tensor([0.0, 1.0, 1.0, 0.0]))
