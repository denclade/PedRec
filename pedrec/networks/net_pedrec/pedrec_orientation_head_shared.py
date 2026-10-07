import math

import torch
import torch.nn as nn

from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.networks.net_pedrec.pedrec_net_helper import default_init
from pedrec.utils.torch_utils.torch_modules import SoftArgmax1d


class PedRecOrientationsHead(nn.Module):
    """
    Body and head orientation from the pooled backbone features and the (detached) 3D pose.

    ``cfg.arch.orientation_head``: "softargmax" (original, expected value over 360 phi / 180 theta bins) or
    "biternion" (unit vector regression, continuous at 0 / 360 degrees). The output is in both cases
    B x 2 (body, head) x 2 (theta / pi, phi / 2pi); the 2nd and 3rd return values are the bin logits (softargmax) or
    the raw (cos, sin) vectors (biternion) for the loss.
    """

    def __init__(self, cfg: PedRecNetConfig, in_channels: int = None):
        super(PedRecOrientationsHead, self).__init__()
        arch = cfg.arch
        in_channels = in_channels or 512 * cfg.layer.block.expansion
        num_joints = cfg.model.num_joints
        self.deconv_with_bias = False
        self.pose_features = arch.orientation_pose_features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.pose_features == "mlp":
            pose_size = 256
            self.pose_mlp = nn.Sequential(nn.Linear(num_joints * 3, pose_size), nn.ReLU(inplace=True),
                                          nn.Linear(pose_size, pose_size), nn.ReLU(inplace=True))
        else:
            pose_size = (num_joints + 6) * 24  # 3 ConvTranspose1d (k=3) layers: length + 6, 24 channels
            self.pose_deconv = self._make_deconv_layer()
        head_cls = BiternionOrientationHead if arch.orientation_head == "biternion" else PedRecOrientationHead
        self.body_orientation = head_cls(in_channels + pose_size)
        self.head_orientation = head_cls(in_channels + pose_size)

    def forward(self, x, x_pose):
        # batch_size, 2048, 8, 6
        x = self.avgpool(x)
        # batch_size, 2048, 1, 1
        x = torch.flatten(x, 1)
        x_pose = x_pose.detach()
        x_pose = x_pose[:, :, :3]  # select x, y and z
        if self.pose_features == "mlp":
            pose_features = self.pose_mlp(torch.flatten(x_pose, 1))
        else:
            # concatenate feature map with 3D pose coordinates as additional features
            pose_features = self.pose_deconv(torch.transpose(x_pose, 1, 2))
        # x_pose_flat =   # detach pose coords to stop gradient flow because this is a label
        # pose_features = self.avgpool(pose_features)
        pose_features = torch.flatten(pose_features, 1)
        x = torch.cat([x, pose_features], dim=1)

        body_orientation, body_theta_map, body_phi_map = self.body_orientation(x)
        body_orientation = torch.unsqueeze(body_orientation, dim=1)
        body_theta_map = torch.unsqueeze(body_theta_map, dim=1)
        body_phi_map = torch.unsqueeze(body_phi_map, dim=1)
        head_orientation, head_theta_map, head_phi_map = self.head_orientation(x)
        head_orientation = torch.unsqueeze(head_orientation, dim=1)
        head_theta_map = torch.unsqueeze(head_theta_map, dim=1)
        head_phi_map = torch.unsqueeze(head_phi_map, dim=1)
        orientation = torch.cat((body_orientation, head_orientation), dim=1)

        theta_maps = torch.cat((body_theta_map, head_theta_map), dim=1)
        phi_maps = torch.cat((body_phi_map, head_phi_map), dim=1)
        return orientation, theta_maps, phi_maps

    def _make_deconv_layer(self):
        layers = []

        layers.append(nn.ConvTranspose1d(
            in_channels=3,
            out_channels=6,
            kernel_size=3
        ))
        layers.append(nn.BatchNorm1d(6))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.ConvTranspose1d(
            in_channels=6,
            out_channels=12,
            kernel_size=3
        ))
        layers.append(nn.BatchNorm1d(12))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.ConvTranspose1d(
            in_channels=12,
            out_channels=24,
            kernel_size=3
        ))
        layers.append(nn.BatchNorm1d(24))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def init_weights(self):
        if self.pose_features == "mlp":
            for m in self.pose_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)
        else:
            for name, m in self.pose_deconv.named_modules():
                default_init(m, name, self.deconv_with_bias)
        self.body_orientation.init_weights()
        self.head_orientation.init_weights()


class PedRecOrientationHead(nn.Module):
    def __init__(self, in_features: int):
        super(PedRecOrientationHead, self).__init__()
        self.phi = nn.Linear(in_features, 360)
        self.theta = nn.Linear(in_features, 180)

        self.body_orientation = SoftArgmax1d()

    def forward(self, x):
        theta_map = self.theta(x)
        theta_softmax, theta = self.body_orientation(theta_map)

        phi_map = self.phi(x)
        phi_softmax, phi = self.body_orientation(phi_map)

        orientation = torch.cat([theta, phi], dim=1)
        return orientation, theta_map, phi_map

    def init_weights(self):
        nn.init.xavier_normal_(self.phi.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.phi.bias.data, 0)

        nn.init.xavier_normal_(self.theta.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.theta.bias.data, 0)



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
