import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pedrec.configs.pedrec_net_config import PedRecNetConfig


def heatmap_statistics(heatmap_logits: torch.Tensor) -> torch.Tensor:
    """
    Per joint statistics of the (softmax normalized) heatmaps: peak probability, normalized entropy and spatial
    spread (std of the coordinates in [0, 1]). B x J x H x W -> B x J x 3
    """
    batch, joints, height, width = heatmap_logits.shape
    prob = F.softmax(heatmap_logits.float().reshape(batch, joints, -1), dim=-1)
    peak = prob.max(dim=-1).values
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum(dim=-1) / math.log(height * width)
    ys = torch.linspace(0, 1, height, device=prob.device).repeat_interleave(width)
    xs = torch.linspace(0, 1, width, device=prob.device).repeat(height)
    mean_x = (prob * xs).sum(-1, keepdim=True)
    mean_y = (prob * ys).sum(-1, keepdim=True)
    spread = torch.sqrt((prob * ((xs - mean_x) ** 2 + (ys - mean_y) ** 2)).sum(-1) + 1e-12)
    return torch.stack((peak, entropy, spread), dim=-1)


class PedRecPoseConfHeadHeatmap(nn.Module):
    """
    Joint confidence ("is this joint visible / correctly localized") from statistics of the 2D and 3D heatmaps with a
    small MLP shared by all joints plus a learned joint embedding.

    Compared to the original head (two convolutions + a fully connected layer over 42240 features) it has ~3k instead
    of ~5.5M parameters and does not depend on the input resolution. Returns probabilities and logits (the loss uses
    the logits, BCEWithLogitsLoss).
    """

    def __init__(self, cfg: PedRecNetConfig, embedding_size: int = 8, hidden: int = 32):
        super().__init__()
        self.joint_embedding = nn.Embedding(cfg.model.num_joints, embedding_size)
        self.mlp = nn.Sequential(nn.Linear(6 + embedding_size, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, 1))

    def forward(self, pose_map_2d: torch.Tensor, pose_map_3d: torch.Tensor):
        features = torch.cat((heatmap_statistics(pose_map_2d.detach()), heatmap_statistics(pose_map_3d.detach())),
                             dim=-1)
        batch, joints, _ = features.shape
        embedding = self.joint_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        logits = self.mlp(torch.cat((features, embedding), dim=-1)).squeeze(-1)
        return torch.sigmoid(logits), logits

    def init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.joint_embedding.weight, std=0.1)
