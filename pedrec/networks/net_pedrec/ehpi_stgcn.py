"""
Action recognition on the EHPI skeleton sequences with a spatial temporal graph convolutional network (ST-GCN, Yan et
al., AAAI 2018) using the PedRec skeleton graph, an adaptive (learned) adjacency as in 2s-AGCN (Shi et al., CVPR 2019)
and the three ST-GCN partitions (self, towards the hip, away from the hip).

The network takes the same EHPI input as before (B x 3 x 32 rows x T frames, normalized with the EHPI mean / std, rows
in EHPI joint order, 26 joints + padding rows), so the EHPI datasets and dataframes are used unchanged. Compared to
the previous ResNet-50 on the EHPI image it uses the skeleton topology explicitly and has ~2M instead of ~23.5M
parameters.
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC, SKELETON_PEDREC_JOINT, \
    SKELETON_PEDREC_TO_PEDRECEHPI3D

NUM_JOINTS = len(SKELETON_PEDREC_JOINT)


def get_ehpi_partitioned_adjacency() -> np.ndarray:
    """3 x V x V normalized adjacency (self, centripetal, centrifugal) in EHPI joint order."""
    edges = [(SKELETON_PEDREC_TO_PEDRECEHPI3D[a], SKELETON_PEDREC_TO_PEDRECEHPI3D[b]) for a, b in SKELETON_PEDREC]
    center = SKELETON_PEDREC_TO_PEDRECEHPI3D[SKELETON_PEDREC_JOINT.hip_center.value]
    neighbours = {v: [] for v in range(NUM_JOINTS)}
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    hops = {center: 0}
    queue = deque([center])
    while queue:
        v = queue.popleft()
        for n in neighbours[v]:
            if n not in hops:
                hops[n] = hops[v] + 1
                queue.append(n)
    adjacency = np.zeros((3, NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    adjacency[0] = np.eye(NUM_JOINTS)
    for a, b in edges:
        for i, j in ((a, b), (b, a)):
            # message from j to i: towards the center (centripetal) or away from it (centrifugal)
            adjacency[1 if hops[j] > hops[i] else 2, i, j] = 1
    # normalize by the in-degree
    for k in range(3):
        degree = adjacency[k].sum(axis=1, keepdims=True)
        adjacency[k] = np.divide(adjacency[k], degree, out=np.zeros_like(adjacency[k]), where=degree > 0)
    return adjacency


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        partitions = adjacency.shape[0]
        self.adaptive = nn.Parameter(torch.zeros_like(adjacency))  # learned residual adjacency
        self.conv = nn.Conv2d(in_channels, out_channels * partitions, kernel_size=1)
        self.partitions = partitions
        self.out_channels = out_channels

    def forward(self, x):  # x: N x C x T x V
        n, _, t, v = x.shape
        x = self.conv(x).view(n, self.partitions, self.out_channels, t, v)
        return torch.einsum("nkctv,kwv->nctw", x, self.adjacency + self.adaptive)


class StGcnBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor, stride: int = 1,
                 dropout: float = 0.1, residual: bool = True):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, adjacency)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.tcn(self.gcn(x))
        if self.residual is not None:
            out = out + self.residual(x)
        return self.relu(out)


class EhpiStGcn(nn.Module):
    def __init__(self, num_actions: int, channels=(64, 64, 64, 128, 128, 256, 256), strides=(1, 1, 1, 2, 1, 2, 1),
                 dropout: float = 0.1):
        super().__init__()
        adjacency = torch.from_numpy(get_ehpi_partitioned_adjacency())
        self.data_bn = nn.BatchNorm1d(3 * NUM_JOINTS)
        blocks = []
        in_channels = 3
        for i, (out_channels, stride) in enumerate(zip(channels, strides)):
            blocks.append(StGcnBlock(in_channels, out_channels, adjacency, stride, dropout, residual=i > 0))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(in_channels, num_actions)

    def forward(self, ehpi: torch.Tensor) -> torch.Tensor:
        """ehpi: B x 3 x rows (>= 26, EHPI joint order) x T -> B x num_actions logits"""
        x = ehpi[:, :, :NUM_JOINTS, :].permute(0, 1, 3, 2).contiguous()  # B x C x T x V
        n, c, t, v = x.shape
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t)).view(n, c, v, t).permute(0, 1, 3, 2)
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        return self.fc(self.dropout(x))
