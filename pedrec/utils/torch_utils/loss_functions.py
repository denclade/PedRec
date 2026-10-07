import math

import torch
import torch.nn as nn


class Pose2DL1Loss(nn.Module):
    def __init__(self):
        super(Pose2DL1Loss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        mask = target[:, :, 3] == 1
        a = output[mask]
        b = target[mask]

        return self.l1(a[:, 0:2], b[:, 0:2])


class Pose3DL1Loss(nn.Module):
    def __init__(self):
        super(Pose3DL1Loss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        mask = target[:, :, 4] == 1
        a = output[mask]
        b = target[mask]
        # num_visible_joints = b.shape[0]

        return self.l1(a[:, 0:3], b[:, 0:3])


class BiternionLoss(nn.Module):
    """
    Cosine loss on (cos, sin) vectors for theta and phi (Biternion nets): 1 - cos(angle difference), computed with the
    normalized predicted vectors. A small penalty keeps the vector norm close to 1 (well conditioned atan2).
    Targets: B x 2 x 5 (theta / pi, phi / 2pi, score, theta visible, phi visible), as for AngularErrorLoss.
    """

    def __init__(self, norm_weight: float = 0.01):
        super(BiternionLoss, self).__init__()
        self.norm_weight = norm_weight
        self.nan = torch.tensor(float('nan'))

    def _cosine_loss(self, vectors, angles, mask):
        vectors = vectors[mask].float()
        angles = angles[mask]
        norm = torch.norm(vectors, dim=-1)
        cos_diff = (vectors[:, 0] * torch.cos(angles) + vectors[:, 1] * torch.sin(angles)) / norm.clamp_min(1e-6)
        return torch.mean(1 - cos_diff) + self.norm_weight * torch.mean((norm - 1) ** 2)

    def forward(self, theta_vectors, phi_vectors, target):
        mask_theta = target[:, :, 3] == 1
        mask_phi = target[:, :, 4] == 1
        losses = []
        if mask_theta.any():
            losses.append(self._cosine_loss(theta_vectors, target[:, :, 0] * math.pi, mask_theta))
        if mask_phi.any():
            losses.append(self._cosine_loss(phi_vectors, target[:, :, 1] * 2 * math.pi, mask_phi))
        if not losses:
            return self.nan.to(target.device)
        return sum(losses) / len(losses)


class JointConfLogitsLoss(nn.Module):
    """Joint confidence loss on logits (BCEWithLogitsLoss, numerically stable and autocast safe)."""

    def __init__(self):
        super(JointConfLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        mask = target[:, :, 4] == 1  # joint supported by the dataset
        if not mask.any():
            return torch.tensor(float('nan'), device=logits.device)
        return self.bce(logits[mask].float(), target[mask][:, 2].float())
