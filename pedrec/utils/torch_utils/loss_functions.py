import math

import torch
import torch.nn as nn


class AngularErrorLoss(nn.Module):
    """
    expects 0 < phi < 2*pi, 0 < theta < pi, normalized to 0-1
    """

    def __init__(self):
        super(AngularErrorLoss, self).__init__()
        self.two_pi = 2 * math.pi
        self.nan = torch.tensor(float('nan'))

    def forward(self, output, target):
        num_visible_joints_theta = torch.sum(target[:, :, 3])
        num_visible_joints_phi = torch.sum(target[:, :, 4])
        if num_visible_joints_theta < 1 and num_visible_joints_phi < 1:
            return self.nan
        # Set predicted values of all invisible joints to 0 (ignore)
        mask_theta = target[:, :, 3] == 1
        target_theta_masked = target[mask_theta]
        output_theta_masked = output[mask_theta]
        theta = target_theta_masked[:, 0] * math.pi
        theta_ = output_theta_masked[:, 0] * math.pi

        mask_phi = target[:, :, 4] == 1
        target_phi_masked = target[mask_phi]
        output_phi_masked = output[mask_phi]
        phi = target_phi_masked[:, 1] * self.two_pi
        phi_ = output_phi_masked[:, 1] * self.two_pi
        dist_phi = torch.abs(phi_ - phi)

        if num_visible_joints_theta > 0 and num_visible_joints_phi > 0:
            dist_phi = torch.sum(torch.min(self.two_pi - dist_phi, dist_phi)) / num_visible_joints_phi
            dist_theta = torch.sum(torch.abs(theta - theta_)) / num_visible_joints_theta
            return (dist_phi + dist_theta) / 2
        elif num_visible_joints_phi > 0:
            return torch.sum(torch.min(self.two_pi - dist_phi, dist_phi)) / num_visible_joints_phi
        return torch.sum(torch.abs(theta - theta_)) / num_visible_joints_theta


class AngularErrorCartesianCoordinatesLoss(nn.Module):
    """
    expects
    phi_gt: 0 <= phi <= 2pi, normalized 0-1
    theta_gt: 0 <= theta <= pi, normalized 0-1

    preds: x, y cartesian coordinates (will be converted to polar for angular error)
    """

    def __init__(self):
        super(AngularErrorCartesianCoordinatesLoss, self).__init__()
        # self.l1 = nn.L1Loss()
        self.two_pi = 2 * math.pi
        # self.rad_to_degree_factor = 180 / math.pi
        self.nan = torch.tensor(float('nan'))

    def forward(self, output, target_o):
        target = target_o.clone()
        num_visible_joints = torch.sum(target[:, :, 3])
        if num_visible_joints < 1:
            return self.nan
        target[:, :, 0] *= math.pi  # theta
        target[:, :, 1] *= self.two_pi  # phi
        target_theta_x = torch.cos(target[:, :, 0])
        target_theta_y = torch.sin(target[:, :, 0])
        target_theta = torch.unsqueeze(torch.stack((target_theta_x, target_theta_y), dim=2), dim=2)
        target_theta[:, :, :, 0] = (target_theta[:, :, :, 0] + 1) / 2
        target_phi_x = torch.cos(target[:, :, 1])
        target_phi_y = torch.sin(target[:, :, 1])
        target_phi = torch.unsqueeze(torch.stack((target_phi_x, target_phi_y), dim=2), dim=2)
        target_phi = (target_phi + 1) / 2
        target_cartesian = torch.cat((target_theta, target_phi), dim=2)
        mask = target[:, :, 3] == 1
        a = output[mask]
        b = target_cartesian[mask]

        loss = torch.mean(torch.norm(a-b, dim=1))

        return loss


class Pose2DL1Loss(nn.Module):
    def __init__(self):
        super(Pose2DL1Loss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        mask = target[:, :, 3] == 1
        a = output[mask]
        b = target[mask]

        return self.l1(a[:, 0:2], b[:, 0:2])


class EnvPositionL2Loss(nn.Module):
    """
    Expects EnvPosition to be x, z, visible
    """
    def __init__(self):
        super(EnvPositionL2Loss, self).__init__()
        self.l2 = nn.MSELoss()

    def forward(self, output, target):
        mask = target[:, 2] == 1
        a = output[mask]
        b = target[mask]

        return self.l2(a[:, 0:2], b[:, 0:2])


class JointConfLoss(nn.Module):
    def __init__(self):
        super(JointConfLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, output, target):
        mask = target[:, :, 4] == 1  # check if joint is supported / correct value provided
        a = output[mask]
        b = target[mask]

        return self.bce(a[:, 2], b[:, 2])


class Pose2DL2Loss(nn.Module):
    def __init__(self):
        super(Pose2DL2Loss, self).__init__()
        # self.l2 = nn.MSELoss()

    def forward(self, output, target):
        mask = target[:, :, 3] == 1
        a = output[mask]
        b = target[mask]
        return torch.mean(torch.norm(a[:, 0:2] - b[:, 0:2], dim=len(b.shape)-1))


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


class Pose3DL2Loss(nn.Module):
    def __init__(self):
        super(Pose3DL2Loss, self).__init__()
        self.l2 = nn.MSELoss()

    def forward(self, output, target):
        mask = target[:, :, 4] == 1
        a = output[mask]
        b = target[mask]
        # return torch.mean(torch.norm(a[:, 0:3] - b[:, 0:3], dim=len(b.shape)-1))
        return self.l2(a[:, 0:3], b[:, 0:3])


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
