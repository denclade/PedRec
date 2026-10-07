import torch
import torch.nn as nn

from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.networks.net_pedrec.pedrec_conv_transpose_base import PedRecConvTransposeBase
from pedrec.networks.net_pedrec.pedrec_orientation_head_shared import PedRecOrientationsHead
from pedrec.networks.net_pedrec.pedrec_pose_conf_head_heatmap import PedRecPoseConfHeadHeatmap
from pedrec.networks.net_pedrec.pedrec_pose_head_2d import PedRecPose2DHead
from pedrec.networks.net_pedrec.pedrec_pose_head_3d import PedRecPose3DHead
from pedrec.networks.net_resnet.resnet_feature_extractor import ResNetHeadless
from pedrec.utils.torch_utils.loss_functions import Pose2DL1Loss, Pose3DL1Loss, BiternionLoss, JointConfLogitsLoss


class PedRecNet(nn.Module):
    """
    PedRecNet v2: ResNet-50 backbone with a shared deconvolution decoder and heads for 2D pose, 3D pose (heatmap +
    depth map with soft-argmax), joint confidence (heatmap statistics) and body / head orientation (biternion).

    Outputs (in this order):
      0 pose_coords_2d  B x J x 3 (x, y normalized to the input (UDP: by size - 1), joint confidence)
      1 pose_coords_3d  B x J x 4 (x, y, z normalized to the 3D range, joint confidence)
      2 orientations    B x 2 x 2 (body / head; theta / pi, phi / 2pi)
      3 pose_map_2d, 4 pose_map_3d (heatmap logits)
      5 theta vectors, 6 phi vectors (B x 2 x 2, (cos, sin) for body / head)
      7 joint confidence logits (B x J)

    The backbone, decoder and pose heads have the same layout as the published PedRecNet (v1), so training can start
    from the v1 checkpoints.
    """

    def __init__(self, cfg: PedRecNetConfig):
        super(PedRecNet, self).__init__()
        self.cfg = cfg
        self.feature_extractor = ResNetHeadless(cfg.layer.block, cfg.layer.layers)
        self.conv_transpose_shared = PedRecConvTransposeBase(cfg, self.feature_extractor.inplanes, num_heads=2)
        self.head_pose_2d = PedRecPose2DHead(cfg, self.conv_transpose_shared.deconv_heads[0])
        self.head_pose_3d = PedRecPose3DHead(cfg, self.conv_transpose_shared.deconv_heads[1])
        self.head_orientation = PedRecOrientationsHead(cfg)
        self.head_conf = PedRecPoseConfHeadHeatmap(cfg)

    def forward(self, x):
        x = self.feature_extractor(x)
        x_deconv = self.conv_transpose_shared(x)
        pose_coords_2d, pose_map_2d = self.head_pose_2d(x_deconv)
        pose_coords_3d, pose_map_3d = self.head_pose_3d(x_deconv)
        pose_conf, conf_logits = self.head_conf(pose_map_2d, pose_map_3d)
        pose_conf = torch.unsqueeze(pose_conf, dim=2)
        pose_coords_2d = torch.cat([pose_coords_2d, pose_conf.to(pose_coords_2d.dtype)], dim=2)
        pose_coords_3d = torch.cat([pose_coords_3d, pose_conf.to(pose_coords_3d.dtype)], dim=2)
        orientations, theta_vectors, phi_vectors = self.head_orientation(x, pose_coords_3d)
        return (pose_coords_2d, pose_coords_3d, orientations, pose_map_2d, pose_map_3d, theta_vectors, phi_vectors,
                conf_logits)

    def init_weights(self):
        self.conv_transpose_shared.init_weights()
        self.head_pose_2d.init_weights()
        self.head_pose_3d.init_weights()
        self.head_orientation.init_weights()
        self.head_conf.init_weights()


class PedRecNetLossHead(nn.Module):
    """
    Multi task loss with the uncertainty weighting of Kendall, Gal, Cipolla (CVPR 2018):

        sum_i c_i exp(-s_i) L_i + s_i / 2,   s_i = log sigma_i^2 (clamped to [-4, 4] for stability)

    with c = 1/2 for the regression tasks (2D pose, 3D pose, orientation) and 1 for the joint confidence
    (classification). Tasks without labels in the batch (loss NaN) are skipped; disabled tasks are not computed.
    """

    def __init__(self, device, use_p3d_loss: bool = True, use_orientation_loss: bool = True,
                 use_conf_loss: bool = True):
        super(PedRecNetLossHead, self).__init__()
        self.pose_loss_2d = Pose2DL1Loss()
        self.pose_loss_3d = Pose3DL1Loss()
        self.orientation_loss = BiternionLoss()
        self.conf_loss = JointConfLogitsLoss()
        self.log_vars = nn.Parameter(torch.zeros(4))
        self.device = device
        self.use_p3d_loss = use_p3d_loss
        self.use_orientation_loss = use_orientation_loss
        self.use_conf_loss = use_conf_loss

    def task_losses(self, outputs, targets):
        losses = [self.pose_loss_2d(outputs[0], targets["skeleton"]), None, None, None]
        if self.use_p3d_loss:
            losses[1] = self.pose_loss_3d(outputs[1], targets["skeleton_3d"])
        if self.use_orientation_loss:
            losses[2] = self.orientation_loss(outputs[5], outputs[6], targets["orientation"])
        if self.use_conf_loss:
            losses[3] = self.conf_loss(outputs[7], targets["skeleton"])
        return losses

    def forward(self, outputs, targets):
        log_vars = self.log_vars.clamp(-4, 4)
        loss = 0
        for i, task_loss in enumerate(self.task_losses(outputs, targets)):
            if task_loss is None or torch.isnan(task_loss):
                continue
            factor = 1.0 if i == 3 else 0.5
            loss = loss + factor * torch.exp(-log_vars[i]) * task_loss + 0.5 * log_vars[i]
        return loss
