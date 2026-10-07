import dataclasses

import torch
import torch.nn as nn

from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.networks.net_pedrec.pedrec_conv_transpose_base import PedRecConvTransposeBase
from pedrec.networks.net_pedrec.pedrec_orientation_head_shared import PedRecOrientationsHead
from pedrec.networks.net_pedrec.pedrec_pose_conf_head import PedRecPoseConfHead
from pedrec.networks.net_pedrec.pedrec_pose_conf_head_heatmap import PedRecPoseConfHeadHeatmap
from pedrec.networks.net_pedrec.pedrec_pose_head_2d import PedRecPose2DHead
from pedrec.networks.net_pedrec.pedrec_pose_head_3d import PedRecPose3DHead
from pedrec.networks.net_resnet.backbones import build_backbone, num_deconv_layers_for_stride
from pedrec.utils.torch_utils.loss_functions import AngularErrorLoss, Pose2DL1Loss, Pose3DL1Loss, JointConfLoss, \
    BiternionLoss, JointConfLogitsLoss


def _adapt_decoder_cfg(cfg: PedRecNetConfig, num_layers: int) -> PedRecNetConfig:
    """Decoder depth for the backbone stride (3 deconvs for stride 32 as in the original, 2 for stride 16, ...)."""
    if num_layers == cfg.model.num_deconv_layers:
        return cfg
    model = dataclasses.replace(cfg.model,
                                num_deconv_layers=num_layers,
                                num_deconv_filters=[cfg.model.num_deconv_filters[0]] * num_layers,
                                num_deconv_kernels=[cfg.model.num_deconv_kernels[0]] * num_layers,
                                num_pose_3d_deconv_layers=num_layers,
                                num_pose_3d_deconv_filters=[cfg.model.num_pose_3d_deconv_filters[0]] * num_layers,
                                num_pose_3d_deconv_kernels=[cfg.model.num_pose_3d_deconv_kernels[0]] * num_layers)
    return dataclasses.replace(cfg, model=model)


class PedRecNet(nn.Module):
    """
    Outputs (in this order):
      0 pose_coords_2d  B x J x 3 (x, y normalized to the input, joint confidence)
      1 pose_coords_3d  B x J x 4 (x, y, z normalized to the 3D range, joint confidence)
      2 orientations    B x 2 x 2 (body / head; theta / pi, phi / 2pi)
      3 pose_map_2d, 4 pose_map_3d (heatmap logits)
      5, 6 orientation bin logits (softargmax head) or (cos, sin) vectors (biternion head)
      7 joint confidence logits (only with the heatmap confidence head)
    """

    def __init__(self, cfg: PedRecNetConfig):
        super(PedRecNet, self).__init__()
        arch = cfg.arch
        self.feature_extractor = build_backbone(arch.backbone, cfg.model.input_size, arch.backbone_pretrained)
        cfg = _adapt_decoder_cfg(cfg, num_deconv_layers_for_stride(self.feature_extractor.stride))
        self.cfg = cfg
        num_heads = 1 if arch.shared_pose_deconv else 2
        self.conv_transpose_shared = PedRecConvTransposeBase(cfg, self.feature_extractor.out_channels,
                                                             num_heads=num_heads)
        heads = self.conv_transpose_shared.deconv_heads
        self.head_pose_2d = PedRecPose2DHead(cfg, heads[0])
        self.head_pose_3d = PedRecPose3DHead(cfg, heads[-1])
        self.head_orientation = PedRecOrientationsHead(cfg, self.feature_extractor.out_channels)
        self.conf_head_type = arch.conf_head
        self.head_conf = PedRecPoseConfHeadHeatmap(cfg) if arch.conf_head == "heatmap" else PedRecPoseConfHead(cfg)

    def forward(self, x):
        x = self.feature_extractor(x)
        x_deconv = self.conv_transpose_shared(x)
        pose_coords_2d, pose_map_2d = self.head_pose_2d(x_deconv)
        pose_coords_3d, pose_map_3d = self.head_pose_3d(x_deconv)
        conf_logits = None
        if self.conf_head_type == "heatmap":
            pose_conf, conf_logits = self.head_conf(pose_map_2d, pose_map_3d)
        else:
            pose_conf = self.head_conf(pose_map_2d, pose_map_3d)
        pose_conf = torch.unsqueeze(pose_conf, dim=2).to(pose_coords_2d.dtype)
        pose_coords_2d = torch.cat([pose_coords_2d, pose_conf], dim=2)
        pose_coords_3d = torch.cat([pose_coords_3d, pose_conf.to(pose_coords_3d.dtype)], dim=2)
        orientations, theta_map, phi_map = self.head_orientation(x, pose_coords_3d)

        outputs = (pose_coords_2d, pose_coords_3d, orientations, pose_map_2d, pose_map_3d, theta_map, phi_map)
        if conf_logits is not None:
            outputs = outputs + (conf_logits,)
        return outputs

    def init_weights(self):
        self.conv_transpose_shared.init_weights()
        self.head_pose_2d.init_weights()
        self.head_pose_3d.init_weights()
        self.head_orientation.init_weights()
        self.head_conf.init_weights()


class PedRecNetLossHead(nn.Module):
    """
    Multi task loss of 2D pose, 3D pose, orientation and joint confidence.

    weighting "pedrec" (original): sum 1 / (2 sigma_i^2) L_i + log(1 + prod sigma_i^2)
    weighting "kendall" (Kendall, Gal, Cipolla 2018): sum c_i exp(-s_i) L_i + s_i / 2 with s_i = log sigma_i^2
        (c = 1/2 for the regression tasks, 1 for the classification), s_i clamped to [-4, 4] for stability
    Tasks without labels in the batch (loss NaN) are skipped.
    """

    def __init__(self, device, use_p3d_loss: bool = True, use_orientation_loss: bool = True,
                 use_conf_loss: bool = True, weighting: str = "pedrec", orientation_head: str = "softargmax"):
        super(PedRecNetLossHead, self).__init__()
        self.pose_loss_2d = Pose2DL1Loss()
        self.pose_loss_3d = Pose3DL1Loss()
        self.orientation_head = orientation_head
        self.orientation_loss = BiternionLoss() if orientation_head == "biternion" else AngularErrorLoss()
        self.coord_conf_loss = JointConfLoss()
        self.coord_conf_logits_loss = JointConfLogitsLoss()
        self.weighting = weighting
        if weighting == "kendall":
            self.log_vars = nn.Parameter(torch.zeros(4))
        else:
            self.sigmas = nn.Parameter(torch.ones(4))
        self.device = device
        self.use_p3d_loss = use_p3d_loss
        self.use_orientation_loss = use_orientation_loss
        self.use_conf_loss = use_conf_loss

    def weighting_parameters(self) -> nn.Parameter:
        return self.log_vars if self.weighting == "kendall" else self.sigmas

    def reset_sigmas(self):
        if self.weighting == "kendall":
            nn.init.zeros_(self.log_vars)
        else:
            nn.init.constant_(self.sigmas, 1)

    def task_losses(self, outputs, targets):
        pose_2d_preds, pose_3d_preds, orientation_preds = outputs[0:3]
        losses = [self.pose_loss_2d(pose_2d_preds, targets["skeleton"]), None, None, None]
        if self.use_p3d_loss:
            losses[1] = self.pose_loss_3d(pose_3d_preds, targets["skeleton_3d"])
        if self.use_orientation_loss:
            if self.orientation_head == "biternion":
                losses[2] = self.orientation_loss(outputs[5], outputs[6], targets["orientation"])
            else:
                losses[2] = self.orientation_loss(orientation_preds, targets["orientation"])
        if self.use_conf_loss:
            if len(outputs) > 7 and outputs[7] is not None:
                losses[3] = self.coord_conf_logits_loss(outputs[7], targets["skeleton"])
            else:
                losses[3] = self.coord_conf_loss(pose_2d_preds, targets["skeleton"])
        return losses

    def forward(self, outputs, targets):
        losses = self.task_losses(outputs, targets)
        if self.weighting == "kendall":
            log_vars = self.log_vars.clamp(-4, 4)
            loss = 0
            for i, task_loss in enumerate(losses):
                if task_loss is None or torch.isnan(task_loss):
                    continue
                factor = 1.0 if i == 3 else 0.5
                loss = loss + factor * torch.exp(-log_vars[i]) * task_loss + 0.5 * log_vars[i]
            return loss

        sigma_prod = 1
        loss = 0
        for i, task_loss in enumerate(losses):
            if task_loss is None or torch.isnan(task_loss):
                continue
            factor = 1.0 if i == 3 else 0.5
            loss += (factor / self.sigmas[i] ** 2) * task_loss
            sigma_prod = sigma_prod * (self.sigmas[i] ** 2)
        loss = loss + torch.log(1 + sigma_prod)  # +1 to enforce positive loss
        return loss
