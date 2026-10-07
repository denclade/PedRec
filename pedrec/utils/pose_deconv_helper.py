import logging
import math
from typing import Union, List, Tuple, Dict

import cv2
import numpy as np
import torch
from torchvision import transforms

from pedrec.configs.pedrec_net_config import PedRecNetModelConfig, PedRecNetConfig
from pedrec.models.human import Human
from pedrec.utils.augmentation_helper import get_affine_transforms
from pedrec.utils.bb_helper import get_bb_class_idx, bb_to_center_scale
from pedrec.utils.skeleton_helper import get_skeleton_mean_score
from pedrec.utils.torch_utils.torch_helper import affine_transform_coords_2d

logger = logging.getLogger(__name__)
pose_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# def transform_preds(coords, center, scale, output_size):
#     target_coords = np.zeros(coords.shape)
#     trans = get_affine_transform(center, scale, 0, output_size, inv=1)
#     for p in range(coords.shape[0]):
#         target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
#         target_coords[p, 2] = coords[p, 2]
#     return target_coords


def get_pose_estimation_prediction(pose_model: torch.nn.Module,
                                   cfg: PedRecNetConfig,
                                   img: np.ndarray,
                                   centers: List[np.ndarray],
                                   scales: List[np.ndarray],
                                   transform,
                                   device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    trans_invs = []

    # PoseResnet pretrain was trained on BGR, thus keep it for now
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for center, scale in zip(centers, scales):
        trans, trans_inv = get_affine_transforms(center, scale, rotation, cfg.model.input_size, add_inv = True)
        trans_invs.append(trans_inv)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            img,
            trans,
            (int(cfg.model.input_size.width), int(cfg.model.input_size.height)),
            flags=cv2.INTER_LINEAR)
        model_input = transform(model_input)
        model_inputs.append(model_input)
    trans_invs = torch.tensor(np.array(trans_invs), dtype=torch.float32).to(device)
    model_inputs_torch = torch.stack(model_inputs)
    output = pose_model(model_inputs_torch.to(device))

    pose_coords_2d = output[0]
    pose_coords_2d[:, :, 0] *= cfg.model.input_size.width
    pose_coords_2d[:, :, 1] *= cfg.model.input_size.height
    for i in range(pose_coords_2d.shape[0]):
        pose_coords_2d[i, :, :2] = affine_transform_coords_2d(pose_coords_2d[i], trans_invs[i], pose_coords_2d.device)

    pose_2d_pred = pose_coords_2d.cpu().detach().numpy()
    pose_3d_pred = output[1].cpu().detach().numpy()
    orientation_pred = output[2].cpu().detach().numpy()

    # return normalized values as well as the denormalized ones... 

    # denormaliz
    orientation_pred[:, :, 0] *= math.pi
    orientation_pred[:, :, 1] *= 2 * math.pi
    pose_3d_pred[:, :, :3] *= 3000
    pose_3d_pred[:, :, :3] -= 1500

    return pose_2d_pred, pose_3d_pred, orientation_pred, model_inputs


def pedrec_recognizer(pose_model: Union[torch.nn.Module],
                      cfg: PedRecNetConfig,
                      img: np.ndarray, bbs: List[np.ndarray], device) -> Dict[str, np.ndarray]:
    centers = []
    scales = []
    for bb in bbs:
        assert len(bb) > 0 and get_bb_class_idx(bb) == 0
        center, scale = bb_to_center_scale(bb, cfg.model.input_size)
        centers.append(center)
        scales.append(scale)
    pose2d_preds = []
    pose3d_preds = []
    orientation_preds = []
    model_input_bbs = []
    env_position_preds = []
    if len(centers) > 0:
        pose2d_preds, pose3d_preds, orientation_preds, model_input_bbs = get_pose_estimation_prediction(pose_model, cfg, img, centers, scales,
                                                                                transform=pose_transform, device=device)
    return {"skeletons": pose2d_preds,
            "skeletons_3d": pose3d_preds,
            "orientations": orientation_preds,
            "model_input_bbs": model_input_bbs}


redetect_count = 0


def do_redetect_pose_recognition(pose_model: Union[torch.nn.Module],
                                 cfg: Union[PedRecNetModelConfig],
                                 img: np.ndarray, humans: List[Human], device, min_skeleton_score: float = 0.4) -> \
        Tuple[List[Human], torch.Tensor]:
    if len(humans) == 0:
        return humans
    global redetect_count
    # print("REDETECT {}".format(redetect_count))
    redetect_count += 1
    centers = []
    scales = []
    for human in humans:
        center, scale = bb_to_center_scale(human.bb, cfg.model.input_size)
        centers.append(center)
        scales.append(scale)
    pose2d_preds = []
    pose3d_preds = []
    orientation_preds = []
    if len(centers) > 0:
        pose2d_preds, pose3d_preds, orientation_preds, model_input_bbs = get_pose_estimation_prediction(pose_model, cfg, img, centers, scales,
                                                                             transform=pose_transform, device=device)

    humans_above_score = []
    for human, pose2d_pred, pose3d_pred, orientation_pred in zip(humans, pose2d_preds, pose3d_preds, orientation_preds):
        if get_skeleton_mean_score(pose2d_pred) < min_skeleton_score:
            continue
        human.skeleton_2d = pose2d_pred
        human.skeleton_3d = pose3d_pred
        human.orientation = orientation_pred
        humans_above_score.append(human)
    return humans_above_score
