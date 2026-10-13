import random

import numpy as np
import torch

from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS, SKELETON_PEDREC_JOINT, \
    SKELETON_PEDREC_LR_PAIRS, SKELETON_PEDREC_TO_PEDRECEHPI3D
from pedrec.models.data_structures import ImageSize
from pedrec.utils.augmentation_helper import affine_transform_pt
from pedrec.utils.ehpi_helper import get_skeleton_sequence_in_ehpi_order
from pedrec.utils.skeleton_helper_3d import get_unit_skeleton_sequence


def worker_init_fn(id):
    seed = torch.initial_seed() // 2 ** 32 + id
    random.seed(seed)
    np.random.seed(seed)


def get_skeleton_2d_affine_transform(skeleton: np.ndarray, trans, input_size: ImageSize):
    for joint in SKELETON_PEDREC_JOINTS:
        joint_num = joint.value
        if skeleton[joint_num][3] > 0.0:
            skeleton[joint_num, 0:2] = affine_transform_pt(skeleton[joint_num, 0:2], trans)
            # UDP: the crop spans [0, W - 1] x [0, H - 1], see get_normalization_size
            if skeleton[joint_num, 0] > input_size.width - 1 \
                    or skeleton[joint_num, 1] > input_size.height - 1 \
                    or skeleton[joint_num, 0] < 0 \
                    or skeleton[joint_num, 1] < 0:
                skeleton[joint_num, 0:4] = 0
        else:
            skeleton[joint_num, 0:4] = 0
    return skeleton

def flip_skeleton_sequence_3d(skeleton_sequence_3d: np.ndarray):
    skeleton_sequence_3d[:len(SKELETON_PEDREC_JOINT), :, 0] *= -1

    # Change left-right parts
    for pair in SKELETON_PEDREC_LR_PAIRS:
        skeleton_sequence_3d[pair[0], :, :], skeleton_sequence_3d[pair[1], :, :] = \
            skeleton_sequence_3d[pair[1], :, :], skeleton_sequence_3d[pair[0], :, :].copy()
    return skeleton_sequence_3d

def normalize_skeleton_sequence_3d(skeleton_sequence_3d: np.ndarray, half_skeleton_range, skeleton_3d_range):
    skeleton_sequence_3d[:len(SKELETON_PEDREC_JOINT), :, :3] += half_skeleton_range  # move negatives to positive, scale 0-2
    skeleton_sequence_3d[:len(SKELETON_PEDREC_JOINT), :, :3] /= skeleton_3d_range  # normalize to 0-1
    return skeleton_sequence_3d


def get_ehpi_skeleton_sequence_3d(index, temporal_field: ImageSize, curr_frame_annotations, skeletons_3d, skeletons_2d, mode: DatasetType, flip: bool,
                                  half_skeleton_range, skeleton_3d_range, use_2d: bool = False, min_score: float = 0.0,
                                  use_unit_skeleton: bool = True, frame_sampling: int = 1):
    start_id = max(index - ((temporal_field.width - 1)*frame_sampling), curr_frame_annotations.scene_start)
    skeleton_sequence_3d = skeletons_3d[start_id:index + 1:frame_sampling].copy()
    # skeleton_sequence_3d = get_skeleton_sequence_in_ehpi_order(skeleton_sequence_3d)
    skeleton_sequence_3d = np.transpose(skeleton_sequence_3d, (1, 0, 2))
    if skeleton_sequence_3d.shape[1] < temporal_field.width or skeleton_sequence_3d.shape[
        0] < temporal_field.height:
        missing_frames = temporal_field.width - skeleton_sequence_3d.shape[1]
        missing_joints = temporal_field.height - skeleton_sequence_3d.shape[0]
        skeleton_sequence_3d = np.pad(skeleton_sequence_3d, ((0, missing_joints), (missing_frames, 0), (0, 0)))
    skeleton_sequence_3d[skeleton_sequence_3d[:, :, 3] < min_score] = 0
    skeleton_sequence_3d = skeleton_sequence_3d[:, :, :3]

    if np.min(skeleton_sequence_3d) == 0 and np.max(skeleton_sequence_3d) == 0:
        return skeleton_sequence_3d, False

    # # Append hip2d for global movement
    if use_2d:
        skeleton_sequence_2d = skeletons_2d[start_id:index + 1]
        length = abs(index + 1 - start_id - 32)
        skel_2d_valid_x = skeleton_sequence_2d[skeleton_sequence_2d[:, :, 0] > 0]
        skel_2d_valid_y = skeleton_sequence_2d[skeleton_sequence_2d[:, :, 1] > 0]
        if len(skel_2d_valid_y) > 0 and len(skel_2d_valid_y) > 0:
            skeleton_sequence_2d_min_x = np.min(skel_2d_valid_x[:, 0])
            skeleton_sequence_2d_min_y = np.min(skel_2d_valid_y[:, 1])
            hip_centers_2d = np.transpose(skeleton_sequence_2d, (1, 0, 2))[SKELETON_PEDREC_JOINT.hip_center.value, :, :2].copy()
            skeleton_sequence_3d[-1, length:, :2] = hip_centers_2d
            skeleton_sequence_3d[-1, length:, 0] = skeleton_sequence_3d[-1, length:, 0] - skeleton_sequence_2d_min_x
            skeleton_sequence_3d[-1, length:, 1] = skeleton_sequence_3d[-1, length:, 1] - skeleton_sequence_2d_min_y
            max_factor_x = 1 / np.max(skeleton_sequence_2d[:, :, 0])
            max_factor_y = 1 / np.max(skeleton_sequence_2d[:, :, 1])
            skeleton_sequence_3d[-1, length:, 0] = skeleton_sequence_3d[-1, length:, 0] * max_factor_x
            skeleton_sequence_3d[-1, length:, 1] = skeleton_sequence_3d[-1, length:, 1] * max_factor_y
    # a = 1

    if mode == DatasetType.TRAIN:
        if flip and random.random() <= 0.5:
            skeleton_sequence_3d = flip_skeleton_sequence_3d(skeleton_sequence_3d)

    skeleton_sequence_3d = normalize_skeleton_sequence_3d(skeleton_sequence_3d, half_skeleton_range, skeleton_3d_range)
    if use_unit_skeleton:
        skeleton_sequence_3d = get_unit_skeleton_sequence(skeleton_sequence_3d, nan_value=0)
        skeleton_sequence_3d[:len(SKELETON_PEDREC_JOINT), :, :3] += 20
        skeleton_sequence_3d[:len(SKELETON_PEDREC_JOINT), :, :3] /= 40
    skeleton_sequence_3d = get_skeleton_sequence_in_ehpi_order(skeleton_sequence_3d)
    return skeleton_sequence_3d, True
