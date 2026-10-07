from typing import List

import numpy as np

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_LR_PAIRS

"""
The base skeleton used in this application is the one of MS COCO
joint structure: x, y, score, vis
"""

def flip_lr_joints(skeleton: np.ndarray, img_width: int) -> np.ndarray:
    skeleton[:, 0] = img_width - skeleton[:, 0] - 1

    # Change left-right parts
    for pair in SKELETON_PEDREC_LR_PAIRS:
        skeleton[pair[0], :], skeleton[pair[1], :] = \
            skeleton[pair[1], :], skeleton[pair[0], :].copy()
    return skeleton


def set_invisible_coords_to_zero(skeleton: np.ndarray) -> np.ndarray:
    return skeleton[0:2] * skeleton[4]


def get_joint_score(joint: np.ndarray) -> float:
    return joint[2]


def get_joint_visibility(joint: np.ndarray) -> float:
    return joint[3]


def is_joint_visible(joint: np.ndarray) -> bool:
    return joint[3] > 0


def joint_has_value(joint: np.ndarray) -> float:
    return joint[0] > 0 or joint[1] > 0


def get_skeleton_mean_score(skeleton: np.ndarray) -> float:
    return float(np.sum(skeleton[:, 2], dtype=np.float64) / skeleton.shape[0])


def get_euclidean_distance_joint(joint_a: np.ndarray, joint_b: np.ndarray) -> float:
    return np.linalg.norm(joint_b[0:2] - joint_a[0:2])


def get_euclidean_joint_distances(skeleton_a: np.ndarray, skeleton_b: np.ndarray, joint_ids: List[int],
                                  coord_split_idx: int, min_joint_score: float):
    """
    Returns the distance of the correspondiong joints of two lists. The lists must have the same skeleton hierarchy.

    coord_split_idx: Index at which the coordinates end, in 2d usually 2, 3d 3.
    """
    joints_a = skeleton_a[joint_ids]
    joints_b = skeleton_b[joint_ids]
    score_filter = (joints_a[:, coord_split_idx] > min_joint_score) & (joints_b[:, coord_split_idx] > min_joint_score)
    joints_a = joints_a[score_filter]
    joints_b = joints_b[score_filter]
    joint_distances = np.linalg.norm(joints_a[:, :coord_split_idx] - joints_b[:, :coord_split_idx], axis=1)
    return joint_distances


def get_middle_joint(joint_a: np.ndarray, joint_b: np.ndarray, check_visibility: bool = True) -> np.ndarray:
    """
    Returns a joint which is in the middle of the two input joints. The visibility and score is estimated by the
    visibility and score of the two surrounding joints.
    :param joint_a: Surrounding joint one
    :param joint_b: Surrounding joint two
    :return: Joint in the middle of joint_a and joint_b
    """
    visibility = 1
    if check_visibility:
        joint_a_visibility = get_joint_visibility(joint_a)
        joint_b_visibility = get_joint_visibility(joint_b)
        visibility = min(joint_a_visibility, joint_b_visibility)
    # if not joint_a_visibility or not joint_b_visibility:
    #     return np.zeros(joint_a.shape, dtype=np.float32)
    x = (joint_a[0] + joint_b[0]) / 2
    y = (joint_a[1] + joint_b[1]) / 2
    score = (get_joint_score(joint_a) + get_joint_score(joint_a)) / 2
    return np.array([x, y, score, visibility], dtype=np.float32)
