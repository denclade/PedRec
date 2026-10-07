import math
from typing import List, Dict

import numpy as np

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_LR_PAIRS, SKELETON_PEDREC_JOINT, \
    SKELETON_PEDREC_JOINTS, SKELETON_PEDREC_BUILDER

"""
The base skeleton used in this application is the one of MS COCO
joint structure: x, y, z, score, vis
"""

def get_unit_skeleton(skeleton_3d: np.ndarray):
    joints = np.zeros((len(SKELETON_PEDREC_JOINTS), 4), dtype=np.float32)
    for limb_idx, limb in enumerate(SKELETON_PEDREC_BUILDER):
        limb_length = limb[2]
        joint_a = skeleton_3d[limb[0]]
        joint_b = skeleton_3d[limb[1]]
        direction = np.array([joint_b[0] - joint_a[0],
                              joint_b[1] - joint_a[1],
                              joint_b[2] - joint_a[2]], dtype=np.float32)

        normalized_direction = direction / np.linalg.norm(direction)
        joints[limb[1], :3] = joints[limb[0], :3] + normalized_direction * limb_length
    joints[:, 3] = 1
    return joints

def get_unit_skeleton_sequence(skeleton_3d_sequence: np.ndarray, nan_value: float = 0):
    skeleton_3d_sequence = np.transpose(skeleton_3d_sequence, (1, 0, 2))
    joints = np.zeros(skeleton_3d_sequence.shape, dtype=np.float32)
    for limb_idx, limb in enumerate(SKELETON_PEDREC_BUILDER):
        limb_length = limb[2]
        joints_a = skeleton_3d_sequence[:, limb[0]]
        joints_b = skeleton_3d_sequence[:, limb[1]]
        direction = np.array([joints_b[:, 0] - joints_a[:, 0],
                              joints_b[:, 1] - joints_a[:, 1],
                              joints_b[:, 2] - joints_a[:, 2]], dtype=np.float32)

        direction = np.transpose(direction, (1, 0))
        with np.errstate(invalid="ignore", divide="ignore"):  # zero padded frames -> nan -> 0 below
            normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
        # normalized_direction = np.transpose(normalized_direction, (1, 0))
        joints[:, limb[1], :3] = joints[:, limb[0], :3] + normalized_direction * limb_length
    # joints[:, :, 3] = 1
    # copy data not from SKELETON3D to the EHPI
    joints[:, len(SKELETON_PEDREC_JOINT):, :] = skeleton_3d_sequence[:, len(SKELETON_PEDREC_JOINT):, :]
    joints = np.nan_to_num(np.transpose(joints, (1, 0, 2)), nan=nan_value)
    return joints


def flip_lr_orientation(orientation: np.ndarray) -> np.ndarray:
    phi = orientation[0, 1] * 2 * math.pi
    phi_flipped = np.mod(math.pi - phi, 2 * math.pi)
    orientation[0, 1] = phi_flipped / (2 * math.pi)
    return orientation


def flip_lr_joints_3d(skeleton_3d: np.ndarray) -> np.ndarray:
    skeleton_3d[:, 0] *= -1

    # Change left-right parts
    for pair in SKELETON_PEDREC_LR_PAIRS:
        skeleton_3d[pair[0], :], skeleton_3d[pair[1], :] = \
            skeleton_3d[pair[1], :], skeleton_3d[pair[0], :].copy()
    return skeleton_3d


def set_invisible_coords_to_zero_3d(skeleton_3d: np.ndarray) -> np.ndarray:
    return skeleton_3d[0:3] * skeleton_3d[5]


def get_joint_score_3d(joint_3d: np.ndarray) -> float:
    return joint_3d[3]


def get_joint_visibility_3d(joint_3d: np.ndarray) -> float:
    return joint_3d[4]


def is_joint_visible_3d(joint_3d: np.ndarray) -> bool:
    return joint_3d[4] > 0


def joint_has_value_3d(joint_3d: np.ndarray) -> float:
    return joint_3d[0] > 0 or joint_3d[1] > 0 or joint_3d[2] > 0


def get_skeleton_mean_score_3d(skeleton_3d: np.ndarray):
    confidence = 0.0
    for joint in skeleton_3d:
        confidence += get_joint_score_3d(joint)
    confidence /= skeleton_3d.shape[0]
    return confidence


def get_euclidean_distance_joint_3d(joint_3d_a: np.ndarray, joint_3d_b: np.ndarray) -> float:
    return np.linalg.norm(joint_3d_b[0:3] - joint_3d_a[0:3])


def get_euclidean_distance_joint_lists(joints_3d_a: np.ndarray,
                                       joints_3d_b: np.ndarray,
                                       min_joint_score: float = 0.0) -> List[float]:
    """
    Returns the distance of the correspondiong joints of two lists. The lists must have the same length
    :param min_joint_score: The minimum score for both joints to be included in the distance check
    :param joints_3d_a:
    :param joints_3d_b:
    :return: List of floats for each joint_id in the lists with the euclidean distance
    """
    assert len(joints_3d_a) == len(joints_3d_b)
    joint_distances = []
    for joint_id, joint_tuple in enumerate(zip(joints_3d_a, joints_3d_b)):
        joint_a, joint_b = joint_tuple
        if get_joint_score_3d(joint_a) >= min_joint_score and get_joint_score_3d(joint_b) >= min_joint_score:
            joint_distances.append(get_euclidean_distance_joint_3d(joint_a, joint_b))
    return joint_distances


def get_middle_joint_3d(joint_3d_a: np.ndarray, joint_3d_b: np.ndarray, check_visibility: bool = True) -> np.ndarray:
    """
    Returns a joint which is in the middle of the two input joints. The visibility and score is estimated by the
    visibility and score of the two surrounding joints.
    :param joint_3d_a: Surrounding joint one
    :param joint_3d_b: Surrounding joint two
    :return: Joint in the middle of joint_a and joint_b
    """
    visibility = 1
    if check_visibility:
        joint_a_visibility = get_joint_visibility_3d(joint_3d_a)
        joint_b_visibility = get_joint_visibility_3d(joint_3d_b)
        visibility = min(joint_a_visibility, joint_b_visibility)
    # if not joint_a_visibility or not joint_b_visibility:
    #     return np.zeros(joint_a.shape, dtype=np.float32)
    x = (joint_3d_a[0] + joint_3d_b[0]) / 2
    y = (joint_3d_a[1] + joint_3d_b[1]) / 2
    z = (joint_3d_a[2] + joint_3d_b[2]) / 2
    score = (get_joint_score_3d(joint_3d_a) + get_joint_score_3d(joint_3d_a)) / 2
    return np.array([x, y, z, score, visibility], dtype=np.float32)

def get_skeleton_dict(skeleton):
    skeleton_dict: Dict[str, np.float32] = {}
    for joint in SKELETON_PEDREC_JOINTS:
        skeleton_dict[joint.name] = skeleton[joint.value]
    return skeleton_dict

def try_get_left_leg_size(skeleton, min_score: float = 0.5):
    left_ankle = skeleton[SKELETON_PEDREC_JOINT.left_ankle.value]
    left_knee = skeleton[SKELETON_PEDREC_JOINT.left_knee.value]
    left_hip = skeleton[SKELETON_PEDREC_JOINT.left_hip.value]
    if get_joint_score_3d(left_ankle) > min_score and \
           get_joint_score_3d(left_knee) > min_score and \
           get_joint_score_3d(left_hip) > min_score:
        return get_euclidean_distance_joint_3d(left_ankle, left_knee) + get_euclidean_distance_joint_3d(left_knee, left_hip)
    return None


def try_get_right_leg_size(skeleton, min_score: float = 0.5):
    right_ankle = skeleton[SKELETON_PEDREC_JOINT.right_ankle.value]
    right_knee = skeleton[SKELETON_PEDREC_JOINT.right_knee.value]
    right_hip = skeleton[SKELETON_PEDREC_JOINT.right_hip.value]
    if get_joint_score_3d(right_ankle) > min_score and \
           get_joint_score_3d(right_knee) > min_score and \
           get_joint_score_3d(right_hip) > min_score:
        return get_euclidean_distance_joint_3d(right_ankle, right_knee) + get_euclidean_distance_joint_3d(right_knee, right_hip)
    return None


def try_get_spine_size(skeleton, min_score: float = 0.5):
    hip_center = skeleton[SKELETON_PEDREC_JOINT.hip_center.value]
    spine_center = skeleton[SKELETON_PEDREC_JOINT.spine_center.value]
    neck = skeleton[SKELETON_PEDREC_JOINT.neck.value]
    if get_joint_score_3d(hip_center) > min_score and \
           get_joint_score_3d(spine_center) > min_score and \
           get_joint_score_3d(neck) > min_score:
        return get_euclidean_distance_joint_3d(hip_center, spine_center) + get_euclidean_distance_joint_3d(spine_center, neck)
    return None


def try_get_head_size(skeleton, min_score: float = 0.5):
    neck = skeleton[SKELETON_PEDREC_JOINT.neck.value]
    lower_head = skeleton[SKELETON_PEDREC_JOINT.head_lower.value]
    upper_head = skeleton[SKELETON_PEDREC_JOINT.head_upper.value]
    if get_joint_score_3d(neck) > min_score and \
           get_joint_score_3d(lower_head) > min_score and \
           get_joint_score_3d(upper_head) > min_score:
        return get_euclidean_distance_joint_3d(neck, lower_head) + get_euclidean_distance_joint_3d(lower_head, upper_head)
    return None


def get_human_size_from_skeleton_3d(skeleton, addition: float = 5) -> float:
    """
    Addition
    """
    spine_size = try_get_spine_size(skeleton)
    if spine_size is None:
        return -1
    return (spine_size / 3) * 8

    # leg_size = try_get_left_leg_size(skeleton)
    # if leg_size is None:
    #     leg_size = try_get_right_leg_size(skeleton)
    # if leg_size is None:
    #     # alternative size estimation
    #     a = 1
    #     return -1
    # spine_size = try_get_spine_size(skeleton)
    # if spine_size is None:
    #     # alternative size estimation
    #     a = 1
    #     return -1
    # head_size = try_get_head_size(skeleton)
    # if head_size is None:
    #     # estimate head size
    #     a = 1
    #     return -1
    # ankle_to_head_size = leg_size + spine_size + head_size
    # ankle_to_head_size += addition  # add foot height estimate (cm)
    # return ankle_to_head_size