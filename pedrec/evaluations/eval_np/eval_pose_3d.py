import warnings

import numpy as np

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_PARENT_CHILD_PAIRS, SKELETON_PEDREC_JOINT as J, \
    SKELETON_PEDREC_LR_PAIRS


def get_euclidean_distances_joint_3d(target: np.ndarray, pred: np.ndarray):
    mask = (target[:, :, 3] != 1) | (target[:, :, 4] != 1)
    # Set all
    target_masked = target.copy()
    pred_masked = pred.copy()
    target_masked[mask] = 0
    pred_masked[mask] = 0
    return np.linalg.norm(target_masked[:, :, :3] - pred_masked[:, :, :3], axis=2)

def get_mpjpe(target: np.ndarray, pred: np.ndarray):
    """
    Returns the depth mean error in meters
    :param target: (n, num_joints, 2)
    :param pred: (n, num_joints, 2)
    :return: mean distance in meters
    """
    num_visibles_per_joint = np.sum(target[:, :, 3], axis=0)
    euclidean_distances = get_euclidean_distances_joint_3d(target, pred)
    mpjpe = np.sum(euclidean_distances, axis=0) / num_visibles_per_joint

    return mpjpe

def get_dim_distances(target: np.ndarray, pred: np.ndarray, dim: int):
    num_visibles_per_joint = np.sum(target[:, :, 3], axis=0)
    mask = (target[:, :, 3] != 1) | (target[:, :, 4] != 1)
    # Set all
    target_masked = target.copy()
    pred_masked = pred.copy()
    target_masked[mask] = 0
    pred_masked[mask] = 0

    distances = np.abs(target_masked[:, :, dim] - pred_masked[:, :, dim])
    distance_per_joint_mean = np.sum(distances, axis=0) / num_visibles_per_joint
    return distance_per_joint_mean


def get_relative_correct_depth(target: np.ndarray, pred: np.ndarray):
    """
    Checks if a joint n has the correct, relative depth to its parent joint m.
    A correct relative depth is assumed, if n is in front / back of the parent joint, as it is in the GT.
    This metric does not provide insights in any concrete measurements, thus one can not tell, if a joint is
    in the correct distance to its parent joint.

    Returns (pct_correct_depth_per_pair, pct_correct_depth_mean)
    """

    pct_correct_depth_per_pair = []
    pct_correct_only_visibles = []
    for pair_idx, pair in enumerate(SKELETON_PEDREC_PARENT_CHILD_PAIRS):
        joint_a_num = pair[0]
        joint_b_num = pair[1]

        #  remove invisible / n/a joints
        a = __get_visible_joint_array(target, joint_a_num, joint_b_num, target)
        if a.shape[0] == 0:
            pct_correct_depth_per_pair.append(0)
            continue
        b = __get_visible_joint_array(target, joint_b_num, joint_a_num, target)
        a_hat = __get_visible_joint_array(pred, joint_a_num, joint_b_num, target)
        b_hat = __get_visible_joint_array(pred, joint_b_num, joint_a_num, target)

        # compare depth order of gt and pred
        ab_gt = a[:, 2] < b[:, 2]
        ab_pred = a_hat[:, 2] < b_hat[:, 2]
        ab_correct = ab_gt == ab_pred
        num_corrects = np.count_nonzero(ab_correct)
        correct_pct = num_corrects / a.shape[0]
        pct_correct_depth_per_pair.append(correct_pct)
        pct_correct_only_visibles.append(correct_pct)

    pct_correct_depth_per_pair = np.array(pct_correct_depth_per_pair)
    pct_correct_depth = np.mean(pct_correct_only_visibles)

    return pct_correct_depth_per_pair, pct_correct_depth


def get_relative_correct_joint_positions(target: np.ndarray, pred: np.ndarray):
    """
    Checks if a joint n has the correct, relative depth to its parent joint m.
    A correct relative depth is assumed, if n is in front / back of the parent joint, as it is in the GT.
    This metric does not provide insights in any concrete measurements, thus one can not tell, if a joint is
    in the correct distance to its parent joint.

    Returns (pct_correct_depth_per_pair, pct_correct_depth_mean)
    """

    pct_correct_per_pair = []
    pct_correct_only_visibles = []
    for pair_idx, pair in enumerate(SKELETON_PEDREC_PARENT_CHILD_PAIRS):
        joint_a_num = pair[0]
        joint_b_num = pair[1]

        #  remove invisible / n/a joints
        a = __get_visible_joint_array(target, joint_a_num, joint_b_num, target)
        if a.shape[0] == 0:
            pct_correct_per_pair.append(0)
            continue
        b = __get_visible_joint_array(target, joint_b_num, joint_a_num, target)
        a_hat = __get_visible_joint_array(pred, joint_a_num, joint_b_num, target)
        b_hat = __get_visible_joint_array(pred, joint_b_num, joint_a_num, target)

        # compare depth order of gt and pred
        ab_x = a[:, 0] < b[:, 0]
        ab_x_pred = a_hat[:, 0] < b_hat[:, 0]
        ab_y = a[:, 1] < b[:, 1]
        ab_y_pred = a_hat[:, 1] < b_hat[:, 1]
        ab_z = a[:, 2] < b[:, 2]
        ab_z_pred = a_hat[:, 2] < b_hat[:, 2]

        x_correct = ab_x == ab_x_pred
        y_correct = ab_y == ab_y_pred
        z_correct = ab_z == ab_z_pred

        ab_correct = np.logical_and(x_correct, np.logical_and(y_correct, z_correct))
        num_corrects = np.count_nonzero(ab_correct)
        correct_pct = num_corrects / a.shape[0]
        pct_correct_per_pair.append(correct_pct)
        pct_correct_only_visibles.append(correct_pct)

    pct_correct_per_pair = np.array(pct_correct_per_pair)
    pct_correct_mean = np.mean(pct_correct_only_visibles)

    return pct_correct_per_pair, pct_correct_mean




def __get_visible_joint_array(arr: np.ndarray, joint_num_a: int, joint_num_b: int, target: np.ndarray):
    selected = arr[:, joint_num_a, :]
    return selected[(target[:, joint_num_a, 3] == 1) & (target[:, joint_num_b, 3] == 1)]



def get_joint_bias_body(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Median signed error (pred - target) per joint in body coordinates of the target: out (away from the body center,
    lateral axis signed per side), up (hip center -> shoulder center) and forward. Unlike the MPJPE it shows systematic
    offsets, e.g. a predicted skeleton convention that differs from the labels (hips 2 cm too wide: hip out +20). The
    median, as the L1 pose losses, is not pulled away by failures on hard poses (a mean would be).
    :return: J x 3 in the units of the input (mm), NaN for joints without labels
    """
    def joint(skeletons, j):
        return skeletons[:, j.value, :3]
    hip_center = (joint(target, J.left_hip) + joint(target, J.right_hip)) / 2
    up = (joint(target, J.left_shoulder) + joint(target, J.right_shoulder)) / 2 - hip_center
    up /= np.linalg.norm(up, axis=1, keepdims=True) + 1e-9
    lateral = joint(target, J.left_hip) - joint(target, J.right_hip)
    lateral -= np.sum(lateral * up, axis=1, keepdims=True) * up
    lateral /= np.linalg.norm(lateral, axis=1, keepdims=True) + 1e-9
    forward = np.cross(up, lateral)
    error = pred[:, :, :3] - target[:, :, :3]
    body = np.stack([np.sum(error * lateral[:, None], axis=2), np.sum(error * up[:, None], axis=2),
                     np.sum(error * forward[:, None], axis=2)], axis=2)
    for _, right in SKELETON_PEDREC_LR_PAIRS:
        body[:, right, 0] *= -1
    valid = (target[:, :, 3] == 1) & (target[:, :, 4] == 1)
    torso = np.all(valid[:, [J.left_hip.value, J.right_hip.value, J.left_shoulder.value, J.right_shoulder.value]], axis=1)
    valid &= torso[:, None]
    return _masked_median(body, valid)


def _masked_median(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """N x J x D values, N x J mask -> J x D median over the valid entries (NaN for joints without any)."""
    masked = np.where(valid[:, :, None], values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "All-NaN slice": joints without labels
        return np.nanmedian(masked, axis=0)
