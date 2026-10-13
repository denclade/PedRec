import numpy as np
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS, SKELETON_PEDREC_JOINT
# TODO: Add PCP? https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Sapp_MODEC_Multimodal_Decomposable_2013_CVPR_paper.pdf


def get_normalized_joint_distances(gt_array: np.ndarray, pred_array: np.ndarray, visible_array: np.array, ref_distances: np.ndarray):
    # TODO: use head for PCKh

    assert gt_array.shape == pred_array.shape
    num_joints = gt_array.shape[1]

    joint_distances = get_joint_distances(pred_array, gt_array)
    # remove entries without ref distance to prevent zero devision
    scales = get_scale_foreach_joint(num_joints, ref_distances)
    # Divide each joint distance by headsize factor for normalization
    normalized_joint_distances = np.divide(joint_distances, scales)
    # set not visible joint errors to 0 -- TODO: Visibility state from gt?
    return np.multiply(normalized_joint_distances, visible_array)


def get_pck_results(gt_array: np.ndarray, pred_array: np.ndarray, visible_array: np.array, threshold: float = 0.5):
    """
    n = number of records
    :param gt_array: (n, num_joints, 2)
    :param pred_array: (n, num_joints, 2)
    :param visible_array: (n, num_joints) # 0 if invisible, 1 if visible
    :param threshold: PCK threshold
    :return:
    """

    ref_distances = get_ref_distance_torso(gt_array)
    pck, normalized_joint_distances = get_pck_normalized_joint_distances(gt_array, pred_array, visible_array, threshold, ref_distances)

    return pck


def get_pck_normalized_joint_distances(gt_array: np.ndarray, pred_array: np.ndarray, visible_array: np.array, threshold, ref_distances):
    """
    n = number of records
    :param gt_array: (n, num_joints, 2)
    :param pred_array: (n, num_joints, 2)
    :param visible_array: (n, num_joints) # 0 if invisible, 1 if visible
    :param threshold: PCK threshold
    :return:
    """

    gt_array = gt_array.copy()[ref_distances != 0]
    pred_array = pred_array.copy()[ref_distances != 0]
    visible_array = visible_array.copy()[ref_distances != 0]
    ref_distances = ref_distances[ref_distances != 0]

    normalized_joint_distances = get_normalized_joint_distances(gt_array, pred_array, visible_array, ref_distances)
    # Counts how many skeletons contain a joint (thus 16, 1)
    num_visible_joints_per_joint = np.sum(visible_array, axis=0)

    # Get the PCK for each joint
    pck = get_pck(normalized_joint_distances, threshold, num_visible_joints_per_joint, visible_array)

    # for joint in SKELETON_PEDREC_JOINTS:
    #     print("{}: {:.2f}%".format(joint.name, pck[joint.value]))
    # pck_wo_nans = pck[~np.isnan(pck)]
    # print("Mean (w/o) NAN: {:.2f}%".format(np.sum(pck_wo_nans) / len(pck_wo_nans)))
    return pck, normalized_joint_distances


#
def get_pck_range(gt_array: np.ndarray, pred_array: np.ndarray, visible_array: np.array, rng: np.ndarray):
    """
    rng => range, example: rng = np.linspace(0, 0.5, 50)
    """
    # num rngs * num_joints matrix
    normalized_joint_distances = get_normalized_joint_distances(gt_array, pred_array, visible_array)
    num_visible_joints_per_joint = np.sum(visible_array, axis=0)

    pck_range = np.zeros((len(rng), gt_array.shape[1]))
    for r in range(len(rng)):
        threshold = rng[r]
        pck_range[r, :] = get_pck(normalized_joint_distances, threshold, num_visible_joints_per_joint, visible_array)
    return pck_range


def get_pck(normalized_joint_distances, threshold, num_visible_joints_per_joint: np.ndarray,
            joints_visible: np.ndarray):
    joint_distances_below_threshold = get_visible_joint_distances_below_threshold(normalized_joint_distances, threshold,
                                                                                  joints_visible)
    # joints without any visible ground truth in the set (e.g. the PedRec joints COCO does not have) -> NaN, ignored
    # by the mean
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.divide(100. * np.sum(np.transpose(joint_distances_below_threshold, (1, 0)), axis=1),
                         num_visible_joints_per_joint)


def get_visible_joint_distances_below_threshold(normalized_joint_distances: np.ndarray, threshold: float,
                                                joints_visible: np.ndarray):
    # Foreach joint in each skeleton checks if the error is below the threshold and if the joint is visible, if so set to 1,
    # all invisibles or joints above the threshold are set to 1
    return np.multiply((normalized_joint_distances < threshold), joints_visible)


def get_joint_distances(pred_array: np.ndarray, gt_array: np.ndarray):
    # First calculate the distance of each x and y coordinate
    joint_distances = pred_array - gt_array
    # Then use the diagonal size of the "distance rectangle with x_dist, y_dist sides"
    return np.linalg.norm(joint_distances, axis=2)


def get_scale_foreach_joint(num_joints: int, ref_distances: np.ndarray):
    """
    ref distances: 1 dimensional array with length: num of data (gt_array.shape[0])
    :param num_joints:
    :param ref_distances:
    :return:
    """
    ones = np.ones((num_joints, 1))
    # Headsizes (29xx long vec, for each skeleton one) is expanded to 16, 29xx, copying each value 16 times for joints
    scales = np.multiply(ref_distances, ones)
    return np.transpose(scales, (1, 0))


def get_ref_distance_torso(gt_array: np.array):
    """
    GT Array: e.g. 3, 19, 2
    calc_array: e.g. 3, 2, 2
    gt_distances: e.g. 3, 2
    :param gt_array:
    :return:
    """
    calc_array = gt_array[:, [SKELETON_PEDREC_JOINT.right_shoulder.value, SKELETON_PEDREC_JOINT.left_hip.value], :]
    gt_distances = calc_array[:, 1, :] - calc_array[:, 0, :]
    return np.linalg.norm(gt_distances, axis=1)
    # TODO: Remove the ones where a joint is not visible?
