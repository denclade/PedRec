import sys

sys.path.append('.')  # allow running as a script from the repository root

import numpy as np
import pandas as pd
from pedrec.utils.pandas_helper import read_pedrec_df

from pedrec.evaluations.eval_np.eval_angular import get_angular_error_statistics, get_angular_distances
from pedrec.evaluations.eval_np.eval_pose_2d import get_pck_normalized_joint_distances, get_ref_distance_torso, \
    get_normalized_joint_distances
from pedrec.evaluations.eval_np.eval_pose_3d import get_euclidean_distances_joint_3d
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.utils.pedrec_dataset_helper import get_gt_arrays, get_pred_arrays


def get_column_names():
    column_names = [
        "index"
    ]
    for joint in SKELETON_PEDREC_JOINTS:
        column_names.append(f"skeleton2d_{joint.name}_dist")


    for joint in SKELETON_PEDREC_JOINTS:
        column_names.append(f"skeleton3d_{joint.name}_dist")

    column_names.append("body_orientation_phi_dist")
    column_names.append("body_orientation_theta_dist")

    column_names.append("head_orientation_phi_dist")
    column_names.append("head_orientation_theta_dist")

    return column_names


def get_pck_2d(gt_array, pred_array, visible_array, threshold):
    ref_distances = get_ref_distance_torso(gt_array)
    normalized_joint_distances = get_normalized_joint_distances(gt_array, pred_array, visible_array, ref_distances)
    # pck, normalized_joint_distances = get_pck_normalized_joint_distances(gt_array, pred_array, visible_array, threshold,
    #                                                                      ref_distances)
    return normalized_joint_distances


def evaluate_conti_catalogue(dataset_df_path: str, result_df_path: str):
    gt_df = read_pedrec_df(dataset_df_path)
    pred_df = read_pedrec_df(result_df_path)
    skeleton2ds_gt, skeleton3ds_gt, body_orientations_gt, head_orientations_gt, bbs_gt, env_positions_gt = get_gt_arrays(gt_df)
    skeleton2ds_pred, skeleton3ds_pred, body_orientations_pred, head_orientations_pred = get_pred_arrays(pred_df)
    normalized_joint_distances_2d = get_pck_2d(skeleton2ds_gt[:, :, 0:2], skeleton2ds_pred[:, :, 0:2], skeleton2ds_gt[:, :, 3], 0.2)

    visible_joints = skeleton2ds_gt[:, :, 3]
    joint_distances_2d = normalized_joint_distances_2d
    joint_distances_3d = get_euclidean_distances_joint_3d(skeleton2ds_gt, skeleton2ds_pred)
    body_dists_phi, body_dists_theta, body_spherical_distances = get_angular_distances(np.squeeze(body_orientations_gt), np.squeeze(body_orientations_pred))
    head_dists_phi, head_dists_theta, head_spherical_distances = get_angular_distances(np.squeeze(head_orientations_gt),
                                                                                    np.squeeze(head_orientations_pred))
    body_angle_visibles = np.squeeze(body_orientations_gt[:, :, 3])
    head_angle_visibles = np.squeeze(head_orientations_gt[:, :, 3])

    # TODO: set all distances from invisible stuff to nan
    skeleton2d_mask = (skeleton2ds_gt[:, :, 3] != 1) | (skeleton2ds_gt[:, :, 4] != 1)

    joint_distances_2d[skeleton2d_mask] = np.nan
    joint_distances_2d[np.isinf(joint_distances_2d)] = np.nan
    skeleton2ds_joint_distances_mean = np.nansum(joint_distances_2d, axis=1) / np.count_nonzero(~np.isnan(joint_distances_2d), axis=1)
    skeleton2ds_joint_distances_mean[np.isinf(skeleton2ds_joint_distances_mean)] = np.nan
    
    # skeleton3d_mask = (skeleton3ds_gt[:, :, 4] != 1) | (skeleton3ds_gt[:, :, 5] != 1)
    joint_distances_3d[skeleton2d_mask] = np.nan
    joint_distances_3d[np.isinf(joint_distances_3d)] = np.nan
    skeleton3ds_joint_distances_mean = np.nansum(joint_distances_3d, axis=1) / np.count_nonzero(~np.isnan(joint_distances_3d), axis=1)
    skeleton3ds_joint_distances_mean[np.isinf(skeleton3ds_joint_distances_mean)] = np.nan
    # TODO: Get mean distance joints

    a = 1

if __name__ == "__main__":
    evaluate_conti_catalogue("data/datasets/Conti01/rt_conti_01_a.pkl",
                             "data/datasets/Conti01/rt_conti_01_a_results_with_orientation.pkl")