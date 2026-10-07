import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import math
import os
from pathlib import Path

from pedrec.evaluations.eval_helper import get_skel_coco, get_skel_h36m, get_skel_h36m_handfootends
from pedrec.evaluations.eval_np.eval_angular import get_angular_distances
from pedrec.evaluations.validate import get_2d_pose_pck_results, get_3d_pose_results
from pedrec.models.constants.skeleton_coco import SKELETON_COCO_JOINTS
from pedrec.models.constants.skeleton_h36m import SKELETON_H36M_JOINTS
from pedrec.models.constants.skeletons import SKELETON
from pedrec.models.validation.orientation_validation_results import FullOrientationValidationResult
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.pedrec_stages import STAGES
from pedrec.utils.skeleton_helper import flip_lr_joints
from pedrec.utils.skeleton_helper_3d import flip_lr_joints_3d, flip_lr_orientation

import numpy as np
import pandas as pd
from pedrec.utils.pandas_helper import read_pedrec_df
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS


sim_c01_root = "data/datasets/Conti01"  # set by main()
sim_c01_val_filename = "rt_conti_01_val_FIN.pkl"


def get_df(dataset_root, result_filename):
    dataset_df_path = os.path.join(dataset_root, "results", result_filename)
    return read_pedrec_df(dataset_df_path)


def get_skeleton2d(df):
    filter_skeleton2d = [col for col in df if col.startswith("skeleton2d")]
    skeleton_2ds = df[filter_skeleton2d].to_numpy(dtype=np.float32).reshape(len(df),
                                                                            len(SKELETON_PEDREC_JOINTS), 5)
    return df, skeleton_2ds


def get_skeleton3d(df):
    filter_skeleton3d = [col for col in df if col.startswith("skeleton3d")]
    skeleton_3ds = df[filter_skeleton3d].to_numpy(dtype=np.float32).reshape(len(df),
                                                                            len(SKELETON_PEDREC_JOINTS), 6)
    return df, skeleton_3ds

def get_orientations(df):
    filter_body = [col for col in df if col.startswith("body_orientation_")]
    orientation_body = df[filter_body].to_numpy(dtype=np.float32).reshape(len(df), 5)
    filter_head = [col for col in df if col.startswith("head_orientation_")]
    orientation_head = df[filter_head].to_numpy(dtype=np.float32).reshape(len(df), 5)
    return orientation_body, orientation_head


def get_msjpe(output, target):
    mask = target[:, :, 4] == 1
    a = output[mask][:, :3]
    b = target[mask][:, :3]
    num_visible_joints = b.shape[0]
    if num_visible_joints < 1:
        print("NOOO")
        return 0
    x = np.linalg.norm(a[:, 0:3] - b[:, 0:3], axis=1)
    return np.mean(x)


def get_results(experiment_name: str, flip_test: bool = False, skeleton: SKELETON = SKELETON.PEDREC):
    df_full_gt = read_pedrec_df(os.path.join(sim_c01_root, sim_c01_val_filename))
    skeleton2d_visibles = [col for col in df_full_gt if col.startswith('skeleton2d') and col.endswith('_visible')]
    df_full_gt["visible_joints"] = df_full_gt[skeleton2d_visibles].sum(axis=1)
    df_valid_filter = (df_full_gt['bb_score'] >= 1) & (df_full_gt['visible_joints'] >= 3)
    df_full_gt = df_full_gt[df_valid_filter]
    df_gt = get_df(sim_c01_root, f"C01F_gt_df_{experiment_name}.pkl")
    df = get_df(sim_c01_root, f"C01F_pred_df_{experiment_name}.pkl")
    df_flipped = get_df(sim_c01_root, f"C01F_pred_df_{experiment_name}_flipped.pkl") if flip_test else None
    # df_flipped = get_df("data/datasets/Conti01", f"C01_pred_df_{experiment_name}_flipped.pkl")

    _, skeleton_3d_full_gt = get_skeleton3d(df_full_gt)
    _, skeleton_3d_gt = get_skeleton3d(df_gt)
    dist_3d = np.abs(skeleton_3d_full_gt[:, :, :3]-skeleton_3d_gt[:, :, :3])
    assert dist_3d.max() <= 0.001  # ignore floating point errors, smaller than 0.001mm == ok
    _, skeleton_3d = get_skeleton3d(df)
    assert skeleton_3d_gt.shape[0] == skeleton_3d.shape[0]
    if flip_test:
        df_flipped_view, skeleton_3d_flipped = get_skeleton3d(df_flipped)
        for i in range(0, skeleton_3d_flipped.shape[0]):
            skeleton_3d_flipped[i, :, :3] = flip_lr_joints_3d(skeleton_3d_flipped[i, :, :3])

        skeleton_3d += skeleton_3d_flipped
        skeleton_3d /= 2

    #
    # skeleton_3d_h36m_gt = np.zeros((skeleton_3d.shape[0], 17, 6), dtype=np.float32)
    # skeleton_3d_h36m = np.zeros((skeleton_3d.shape[0], 17, 6), dtype=np.float32)
    # for idx, joint in enumerate(SKELETON_H36M_JOINTS):
    #     skeleton_3d_h36m[:, idx, :] = skeleton_3d[:, joint.value, :]
    #     skeleton_3d_h36m_gt[:, idx, :] = skeleton_3d_gt[:, joint.value, :]

    msjpe = get_msjpe(skeleton_3d, skeleton_3d_gt)

    _, skeleton_2d_gt = get_skeleton2d(df_gt)
    _, skeleton_2d_pred = get_skeleton2d(df)
    if flip_test:
        _, skeleton_2d_pred_flipped = get_skeleton2d(df_flipped)
        for i in range(0, skeleton_2d_pred_flipped.shape[0]):
            skeleton_2d_pred_flipped[i, :, :2] = flip_lr_joints(skeleton_2d_pred_flipped[i, :, :2], 1980)
        skeleton_2d_pred += skeleton_2d_pred_flipped
        skeleton_2d_pred /= 2
    if skeleton == SKELETON.COCO:
        skeleton_2d_gt = get_skel_coco(skeleton_2d_gt)
        skeleton_2d_pred = get_skel_coco(skeleton_2d_pred)
    elif skeleton == SKELETON.H36M:
        skeleton_2d_gt = get_skel_h36m(skeleton_2d_gt)
        skeleton_2d_pred = get_skel_h36m(skeleton_2d_pred)
    elif skeleton == SKELETON.H36M_HANDFOOTENDS:
        skeleton_2d_gt = get_skel_h36m_handfootends(skeleton_2d_gt)
        skeleton_2d_pred = get_skel_h36m_handfootends(skeleton_2d_pred)
    pck_results = get_2d_pose_pck_results(skeleton_2d_gt, skeleton_2d_pred)

    o_body_gt, o_head_gt = get_orientations(df_gt)
    o_body_pred, o_head_pred = get_orientations(df)
    # Hack, forgot to add phi vis / supported, always 1
    temp = np.ones(o_body_gt.shape[0])
    temp = np.array([temp, temp]).transpose(1, 0)
    o_body_gt = np.concatenate((o_body_gt, temp), axis=1)
    o_head_gt = np.concatenate((o_head_gt, temp), axis=1)
    o_body_gt[:, 0] *= math.pi
    o_body_gt[:, 1] *= 2 * math.pi
    o_head_gt[:, 0] *= math.pi
    o_head_gt[:, 1] *= 2 * math.pi
    o_body_pred[:, 0] *= math.pi
    o_body_pred[:, 1] *= 2 * math.pi
    o_head_pred[:, 0] *= math.pi
    o_head_pred[:, 1] *= 2 * math.pi

    if flip_test:
        o_body_pred_flipped, o_head_pred_flipped = get_skeleton2d(df_flipped)
        for i in range(0, o_body_pred_flipped.shape[0]):
            o_body_pred_flipped[i, :, :2] = flip_lr_orientation(o_body_pred_flipped[i, :, :2])
            o_head_pred_flipped[i, :, :2] = flip_lr_orientation(o_head_pred_flipped[i, :, :2])
        o_body_pred += o_body_pred_flipped
        o_body_pred /= 2
        o_head_pred += o_head_pred_flipped
        o_head_pred /= 2
    dist_phi_body, dist_theta_body, spherical_distance_body = get_angular_distances(o_body_gt, o_body_pred)
    dist_phi_body = np.degrees(dist_phi_body)
    dist_theta_body = np.degrees(dist_theta_body)
    spherical_distance_body = np.degrees(spherical_distance_body)
    o_body_results = FullOrientationValidationResult(
        angle_error_theta_5=len(np.where(dist_theta_body <= 5)[0]) / dist_theta_body.shape[0],
        angle_error_theta_15=len(np.where(dist_theta_body <= 15)[0]) / dist_theta_body.shape[0],
        angle_error_theta_22_5=len(np.where(dist_theta_body <= 22.5)[0]) / dist_theta_body.shape[0],
        angle_error_theta_30=len(np.where(dist_theta_body <= 30)[0]) / dist_theta_body.shape[0],
        angle_error_theta_45=len(np.where(dist_theta_body <= 45)[0]) / dist_theta_body.shape[0],
        angle_error_theta_mean=np.mean(dist_theta_body),
        angle_error_theta_std=np.std(dist_theta_body),
        angle_error_phi_5=len(np.where(dist_phi_body <= 5)[0]) / dist_phi_body.shape[0],
        angle_error_phi_15=len(np.where(dist_phi_body <= 15)[0]) / dist_phi_body.shape[0],
        angle_error_phi_22_5=len(np.where(dist_phi_body <= 22.5)[0]) / dist_phi_body.shape[0],
        angle_error_phi_30=len(np.where(dist_phi_body <= 30)[0]) / dist_phi_body.shape[0],
        angle_error_phi_45=len(np.where(dist_phi_body <= 45)[0]) / dist_phi_body.shape[0],
        angle_error_phi_mean=np.mean(dist_phi_body),
        angle_error_phi_std=np.std(dist_phi_body),
        spherical_distance_mean=spherical_distance_body
    )
    
    dist_phi_head, dist_theta_head, spherical_distance_head = get_angular_distances(o_head_gt, o_head_pred)
    dist_phi_head = np.degrees(dist_phi_head)
    dist_theta_head = np.degrees(dist_theta_head)
    spherical_distance_head = np.degrees(spherical_distance_head)
    o_head_results = FullOrientationValidationResult(
        angle_error_theta_5=len(np.where(dist_theta_head <= 5)[0]) / dist_theta_head.shape[0],
        angle_error_theta_15=len(np.where(dist_theta_head <= 15)[0]) / dist_theta_head.shape[0],
        angle_error_theta_22_5=len(np.where(dist_theta_head <= 22.5)[0]) / dist_theta_head.shape[0],
        angle_error_theta_30=len(np.where(dist_theta_head <= 30)[0]) / dist_theta_head.shape[0],
        angle_error_theta_45=len(np.where(dist_theta_head <= 45)[0]) / dist_theta_head.shape[0],
        angle_error_theta_mean=np.mean(dist_theta_head),
        angle_error_theta_std=np.std(dist_theta_head),
        angle_error_phi_5=len(np.where(dist_phi_head <= 5)[0]) / dist_phi_head.shape[0],
        angle_error_phi_15=len(np.where(dist_phi_head <= 15)[0]) / dist_phi_head.shape[0],
        angle_error_phi_22_5=len(np.where(dist_phi_head <= 22.5)[0]) / dist_phi_head.shape[0],
        angle_error_phi_30=len(np.where(dist_phi_head <= 30)[0]) / dist_phi_head.shape[0],
        angle_error_phi_45=len(np.where(dist_phi_head <= 45)[0]) / dist_phi_head.shape[0],
        angle_error_phi_mean=np.mean(dist_phi_head),
        angle_error_phi_std=np.std(dist_phi_head),
        spherical_distance_mean=spherical_distance_head
    )

    return msjpe, pck_results, o_body_results, o_head_results


DEFAULT_EXPERIMENTS = ["p2d3d_c_o_h36m_sim_mebow"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SIM-C01 MSJPE / PCK / orientation tables from the result dataframes "
                                                 "written by tools/resultwriters/sim_c01_to_results.py.")
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS, choices=sorted(STAGES.keys()),
                        metavar="STAGE", help="Training stages to evaluate.")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    return parser.parse_args(argv)


def main(argv=None):
    global sim_c01_root, sim_c01_val_filename
    args = parse_args(argv)
    experiment_paths = get_experiment_paths(args.data_dir)
    sim_c01_root = experiment_paths.sim_c01_val_dir
    sim_c01_val_filename = experiment_paths.sim_c01_val_filename
    experiments = [experiment_paths.get_stage_checkpoint_path(name) for name in args.experiments]

    latex_overview = [
        "Experiment & Avg \\\\\\midrule"]

    pck_overview = [
        "| Experiment | PCK@0.05 | PCK@0.2 |"]
    pck_overview += ["| ---  | --- | --- |"]

    msjpe_overview = [
        "| Experiment | MSJPE (Mean) |"]
    msjpe_overview += ["| ---  | --- |"]
    
    o_body_overview = [
        " | Experiment | O@5 | O@15 | O@22.5 | O@30 | O@45 | O (mean) | O (std)"]
    o_body_overview += ["| --- | --- | --- | --- | --- | --- | --- | --- |"]
    
    o_head_overview = [
        " | Experiment | O@5 | O@15 | O@22.5 | O@30 | O@45 | O (mean) | O (std)"]
    o_head_overview += ["| --- | --- | --- | --- | --- | --- | --- | --- |"]

    o_head_theta_overview = [
        " | Experiment | O@5 | O@15 | O@22.5 | O@30 | O@45 | O (mean) | O (std)"]
    o_head_theta_overview += ["| --- | --- | --- | --- | --- | --- | --- | --- |"]
    
    for experiment in experiments:
        experiment_name = Path(experiment).stem
        print(f"## {experiment_name}")
        msjpe, pck_results, o_body_results, o_head_results = get_results(experiment_name, flip_test=False)
        o_body_results: FullOrientationValidationResult = o_body_results
        latex_overview += [f"{experiment_name} & {msjpe:2f} \\\\"]
        pck_overview += [f"| {experiment_name} | {pck_results.pck_05_mean:.2f} | {pck_results.pck_2_mean:.2f} |"]
        msjpe_overview += [f"| {experiment_name} | {msjpe:.2f} |"]
        print("Body Phi")
        o_body_overview += [f"| {experiment_name}"
                            f" | {o_body_results.angle_error_phi_5 * 100:.2f}"
                            f" | {o_body_results.angle_error_phi_15 * 100:.2f}"
                            f" | {o_body_results.angle_error_phi_22_5 * 100:.2f}"
                            f" | {o_body_results.angle_error_phi_30 * 100:.2f}"
                            f" | {o_body_results.angle_error_phi_45 * 100:.2f}"
                            f" | {o_body_results.angle_error_phi_mean:.2f}"
                            f" | {o_body_results.angle_error_phi_std:.2f} |"]
        print("Head Phi")
        o_head_overview += [f"| {experiment_name}"
                            f" | {o_head_results.angle_error_phi_5 * 100:.2f}"
                            f" | {o_head_results.angle_error_phi_15 * 100:.2f}"
                            f" | {o_head_results.angle_error_phi_22_5 * 100:.2f}"
                            f" | {o_head_results.angle_error_phi_30 * 100:.2f}"
                            f" | {o_head_results.angle_error_phi_45 * 100:.2f}"
                            f" | {o_head_results.angle_error_phi_mean:.2f}"
                            f" | {o_head_results.angle_error_phi_std:.2f} |"]
        
        print("Head Theta")
        o_head_theta_overview += [f"| {experiment_name}"
                            f" | {o_head_results.angle_error_theta_5 * 100:.2f}"
                            f" | {o_head_results.angle_error_theta_15 * 100:.2f}"
                            f" | {o_head_results.angle_error_theta_22_5 * 100:.2f}"
                            f" | {o_head_results.angle_error_theta_30 * 100:.2f}"
                            f" | {o_head_results.angle_error_theta_45 * 100:.2f}"
                            f" | {o_head_results.angle_error_theta_mean:.2f}"
                            f" | {o_head_results.angle_error_theta_std:.2f} |"]

    # TODO: Orientation results
    for text in latex_overview:
        print(text)
        
    for text in pck_overview:
        print(text)

    for text in msjpe_overview:
        print(text)

    for text in o_body_overview:
        print(text)

    for text in o_head_overview:
        print(text)

    for text in o_head_theta_overview:
        print(text)


if __name__ == "__main__":
    main()
