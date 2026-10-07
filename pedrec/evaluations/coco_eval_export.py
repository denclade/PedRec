import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import math
import os
from pathlib import Path

from pedrec.evaluations.eval_np.eval_angular import get_angular_distances
from pedrec.evaluations.validate import get_2d_pose_pck_results
from pedrec.models.validation.orientation_validation_results import FullOrientationValidationResult
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.pedrec_stages import STAGES
from pedrec.utils.skeleton_helper import flip_lr_joints

import numpy as np
import pandas as pd
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.models.constants.skeleton_coco import SKELETON_COCO_JOINTS


dataset_root = "data/datasets/COCO"  # set by main()


def get_skeleton2d(result_filename):
    dataset_df_path = os.path.join(dataset_root, "results", result_filename)
    df = pd.read_pickle(dataset_df_path)

    filter_skeleton2d = [col for col in df if col.startswith("skeleton2d")]
    img_widths = df["img_size_w"]
    skeleton_2ds = df[filter_skeleton2d].to_numpy(dtype=np.float32).reshape(len(df), len(SKELETON_PEDREC_JOINTS), 5)
    return df, skeleton_2ds, img_widths

def get_orientations(df):
    filter_body = [col for col in df if col.startswith("body_orientation_")]
    orientation_body = df[filter_body].to_numpy(dtype=np.float32).reshape(len(df), 5)
    return orientation_body

def flip_lr_orientation(orientation: np.ndarray) -> np.ndarray:
    orientation[1] = np.mod(math.pi - orientation[1], 2 * math.pi)
    return orientation

def get_orientation_results(df_gt, df, df_flipped, flip_test):
    o_body_gt = get_orientations(df_gt)
    o_body_pred = get_orientations(df)
    # Hack, forgot to add phi vis
    # temp = np.array(o_body_gt[0:1, 2:4])
    # o_body_gt = np.concatenate((o_body_gt, temp), axis=1)
    o_body_gt[:, 0] *= math.pi
    o_body_gt[:, 1] *= 2 * math.pi
    o_body_pred[:, 0] *= math.pi
    o_body_pred[:, 1] *= 2 * math.pi

    if flip_test:
        o_body_pred_flipped = get_orientations(df_flipped)
        o_body_pred_flipped[:, 0] *= math.pi
        o_body_pred_flipped[:, 1] *= 2 * math.pi
        for i in range(0, o_body_pred_flipped.shape[0]):
            o_body_pred_flipped[i, :2] = flip_lr_orientation(o_body_pred_flipped[i, :2])
        o_body_pred += o_body_pred_flipped
        o_body_pred /= 2
    dist_phi_body, dist_theta_body, spherical_distance_body = get_angular_distances(o_body_gt, o_body_pred)
    dist_phi_body = np.degrees(dist_phi_body)
    dist_theta_body = np.degrees(dist_theta_body)
    spherical_distance_body = np.degrees(spherical_distance_body)
    thetas = dist_theta_body.shape[0]
    phis = dist_phi_body.shape[0]
    return FullOrientationValidationResult(
        angle_error_theta_5=len(np.where(dist_theta_body <= 5)[0]) / thetas if thetas > 0 else -1,
        angle_error_theta_15=len(np.where(dist_theta_body <= 15)[0]) / thetas if thetas > 0 else -1,
        angle_error_theta_22_5=len(np.where(dist_theta_body <= 22.5)[0]) / thetas if thetas > 0 else -1,
        angle_error_theta_30=len(np.where(dist_theta_body <= 30)[0]) / thetas if thetas > 0 else -1,
        angle_error_theta_45=len(np.where(dist_theta_body <= 45)[0]) / thetas if thetas > 0 else -1,
        angle_error_theta_mean=np.mean(dist_theta_body) if thetas > 0 else -1,
        angle_error_theta_std=np.std(dist_theta_body) if thetas > 0 else -1,
        angle_error_phi_5=len(np.where(dist_phi_body <= 5)[0]) / phis if phis > 0 else -1,
        angle_error_phi_15=len(np.where(dist_phi_body <= 15)[0]) / phis if phis > 0 else -1,
        angle_error_phi_22_5=len(np.where(dist_phi_body <= 22.5)[0]) / phis if phis > 0 else -1,
        angle_error_phi_30=len(np.where(dist_phi_body <= 30)[0]) / phis if phis > 0 else -1,
        angle_error_phi_45=len(np.where(dist_phi_body <= 45)[0]) / phis if phis > 0 else -1,
        angle_error_phi_mean=np.mean(dist_phi_body) if phis > 0 else -1,
        angle_error_phi_std=np.std(dist_phi_body) if phis > 0 else -1,
        spherical_distance_mean=spherical_distance_body
    )


def get_results(experiment_name: str, flip_test: bool = False):
    df_gt, skeleton_2d_gt, _ = get_skeleton2d(f"COCO_gt_df_{experiment_name}.pkl")
    df, skeleton_2d, _ = get_skeleton2d(f"COCO_pred_df_{experiment_name}.pkl")
    df_flipped = None
    if flip_test:
        df_flipped, skeleton_2d_flipped, img_widths = get_skeleton2d(f"COCO_pred_df_{experiment_name}_flipped.pkl")
        for i in range(0, skeleton_2d_flipped.shape[0]):
            skeleton_2d_flipped[i, :, :] = flip_lr_joints(skeleton_2d_flipped[i, :, :], img_widths[i])

        skeleton_2d += skeleton_2d_flipped
        skeleton_2d /= 2
    coco_out = []
    coco_out_gt = []
    for joint in SKELETON_COCO_JOINTS:
        coco_out.append(skeleton_2d[:, joint.value, :])
        coco_out_gt.append(skeleton_2d_gt[:, joint.value, :])
    coco_out = np.transpose(np.array(coco_out, dtype=np.float32), (1, 0, 2))
    coco_out_gt = np.transpose(np.array(coco_out_gt, dtype=np.float32), (1, 0, 2))
    jsons = []
    for idx, row in df.iterrows():
        img_path = row["img_path"]
        img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        json_data = {}
        json_data["category_id"] = 1
        # json_data["center"] = [row["center_x"], row["center_y"]]
        json_data["image_id"] = img_id
        keypoint_list = []
        skeleton_coco = coco_out[idx, :, :]
        skeleton_coco_gt = coco_out_gt[idx, :, :]
        json_data["keypoints"] = skeleton_coco[:, :3].flatten().tolist()
        # json_data["scale"] = [row["scale_x"], row["scale_y"]]
        test = skeleton_coco[skeleton_coco_gt[:, 3] == 1]
        score = float(np.sum(test[:, 2]) / np.sum(skeleton_coco_gt[:, 3]))
        json_data["score"] = score
        jsons.append(json_data)

        a = 1

    pck_results = get_2d_pose_pck_results(skeleton_2d_gt, skeleton_2d)
    o_body_results = get_orientation_results(df_gt, df, df_flipped, flip_test)

    coco_gt_path = os.path.join(dataset_root, "annotations", "person_keypoints_val2017.json")
    result_file_path = os.path.join(dataset_root, "results", f"coco_out_{experiment_name}.json")
    with open(result_file_path, 'w') as outfile:
        json.dump(jsons, outfile, sort_keys=True, indent=4)

    cocoGt = COCO(coco_gt_path)
    cocoDt = cocoGt.loadRes(result_file_path)
    # cocoDt = cocoGt.loadRes("data/python/human-pose-estimation.pytorch/output/coco/pose_resnet_50/256x192_d256x3_adam_lr1e-3/results/keypoints_val2017_results.json")
    cocoEval = COCOeval(cocoGt, cocoDt, "keypoints")
    cocoEval.params.useSegm = None
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return list(cocoEval.stats), pck_results, o_body_results


DEFAULT_EXPERIMENTS = ["p2d_c", "p2d3d_c_h36m", "p2d3d_c_sim", "p2d3d_c_h36m_sim", "p2d3d_c_o_h36m_mebow", "p2d3d_c_o_sim",
                       "p2d3d_c_o_h36m_sim", "p2d3d_c_o_h36m_sim_mebow"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="COCO keypoint AP / PCK / orientation tables from the result "
                                                 "dataframes written by tools/resultwriters/coco_to_results.py.")
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS, choices=sorted(STAGES.keys()),
                        metavar="STAGE", help="Training stages to evaluate.")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--no-flip-test", action="store_true", help="Do not average with the flipped results.")
    return parser.parse_args(argv)


def main(argv=None):
    global dataset_root
    args = parse_args(argv)
    experiment_paths = get_experiment_paths(args.data_dir)
    dataset_root = experiment_paths.coco_dir
    experiments = [experiment_paths.get_stage_checkpoint_path(name) for name in args.experiments]
    flip_test = not args.no_flip_test

    result_overview = [
        "| Experiment | AP |  Ap .5 |  AP .75 |  AP (M) |  AP (L) |  AR |  AR .5 |  AR .75 |  AR (M) |  AR (L) |"]
    result_overview += ["| ---  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    
    latex_overview = [
        "Datasets & AP &  Ap .5 &  AP .75 &  AP (M) &  AP (L) &  AR &  AR .5 &  AR .75 &  AR (M) &  AR (L)\\\\ \\midrule"]

    pck_overview = [
        "| Experiment | PCK@0.05 | PCK@0.2 |"]
    pck_overview += ["| ---  | --- | --- |"]

    o_body_overview = [
        " | Experiment | O@5 | O@15 | O@22.5 | O@30 | O@45 | O (mean) | O (std)"]
    o_body_overview += ["| --- | --- | --- | --- | --- | --- | --- | --- |"]

    for experiment in experiments:
        experiment_name = Path(experiment).stem
        print(f"## {experiment_name}")
        results, pck_results, o_body_results = get_results(experiment_name, flip_test=flip_test)
        results = [f"{result * 100:.2f}" for result in results]
        result_overview += [f"| {experiment_name} | {' | '.join(results)} |"]
        results = [f"${result}$" for result in results]
        datasets = "C"
        if "h36m" in experiment_name:
            datasets += "+H"
        if "sim" in experiment_name:
            datasets += "+S"
        if "mebow" in experiment_name:
            datasets += "+M"
        latex_overview += [f"{datasets} & {' & '.join(results)} \\\\"]
        pck_overview += [f"| {experiment_name} | {pck_results.pck_05_mean:.2f} | {pck_results.pck_2_mean:.2f} |"]
        o_body_overview += [f"| {experiment_name}"
                            f" | {o_body_results.angle_error_phi_5 * 100:.2f}"
                            f" | {o_body_results.angle_error_phi_15 * 100:.2f}"
                            f" | {o_body_results.angle_error_phi_22_5 * 100:.2f}"
                            f" | {o_body_results.angle_error_phi_30 * 100:.2f}"
                            f" | {o_body_results.angle_error_phi_45 * 100:.2f}"
                            f" | {o_body_results.angle_error_phi_mean:.2f}"
                            f" | {o_body_results.angle_error_phi_std:.2f} |"]

    for text in result_overview:
        print(text)#

    for text in latex_overview:
        print(text)

    for text in pck_overview:
        print(text)

    for text in o_body_overview:
        print(text)


if __name__ == "__main__":
    main()
