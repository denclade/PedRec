import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import os
from pathlib import Path

from pedrec.evaluations.eval_helper import get_skel_coco, get_skel_h36m, get_skel_h36m_handfootends
from pedrec.evaluations.validate import get_2d_pose_pck_results, get_3d_pose_results
from pedrec.models.constants.skeleton_coco import SKELETON_COCO_JOINTS
from pedrec.models.constants.skeleton_h36m import SKELETON_H36M_JOINTS, SKELETON_H36M_HANDFOOTENDS_JOINTS
from pedrec.models.constants.skeletons import SKELETON
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.pedrec_stages import STAGES
from pedrec.utils.skeleton_helper import flip_lr_joints
from pedrec.utils.skeleton_helper_3d import flip_lr_joints_3d

import numpy as np
import pandas as pd
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS

h36m_actions = [
    r"Directions(?: .)*\.",
    r"Discussion(?: .)*\.",
    r"Eating(?: .)*\.",
    r"Greeting(?: .)*\.",
    r"Phoning(?: .)*\.",
    r"Photo(?: .)*\.",
    r"Posing(?: .)*\.",
    r"Purchases(?: .)*\.",
    r"Sitting(?: .)*\.",
    r"SittingDown(?: .)*\.",
    r"Smoking(?: .)*\.",
    r"Waiting(?: .)*\.",
    r"WalkDog(?: .)*\.|WalkingDog(?: .)*\.",
    r"Walking(?: .)*\.",
    r"WalkTogether(?: .)*\."
]


h36m_root = "data/datasets/Human3.6m"  # set by main()


def get_df(dataset_root, result_filename):
    dataset_df_path = os.path.join(dataset_root, "results", result_filename)
    # TODO EVERY 64th
    df = pd.read_pickle(dataset_df_path)
    # df = df.loc[range(0, len(df), 64)]
    return df


def get_skeleton2d(df, action_filter):
    df_view = df
    if action_filter is not None:
        filter_action = (df['img_path'].str.contains(action_filter))
        df_view = df.loc[filter_action]
    filter_skeleton2d = [col for col in df_view if col.startswith("skeleton2d")]
    skeleton_2ds = df_view[filter_skeleton2d].to_numpy(dtype=np.float32).reshape(len(df_view),
                                                                                 len(SKELETON_PEDREC_JOINTS), 5)


    return df_view, skeleton_2ds


def get_skeleton3d(df, action_filter):
    df_view = df
    if action_filter is not None:
        filter_action = (df['img_path'].str.contains(action_filter))
        df_view = df.loc[filter_action]
    filter_skeleton3d = [col for col in df_view if col.startswith("skeleton3d")]
    skeleton_3ds = df_view[filter_skeleton3d].to_numpy(dtype=np.float32).reshape(len(df_view),
                                                                                 len(SKELETON_PEDREC_JOINTS), 6)
    return df_view, skeleton_3ds


def get_msjpe(output, target):
    a = output[:, :, :3]
    b = target[:, :, :3]
    x = np.linalg.norm(a[:, :, 0:3] - b[:, :, 0:3], axis=2)
    return np.mean(x)


# def get_msjpe(output, target):
#     a = output[:, :3]
#     b = target[:, :3]
#     test = a[:, :, 0:3] - b[:, :, 0:3]
#     x = np.linalg.norm(test, axis=2)
#     y = np.linalg.norm(a[:, :, 0:3] - b[:, :, 0:3], axis=2)
#     return np.mean(x)

def get_results(experiment_name: str, flip_test: bool = False, skeleton: SKELETON = SKELETON.H36M):
    df_gt = get_df(h36m_root, f"H36M_gt_df_{experiment_name}.pkl")
    df = get_df(h36m_root, f"H36M_pred_df_{experiment_name}.pkl")
    df_flipped = get_df(h36m_root, f"H36M_pred_df_{experiment_name}_flipped.pkl") if flip_test else None
    msjpes = []
    used_idx = []
    for action_filter in h36m_actions:
        df_gt_view, skeleton_3d_gt = get_skeleton3d(df_gt, action_filter)
        used_idx += df_gt_view["img_path"].to_list()
        df_view, skeleton_3d = get_skeleton3d(df, action_filter)
        assert skeleton_3d_gt.shape[0] == skeleton_3d.shape[0]
        if flip_test:
            df_flipped_view, skeleton_3d_flipped = get_skeleton3d(df_flipped, action_filter)
            for i in range(0, skeleton_3d_flipped.shape[0]):
                skeleton_3d_flipped[i, :, :3] = flip_lr_joints_3d(skeleton_3d_flipped[i, :, :3])

            skeleton_3d += skeleton_3d_flipped
            skeleton_3d /= 2

        if skeleton == SKELETON.H36M_HANDFOOTENDS:
            skeleton_3d_h36m_gt = np.zeros((skeleton_3d.shape[0], 21, 6), dtype=np.float32)
            skeleton_3d_h36m = np.zeros((skeleton_3d.shape[0], 21, 6), dtype=np.float32)
            for idx, joint in enumerate(SKELETON_H36M_HANDFOOTENDS_JOINTS):
                skeleton_3d_h36m[:, idx, :] = skeleton_3d[:, joint.value, :]
                skeleton_3d_h36m_gt[:, idx, :] = skeleton_3d_gt[:, joint.value, :]

            msjpe = get_msjpe(skeleton_3d_h36m, skeleton_3d_h36m_gt)
            msjpes.append((action_filter, msjpe))
        else:
            skeleton_3d_h36m_gt = np.zeros((skeleton_3d.shape[0], 17, 6), dtype=np.float32)
            skeleton_3d_h36m = np.zeros((skeleton_3d.shape[0], 17, 6), dtype=np.float32)
            for idx, joint in enumerate(SKELETON_H36M_JOINTS):
                skeleton_3d_h36m[:, idx, :] = skeleton_3d[:, joint.value, :]
                skeleton_3d_h36m_gt[:, idx, :] = skeleton_3d_gt[:, joint.value, :]

            msjpe = get_msjpe(skeleton_3d_h36m, skeleton_3d_h36m_gt)
            msjpes.append((action_filter, msjpe))

    # for x in df["img_path"]:
    #     if x not in used_idx:
    #         a = 1
    _, skeleton_2d_gt = get_skeleton2d(df_gt, None)
    _, skeleton_2d_pred = get_skeleton2d(df, None)
    if flip_test:
        _, skeleton_2d_pred_flipped = get_skeleton2d(df_flipped, None)
        for i in range(0, skeleton_2d_pred_flipped.shape[0]):
            skeleton_2d_pred_flipped[i, :, :2] = flip_lr_joints(skeleton_2d_pred_flipped[i, :, :2], 1000)
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

    _, skeleton_3d_gt = get_skeleton3d(df_gt, None)
    _, skeleton_3d_pred = get_skeleton3d(df, None)
    if flip_test:
        _, skeleton_3d_flipped = get_skeleton3d(df_flipped, None)
        for i in range(0, skeleton_3d_flipped.shape[0]):
            skeleton_3d_flipped[i, :, :3] = flip_lr_joints_3d(skeleton_3d_flipped[i, :, :3])
        skeleton_3d_pred += skeleton_3d_flipped
        skeleton_3d_pred /= 2
    msjpe_results = get_3d_pose_results(skeleton_3d_gt, skeleton_3d_pred)
    return msjpes, pck_results, msjpe_results


DEFAULT_EXPERIMENTS = ["p2d3d_c_o_h36m_sim_mebow"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Human3.6m per-action MPJPE / PCK tables from the result dataframes "
                                                 "written by tools/resultwriters/h36m_to_results.py.")
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS, choices=sorted(STAGES.keys()),
                        metavar="STAGE", help="Training stages to evaluate.")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--flip-test", action="store_true", help="Average with the flipped results.")
    parser.add_argument("--skeleton", choices=[s.name for s in SKELETON], default=SKELETON.H36M.name)
    return parser.parse_args(argv)


def main(argv=None):
    global h36m_root
    args = parse_args(argv)
    experiment_paths = get_experiment_paths(args.data_dir)
    h36m_root = os.path.dirname(os.path.normpath(experiment_paths.h36m_val_dir))
    experiments = [experiment_paths.get_stage_checkpoint_path(name) for name in args.experiments]

    result_overview = [
        "| Experiment | Dir. |  Disc. | Eat | Greet | Phone |  Photo |  Pose | Purch. |  Sit |  SitD. | Smoke | Wait | WalkD. | Walk | WalkT. | Avg |"]
    result_overview += ["| ---  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    
    latex_overview = [
        "Experiment & Dir. &  Disc. & Eat & Greet & Phone &  Photo &  Pose & Purch. &  Sit &  SitD. & Smoke & Wait & WalkD. & Walk & WalkT. & Avg \\\\\\midrule"]


    pck_overview = [
        "| Experiment | PCK@0.05 | PCK@0.2 |"]
    pck_overview += ["| ---  | --- | --- |"]

    msjpe_overview = [
        "| Experiment | MSJPE (Mean) |"]
    msjpe_overview += ["| ---  | --- |"]
    for experiment in experiments:
        experiment_name = Path(experiment).stem
        print(f"## {experiment_name}")
        msjpes, pck_results, msjpe_results = get_results(experiment_name, flip_test=args.flip_test,
                                                          skeleton=SKELETON[args.skeleton])
        msjpe_mean = 0
        for msjpe in msjpes:
            print(f"{msjpe[0]}: {msjpe[1]}")
            msjpe_mean += msjpe[1]
        msjpe_mean /= len(msjpes)
        msjpes = [f"{msjpe[1]:.1f}" for msjpe in msjpes]
        result_overview += [f"| {experiment_name} | {' | '.join(msjpes)} | {msjpe_mean} |"]
        print(f"Mean: {msjpe_mean:1f}")
        msjpes = [f"${msjpe}$" for msjpe in msjpes]
        latex_overview += [f"{experiment_name} | {' & '.join(msjpes)} & {msjpe_mean} \\\\"]
        pck_overview += [f"| {experiment_name} | {pck_results.pck_05_mean:.2f} | {pck_results.pck_2_mean:.2f} |"]
        msjpe_overview += [f"| {experiment_name} | {msjpe_results.mpjpe_mean:.2f} |"]

    for text in result_overview:
        print(text)

    for text in latex_overview:
        print(text)
        
    for text in pck_overview:
        print(text)

    for text in msjpe_overview:
        print(text)


if __name__ == "__main__":
    main()
