"""
Runs PedRecNet (via the inference pipeline: RT-DETR -> PedRecNet -> ByteTrack -> One Euro) on the EHPI video dataset
and writes the skeleton dataframe used by the ``*_ehpi2dvids`` action recognition variants. Per frame the most
consistently tracked person is stored (same column layout as the published dataframe).

    python pedrec/ehpi_dataset_vids_to_dataframe.py [--src-dir data/videos/ehpi_videos] [--weights ...] [--output ...]
"""
import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from pedrec.configs.app_config import AppConfig
from pedrec.inference.pipeline import PedRecPipeline, PipelineConfig
from pedrec.models.constants.action_mappings import ACTION
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.utils.input_providers.img_dir_provider import ImgDirProvider
from pedrec.utils.log_helper import configure_logger
from pedrec.utils.torch_utils.torch_helper import get_device


def set_df_dtypes(df: pd.DataFrame):
    df["path"] = df["path"].astype("category")
    df["frame"] = df["frame"].astype("int")
    df["uid"] = df["uid"].astype("int")
    df["action"] = df["action"].astype("category")

    for joint in SKELETON_PEDREC_JOINTS:
        for dim in ("2d", "3d"):
            coords = ("x", "y") if dim == "2d" else ("x", "y", "z")
            for coord in coords + ("score",):
                df[f"skeleton{dim}_{joint.name}_{coord}"] = df[f"skeleton{dim}_{joint.name}_{coord}"].astype("float32")
            for flag in ("visible", "supported"):
                df[f"skeleton{dim}_{joint.name}_{flag}"] = df[f"skeleton{dim}_{joint.name}_{flag}"].astype("category")

    for part in ("body", "head"):
        for value in ("phi", "theta", "score"):
            df[f"{part}_orientation_{value}"] = df[f"{part}_orientation_{value}"].astype("float32")
        df[f"{part}_orientation_visible"] = df[f"{part}_orientation_visible"].astype("category")

def get_column_names():
    column_names = [
        "path",
        "frame",
        "uid",
        "action"
    ]
    for joint in SKELETON_PEDREC_JOINTS:
        column_names.append(f"skeleton2d_{joint.name}_x")
        column_names.append(f"skeleton2d_{joint.name}_y")
        column_names.append(f"skeleton2d_{joint.name}_score")
        column_names.append(f"skeleton2d_{joint.name}_visible")
        column_names.append(f"skeleton2d_{joint.name}_supported")
    for joint in SKELETON_PEDREC_JOINTS:
        column_names.append(f"skeleton3d_{joint.name}_x")
        column_names.append(f"skeleton3d_{joint.name}_y")
        column_names.append(f"skeleton3d_{joint.name}_z")
        column_names.append(f"skeleton3d_{joint.name}_score")
        column_names.append(f"skeleton3d_{joint.name}_visible")
        column_names.append(f"skeleton3d_{joint.name}_supported")

    column_names.append("body_orientation_phi")
    column_names.append("body_orientation_theta")
    column_names.append("body_orientation_score")
    column_names.append("body_orientation_visible")

    column_names.append("head_orientation_phi")
    column_names.append("head_orientation_theta")
    column_names.append("head_orientation_score")
    column_names.append("head_orientation_visible")

    return column_names


def get_dummy_human() -> Human:
    return Human(bb=[0, 0, 0, 0, 0, 0],
                 skeleton_2d=np.zeros((len(SKELETON_PEDREC_JOINTS), 3), dtype=np.float32),
                 skeleton_3d=np.zeros((len(SKELETON_PEDREC_JOINTS), 4), dtype=np.float32),
                 orientation=np.zeros((2, 2), dtype=np.float32))


def select_human(humans: List[Human], last_uid: Optional[int]) -> Optional[Human]:
    """The person followed in the previous frame if still tracked, otherwise the one with the highest score."""
    for human in humans:
        if human.uid == last_uid:
            return human
    return max(humans, key=lambda h: h.score, default=None)


def video_to_rows(pipeline: PedRecPipeline, img_dir: str, rel_path: str, action_label: ACTION) -> list:
    rows = []
    last_uid = None
    for frame_nr, img in enumerate(ImgDirProvider(img_dir, image_size=pipeline.img_size).get_data()):
        result = pipeline.process(frame_nr, img)
        human = select_human(result.humans, last_uid)
        orientation_vis_supp = [1, 1]
        if human is None:
            human = get_dummy_human()
            orientation_vis_supp = [0, 0]
        else:
            last_uid = human.uid
        visibles = (human.skeleton_2d[:, 2] > 0.5).astype(np.int32)
        visible_supported = np.stack([visibles, np.ones_like(visibles)], axis=1)
        pose2d = np.concatenate((human.skeleton_2d, visible_supported), axis=1).reshape(-1).tolist()
        pose3d = np.concatenate((human.skeleton_3d, visible_supported), axis=1).reshape(-1).tolist()
        rows.append([rel_path, frame_nr, human.uid, action_label.value] + pose2d + pose3d
                    + human.orientation[0].reshape(-1).tolist() + orientation_vis_supp
                    + human.orientation[1].reshape(-1).tolist() + orientation_vis_supp)
    return rows


def get_action_from_str(action_str: str):
    if "wave" == action_str:
        return ACTION.WAVE_CAR_OUT
    elif "walk" == action_str:
        return ACTION.WALK
    elif "idle" == action_str:
        return ACTION.IDLE
    elif "sit" == action_str:
        return ACTION.SIT
    elif "jump" == action_str:
        return ACTION.JUMP

    raise ValueError(f"Action '{action_str}' not found.")


def get_dataset_data(src_dir):
    dataset_data = []
    for folder_name in os.listdir(src_dir):
        if folder_name == "2019_ITS_Journal_Eval2":
            continue
        full_path = os.path.join(src_dir, folder_name)
        if not os.path.isdir(full_path):
            continue
        if folder_name == src_dir:
            continue
        is_sim = False
        action_label = None
        if folder_name[0:3] == "SIM":
            is_sim = True
            if "wave" in folder_name:
                action_label = ACTION.WAVE_CAR_OUT
            elif "walk" in folder_name:
                action_label = ACTION.WALK
            elif "idle" in folder_name:
                action_label = ACTION.IDLE
            elif "sit" in folder_name:
                action_label = ACTION.SIT
            elif "jump" in folder_name:
                action_label = ACTION.JUMP
            else:
                continue
        img_dir = os.path.join(full_path, "imgs")
        for vid_folder in os.listdir(img_dir):
            vid_folder_path = os.path.join(img_dir, vid_folder)
            if not is_sim:
                action_label = get_action_from_str(vid_folder.split('_')[0])
            assert action_label is not None
            dataset_data.append((vid_folder_path, action_label))
    return dataset_data


def main(argv=None):
    configure_logger()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-dir", default="data/videos/ehpi_videos")
    parser.add_argument("--weights", default=None, help="PedRecNet weights (default: the demo weights)")
    parser.add_argument("--output", default=None, help="Default: <src-dir>/pedrec_v2_results.pkl")
    parser.add_argument("--size", default="1280x720", help="Processing size WIDTHxHEIGHT")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    app_cfg = AppConfig()
    width, height = args.size.lower().split("x")
    app_cfg.inference.img_size = ImageSize(int(width), int(height))
    device = get_device(not args.cpu)
    cfg = PipelineConfig(use_action=False, pedrec_weights=args.weights, human_min_score=0.65)

    result_rows = []
    for img_dir, action_label in get_dataset_data(args.src_dir):
        print(f"Working on: {img_dir} ({action_label.name})")
        pipeline = PedRecPipeline(cfg, app_cfg, device)  # fresh tracker per video
        result_rows += video_to_rows(pipeline, img_dir, os.path.relpath(img_dir, args.src_dir), action_label)

    df = pd.DataFrame(data=result_rows, columns=get_column_names())
    set_df_dtypes(df)
    output = args.output or os.path.join(args.src_dir, "pedrec_v2_results.pkl")
    df.to_pickle(output)
    print(f"Wrote {len(df)} rows to {output}")


if __name__ == '__main__':
    main()
