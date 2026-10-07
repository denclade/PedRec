import math
from typing import Tuple

import numpy as np
import pandas as pd

from pedrec.configs.dataset_configs import PedRecDatasetConfig
from pedrec.models.constants.sample_method import SAMPLE_METHOD
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.models.data_structures import ImageSize
from pedrec.models.datasets.pedrec_dataset_entry_annotations import PedRecDatasetEntries
from pedrec.models.datasets.pedrec_dataset_info import PedRecDatasetInfo
from pedrec.utils.bb_helper import bbs_to_centers_scales
from pedrec.utils.pandas_helper import get_subsampled_df
from pedrec.utils.pedrec_dataset_helper import get_filter_skeleton2d, get_filter_body_orientation, \
    get_filter_skeleton3d, get_filter_bb, get_filter_head_orientation, get_filter_env


def set_scene_start_end(df: pd.DataFrame):
    scene_groups = df.groupby(["scene_id", "img_dir"])
    df['scene_start'] = 0
    df['scene_end'] = 0
    for name, group in scene_groups:
        indices = group.index
        scene_id = name[0]
        img_dir = name[1]
        mask = (df['scene_id'] == scene_id) & (df['img_dir'] == img_dir)
        df.loc[mask, 'scene_start'] = indices.min()
        df.loc[mask, 'scene_end'] = indices.max()
    a = 1

def get_filter_actions(df: pd.DataFrame, action_filter):
    filter_action = None
    if action_filter is not None:
        filter_action = (df["actions"].apply(lambda x: action_filter in x))
    return filter_action

def get_valid_filter(df: pd.DataFrame):
    skeleton2d_visibles = [col for col in df if col.startswith('skeleton2d') and col.endswith('_visible')]
    df["visible_joints"] = df[skeleton2d_visibles].sum(axis=1)
    return (df['bb_score'] >= 1) & (df['visible_joints'] >= 3)

def get_annotations_from_pedrec_df(df_path: str,
                                   cfg: PedRecDatasetConfig,
                                   input_size: ImageSize,
                                   result_df_path: str,
                                   action_filter: str = None,
                                   is_h36m: bool = False) -> Tuple[
    PedRecDatasetEntries, PedRecDatasetInfo]:
    df = get_subsampled_df(df_path, cfg)
    filters = get_valid_filter(df)
    if action_filter is not None:
        filters = filters & get_filter_actions(df, action_filter)

    if is_h36m:
        filter_s11 = ~((df['subject_id'] == "S11") & (df['img_dir'].str.contains("Directions\\.")))
        filters = filters & filter_s11
    df['original_index'] = range(0, len(df))
    df_only_valid_skels = df[filters]
    df_only_valid_skels.reset_index(drop=True, inplace=True)
    set_scene_start_end(df)
    # df2 = pd.read_pickle("data/datasets/ROMb/v5_pose_results.pkl")
    df_full_length = len(df)

    df_used_length = len(df_only_valid_skels)
    filter_skeleton2d = get_filter_skeleton2d(df)
    filter_skeleton3d = get_filter_skeleton3d(df)
    filter_bb = get_filter_bb(df)
    filter_body_orientation = get_filter_body_orientation(df)
    filter_head_orientation = get_filter_head_orientation(df)
    filter_env = get_filter_env(df)

    skeleton2ds_results = None
    skeleton3ds_results = None
    if result_df_path is not None:
        df_results = get_subsampled_df(result_df_path, cfg)
        # df_results = df_results[filters]

        skeleton2ds_results = df_results[filter_skeleton2d].to_numpy(dtype=np.float32).reshape(df_full_length, len(SKELETON_PEDREC_JOINTS), 5)
        skeleton3ds_results = df_results[filter_skeleton3d].to_numpy(dtype=np.float32).reshape(df_full_length, len(SKELETON_PEDREC_JOINTS), 6)

    # test = df[filter_skeleton2d]
    annotations = PedRecDatasetEntries(
        img_ids=df['img_id'].to_numpy(dtype=np.uint32),
        img_dirs=df['img_dir'].to_numpy(dtype=str),
        img_types=df['img_type'].to_numpy(dtype=str),
        scene_ids=df['scene_id'].to_numpy(dtype=np.uint32),
        scene_starts=df['scene_start'].to_numpy(dtype=np.uint32),
        scene_ends=df['scene_end'].to_numpy(dtype=np.uint32),
        frame_nr_globals=df['frame_nr_global'].to_numpy(dtype=np.uint32),
        frame_nr_locals=df['frame_nr_local'].to_numpy(dtype=np.uint32),
        subject_ids=df['subject_id'].to_numpy(dtype=str),
        genders=df['gender'].to_numpy(dtype=np.uint32),
        skin_colors=df['skin_color'].to_numpy(dtype=np.uint32),
        sizes=df['size'].to_numpy(dtype=np.uint32),
        bmis=df['bmi'].to_numpy(dtype=np.uint32),
        ages=df['age'].to_numpy(dtype=np.uint32),
        movements=df['movement'].to_numpy(dtype=np.uint32),
        movement_speeds=df['movement_speed'].to_numpy(dtype=np.uint32),
        is_real_imgs=df['is_real_img'].to_numpy(dtype=np.uint32),
        actions=df['actions'].tolist(),
        bbs=df[filter_bb].to_numpy(dtype=np.float32),
        env_positions=df[filter_env].to_numpy(dtype=np.float32),
        body_orientations=df[filter_body_orientation].to_numpy(dtype=np.float32).reshape(df_full_length, 1, 4),
        head_orientations=df[filter_head_orientation].to_numpy(dtype=np.float32).reshape(df_full_length, 1, 4),
        skeleton2ds=df[filter_skeleton2d].to_numpy(dtype=np.float32).reshape(df_full_length, len(SKELETON_PEDREC_JOINTS), 5),
        skeleton3ds=df[filter_skeleton3d].to_numpy(dtype=np.float32).reshape(df_full_length, len(SKELETON_PEDREC_JOINTS), 6),
        skeleton2ds_results=skeleton2ds_results,
        skeleton3ds_results=skeleton3ds_results
    )
    # test = df['movement'].unique()
    # print(f"SKELETON3D MIN: {np.min(annotations.skeleton3ds)}")
    # print(f"SKELETON3D MAX: {np.max(annotations.skeleton3ds)}")

    centers, scales = bbs_to_centers_scales(annotations.bbs, input_size)
    annotations.centers = centers
    annotations.scales = scales
    # todo: flip y in skeleton3d, bzw. flip im netz einfach am ende?

    dataset_info = PedRecDatasetInfo(
        full_length=df_full_length,
        used_length=df_used_length,
        subsampling=cfg.subsample,
        provides_bbs=not ((df[filter_bb] == 0).all()).all(),
        provides_skeleton_2ds=not ((df[filter_skeleton2d] == 0).all()).all(),
        provides_skeleton_3ds=not ((df[filter_skeleton3d] == 0).all()).all(),
        provides_env_positions=not ((df[filter_env] == 0).all()).all(),
        provides_body_orientations=not ((df[filter_body_orientation] == 0).all()).all(),
        provides_head_orientations=not ((df[filter_head_orientation] == 0).all()).all(),
        provides_scene_ids=not (df['scene_id'] == -1).all(),
        provides_actions=not (df['actions'] == -1).all(),
        provides_movements=not (df['movement'] == -1).all(),
        provides_movement_speeds=not (df['movement_speed'] == -1).all(),
        provides_genders=not (df['gender'] == -1).all(),
        provides_skin_colors=not (df['skin_color'] == -1).all(),
        provides_sizes=not (df['size'] == -1).all(),
        provides_weights=not (df['bmi'] == -1).all(),
        provides_ages=not (df['age'] == -1).all(),
        provides_frame_nr_locals=not (df['frame_nr_local'] == -1).all(),
        provides_frame_nr_globals=not (df['frame_nr_global'] == -1).all(),
        provides_is_real_img=not (df['is_real_img'] == -1).all(),
    )
    return df_only_valid_skels["original_index"], annotations, dataset_info
