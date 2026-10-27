import numpy as np
import pandas as pd

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS


# longest names first: "left_foot_end" must not match as "left_foot"
_JOINT_NAMES_BY_LENGTH = sorted(((joint.name, joint.value) for joint in SKELETON_PEDREC_JOINTS),
                                key=lambda item: -len(item[0]))


def _split_column(column: str, prefix: str):
    """'skeleton2d_left_foot_end_x' -> (joint index, 'x')"""
    rest = column[len(prefix) + 1:]
    for name, index in _JOINT_NAMES_BY_LENGTH:
        if rest.startswith(name + "_"):
            return index, rest[len(name) + 1:]
    raise ValueError(f"Unknown joint in column '{column}'")


def get_skeleton_columns(df: pd.DataFrame, prefix: str):
    """
    The skeleton columns ordered by joint and value (x, y, ... in the order the values first appear in the file),
    independent of the column positions in the file. The arrays are built by reshaping these columns to joints x values.
    """
    columns = [col for col in df if col.startswith(prefix)]
    split = {col: _split_column(col, prefix) for col in columns}
    value_order = {}
    for col in columns:
        value_order.setdefault(split[col][1], len(value_order))
    return sorted(columns, key=lambda col: (split[col][0], value_order[split[col][1]]))


def get_filter_skeleton2d(df: pd.DataFrame):
    return get_skeleton_columns(df, 'skeleton2d')


def get_filter_skeleton3d(df: pd.DataFrame):
    return get_skeleton_columns(df, 'skeleton3d')


def order_skeleton_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The dataframe with its skeleton 2D / 3D columns in joint order (all other columns keep their position)."""
    columns = list(df.columns)
    for prefix in ('skeleton2d', 'skeleton3d'):
        positions = [i for i, col in enumerate(columns) if col.startswith(prefix)]
        for position, col in zip(positions, get_skeleton_columns(df, prefix)):
            columns[position] = col
    return df[columns]


def get_filter_bb(df: pd.DataFrame):
    return [col for col in df if col.startswith('bb')]


def get_filter_body_orientation(df: pd.DataFrame):
    return [col for col in df if col.startswith('body_orientation')]


def get_filter_head_orientation(df: pd.DataFrame):
    return [col for col in df if col.startswith('head_orientation')]


def get_filter_env(df: pd.DataFrame):
    return [col for col in df if col.startswith('env')]


def get_gt_arrays(df: pd.DataFrame):
    df_len = len(df)
    filter_skeleton2d = get_filter_skeleton2d(df)
    filter_skeleton3d = get_filter_skeleton3d(df)
    filter_bb = get_filter_bb(df)
    filter_body_orientation = get_filter_body_orientation(df)
    filter_head_orientation = get_filter_head_orientation(df)
    filter_env = get_filter_env(df)

    bbs = df[filter_bb].to_numpy(dtype=np.float32)
    env_positions = df[filter_env].to_numpy(dtype=np.float32)
    body_orientations = df[filter_body_orientation].to_numpy(dtype=np.float32).reshape(df_len, 1, 4)
    head_orientations = df[filter_head_orientation].to_numpy(dtype=np.float32).reshape(df_len, 1, 4)
    skeleton2ds = df[filter_skeleton2d].to_numpy(dtype=np.float32).reshape(df_len, len(SKELETON_PEDREC_JOINTS), 5)
    skeleton3ds = df[filter_skeleton3d].to_numpy(dtype=np.float32).reshape(df_len, len(SKELETON_PEDREC_JOINTS), 6)

    return skeleton2ds, skeleton3ds, body_orientations, head_orientations, bbs, env_positions


def get_pred_arrays(df: pd.DataFrame):
    df_len = len(df)
    filter_skeleton2d = get_filter_skeleton2d(df)
    filter_skeleton3d = get_filter_skeleton3d(df)
    filter_body_orientation = get_filter_body_orientation(df)
    filter_head_orientation = get_filter_head_orientation(df)

    body_orientations = df[filter_body_orientation].to_numpy(dtype=np.float32).reshape(df_len, 1, 4)
    head_orientations = df[filter_head_orientation].to_numpy(dtype=np.float32).reshape(df_len, 1, 4)
    skeleton2ds = df[filter_skeleton2d].to_numpy(dtype=np.float32).reshape(df_len, len(SKELETON_PEDREC_JOINTS), 5)
    skeleton3ds = df[filter_skeleton3d].to_numpy(dtype=np.float32).reshape(df_len, len(SKELETON_PEDREC_JOINTS), 6)

    return skeleton2ds, skeleton3ds, body_orientations, head_orientations
