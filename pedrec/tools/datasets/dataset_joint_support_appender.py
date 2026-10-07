import sys

sys.path.append(".")
import numpy as np
import pandas as pd
from pedrec.utils.pandas_helper import read_pedrec_df

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS


def run_h36m(df: pd.DataFrame):
    df_size = len(df)
    for joint in SKELETON_PEDREC_JOINTS:
        # nose, eyes and ears are not provided
        new_col = [True] * df_size if joint.value > 4 else [False] * df_size

        # 2d
        visible_col_idx = df.columns.get_loc(f"skeleton2d_{joint.name}_visible")
        new_col_idx = visible_col_idx + 1
        new_col_name = f"skeleton2d_{joint.name}_supported"
        df.insert(loc=new_col_idx, column=new_col_name, value=new_col)
        df["is_real_img"] = df["is_real_img"].astype("uint32")
        # 3d
        visible_col_idx = df.columns.get_loc(f"skeleton3d_{joint.name}_visible")
        new_col_idx = visible_col_idx + 1
        new_col_name = f"skeleton3d_{joint.name}_supported"
        df.insert(loc=new_col_idx, column=new_col_name, value=new_col)
    return df


def run_sim(df: pd.DataFrame):
    df_size = len(df)
    for joint in SKELETON_PEDREC_JOINTS:
        new_col = [True] * df_size
        # 2d
        visible_col_idx = df.columns.get_loc(f"skeleton2d_{joint.name}_visible")
        new_col_idx = visible_col_idx + 1
        new_col_name = f"skeleton2d_{joint.name}_supported"
        df.insert(loc=new_col_idx, column=new_col_name, value=new_col)
        df["is_real_img"] = df["is_real_img"].astype("uint32")
        # 3d
        visible_col_idx = df.columns.get_loc(f"skeleton3d_{joint.name}_visible")
        new_col_idx = visible_col_idx + 1
        new_col_name = f"skeleton3d_{joint.name}_supported"
        df.insert(loc=new_col_idx, column=new_col_name, value=new_col)
    return df


if __name__ == "__main__":
    # sim_train_path = "data/datasets/ROMb/rt_rom_01b.pkl"
    # sim_val_path = "data/datasets/RT3DValidate/rt_validate_3d.pkl"
    # h36m_train_path = "data/datasets/Human3.6m/train/h36m_train.pkl"
    # h36m_val_path = "data/datasets/Human3.6m/val/h36m_val.pkl"
    
    # sim_train_path = "data/datasets/ROMb/rt_rom_01b.pkl"
    # sim_val_path = "data/datasets/RT3DValidate/rt_validate_3d.pkl"
    # h36m_train_path = "data/datasets/Human3.6m/train/h36m_train.pkl"
    # h36m_val_path = "data/datasets/Human3.6m/val/h36m_val.pkl"

    sim_train_path = "data/datasets/ROMb/rt_rom_01b.pkl"
    sim_val_path = "data/datasets/RT3DValidate/rt_validate_3d.pkl"
    h36m_train_path = "data/datasets/Human3.6M/train/h36m_train.pkl"
    h36m_val_path = "data/datasets/Human3.6M/val/h36m_val.pkl"
    
    pd.to_pickle(run_sim(read_pedrec_df(sim_train_path)), sim_train_path)
    pd.to_pickle(run_sim(read_pedrec_df(sim_val_path)), sim_val_path)
    pd.to_pickle(run_h36m(read_pedrec_df(h36m_train_path)), h36m_train_path)
    pd.to_pickle(run_h36m(read_pedrec_df(h36m_val_path)), h36m_val_path)
