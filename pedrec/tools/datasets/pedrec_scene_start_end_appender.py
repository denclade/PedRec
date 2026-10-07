import pandas as pd
from pedrec.utils.pandas_helper import read_pedrec_df
import numpy as np


def run(df_path: str):
    df = read_pedrec_df(df_path)
    last_folder = None
    last_uid = None
    start_idx = -1
    df.insert(3, "scene_start", -1)
    df.insert(4, "scene_end", -1)
    for idx, row in df.iterrows():
        scene_uid = row[2]
        folder = row[7]
        if scene_uid != last_uid or folder != last_folder:
            # end_idx = idx-1
            if start_idx >= 0:
                # enter data from old values
                df.loc[start_idx:idx-1, ["scene_start"]] = [start_idx]
                df.loc[start_idx:idx-1, ["scene_end"]] = [idx-1]
            last_uid = scene_uid
            start_idx = idx
            last_folder = folder
    if start_idx == -1:
        start_idx = 0
    idx = len(df)
    df.loc[start_idx:idx, ["scene_start"]] = [start_idx]
    df.loc[start_idx:idx-1, ["scene_end"]] = [idx-1]
    df.to_pickle(df_path)

if __name__ == "__main__":
    # run("data/datasets/Human3.6m/train/h36m_train_pedrec.pkl")
    run("data/datasets/Human3.6m/val/h36m_val_pedrec.pkl")