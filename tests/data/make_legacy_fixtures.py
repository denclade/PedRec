"""
Generates small dataset / result dataframes in the original PedRec format with the ORIGINAL library versions
(Python 3.9, pandas 1.3.5, numpy 1.21.6) to verify that the current code base can still read the published
datasets. Run with:

    uv run --python 3.9 --with pandas==1.3.5 --with numpy==1.21.6 python tests/data/make_legacy_fixtures.py

The structure mirrors ``pedrec.tools.datasets.sim_dataset_generator.get_column_names`` / ``set_df_dtypes``; the
script is self contained so that it does not depend on the (newer) PedRec code.
"""
import os

import numpy as np
import pandas as pd

JOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
          "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
          "right_ankle", "hip_center", "spine_center", "neck", "head_lower", "head_upper", "left_foot_end",
          "right_foot_end", "left_hand_end", "right_hand_end"]
NUM_FRAMES = 40  # two scenes a 20 frames
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_gt_df(rng: np.random.RandomState) -> pd.DataFrame:
    rows = {}
    n = NUM_FRAMES
    scene = np.repeat([0, 1], n // 2)
    rows["dataset"] = ["RT3DConti01"] * n
    rows["dataset_type"] = ["TRAIN"] * n
    rows["scene_id"] = scene
    rows["scene_start"] = np.where(scene == 0, 0, n // 2)
    rows["scene_end"] = np.where(scene == 0, n // 2 - 1, n - 1)
    rows["frame_nr_global"] = np.arange(n)
    rows["frame_nr_local"] = np.arange(n) % (n // 2)
    rows["img_dir"] = [f"Conti01_01/view_{s}" for s in scene]
    rows["img_id"] = np.arange(n)
    rows["img_type"] = ["png"] * n
    rows["subject_id"] = [f"char_{s}" for s in scene]
    for col in ["gender", "skin_color", "size", "bmi", "age"]:
        rows[col] = scene
    rows["movement"] = np.where(scene == 0, 1, 19)  # WALK / STAND
    rows["movement_speed"] = scene
    rows["is_real_img"] = [False] * n
    rows["actions"] = [[1, 23] if s == 0 else [24] for s in scene]  # WALK + LOOK_FOR_TRAFFIC / HITCHHIKE
    rows["bb_center_x"] = 960 + rng.randn(n) * 10
    rows["bb_center_y"] = 540 + rng.randn(n) * 10
    rows["bb_width"] = 150 + rng.rand(n) * 10
    rows["bb_height"] = 400 + rng.rand(n) * 10
    rows["bb_score"] = np.ones(n)
    rows["bb_class"] = np.zeros(n, dtype=int)
    for axis in "xyz":
        rows[f"env_position_{axis}"] = rng.randn(n) * 1000
    for part in ["body", "head"]:
        rows[f"{part}_orientation_theta"] = rng.rand(n)
        rows[f"{part}_orientation_phi"] = rng.rand(n)
        rows[f"{part}_orientation_score"] = np.ones(n)
        rows[f"{part}_orientation_visible"] = np.ones(n, dtype=int)
    for joint in JOINTS:
        rows[f"skeleton2d_{joint}_x"] = 900 + rng.rand(n) * 120
        rows[f"skeleton2d_{joint}_y"] = 350 + rng.rand(n) * 380
        rows[f"skeleton2d_{joint}_score"] = np.ones(n)
        rows[f"skeleton2d_{joint}_visible"] = np.ones(n, dtype=int)
        rows[f"skeleton2d_{joint}_supported"] = np.ones(n, dtype=int)
    for joint in JOINTS:
        for axis in "xyz":
            rows[f"skeleton3d_{joint}_{axis}"] = rng.randn(n) * 400
        rows[f"skeleton3d_{joint}_score"] = np.ones(n)
        rows[f"skeleton3d_{joint}_visible"] = np.ones(n, dtype=int)
        rows[f"skeleton3d_{joint}_supported"] = np.ones(n, dtype=int)
    df = pd.DataFrame(rows)

    # dtypes as in pedrec.tools.datasets.dataset_generator_helper.set_df_dtypes
    for col in ["dataset", "dataset_type", "scene_id", "scene_start", "scene_end", "img_dir", "img_type", "subject_id",
                "gender", "skin_color", "size", "bmi", "age", "movement", "movement_speed", "bb_class",
                "body_orientation_visible", "head_orientation_visible"]:
        df[col] = df[col].astype("category")
    for col in ["frame_nr_global", "frame_nr_local", "img_id"]:
        df[col] = df[col].astype("uint32")
    df["is_real_img"] = df["is_real_img"].astype("bool")
    for col in df.columns:
        if col.startswith(("bb_center", "bb_width", "bb_height", "bb_score", "env_position", "body_orientation_",
                           "head_orientation_", "skeleton2d_", "skeleton3d_")) \
                and not col.endswith(("_visible", "_supported")):
            df[col] = df[col].astype("float32")
    for joint in JOINTS:
        for prefix in ["skeleton2d", "skeleton3d"]:
            df[f"{prefix}_{joint}_visible"] = df[f"{prefix}_{joint}_visible"].astype("category")
            df[f"{prefix}_{joint}_supported"] = df[f"{prefix}_{joint}_visible"].astype("category")
    return df


def make_result_df(gt_df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    """Result dataframe of the PedRecNet (same index as the gt, as written by the result writers)."""
    rows = {}
    n = len(gt_df)
    for part in ["body", "head"]:
        rows[f"{part}_orientation_theta"] = rng.rand(n)
        rows[f"{part}_orientation_phi"] = rng.rand(n)
        rows[f"{part}_orientation_score"] = np.ones(n)
        rows[f"{part}_orientation_visible"] = np.ones(n, dtype=int)
    for joint in JOINTS:
        rows[f"skeleton2d_{joint}_x"] = gt_df[f"skeleton2d_{joint}_x"].to_numpy() + rng.randn(n)
        rows[f"skeleton2d_{joint}_y"] = gt_df[f"skeleton2d_{joint}_y"].to_numpy() + rng.randn(n)
        rows[f"skeleton2d_{joint}_score"] = rng.rand(n)
        rows[f"skeleton2d_{joint}_visible"] = np.ones(n, dtype=int)
        rows[f"skeleton2d_{joint}_supported"] = np.ones(n, dtype=int)
    for joint in JOINTS:
        for axis in "xyz":
            rows[f"skeleton3d_{joint}_{axis}"] = gt_df[f"skeleton3d_{joint}_{axis}"].to_numpy() + rng.randn(n) * 10
        rows[f"skeleton3d_{joint}_score"] = rng.rand(n)
        rows[f"skeleton3d_{joint}_visible"] = np.ones(n, dtype=int)
        rows[f"skeleton3d_{joint}_supported"] = np.ones(n, dtype=int)
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col.endswith(("_visible", "_supported")):
            df[col] = df[col].astype("category")
        else:
            df[col] = df[col].astype("float32")
    return df


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    gt = make_gt_df(rng)
    res = make_result_df(gt, rng)
    gt.to_pickle(os.path.join(OUT_DIR, "legacy_pedrec_gt_df.pkl"))
    res.to_pickle(os.path.join(OUT_DIR, "legacy_pedrec_result_df.pkl"))
    print(f"pandas {pd.__version__}, numpy {np.__version__}: wrote {gt.shape} / {res.shape}")
