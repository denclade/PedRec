import sys

sys.path.append('.')  # allow running as a script from the repository root

import numpy as np
import pandas as pd

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.utils.pandas_helper import read_pedrec_df


def set_df_dtypes(df: pd.DataFrame):
    df["original_index"] = df["original_index"].astype("int")

    for joint in SKELETON_PEDREC_JOINTS:
        df[f"skeleton2d_{joint.name}_x"] = df[f"skeleton2d_{joint.name}_x"].astype("float32")
        df[f"skeleton2d_{joint.name}_y"] = df[f"skeleton2d_{joint.name}_y"].astype("float32")
        df[f"skeleton2d_{joint.name}_score"] = df[f"skeleton2d_{joint.name}_score"].astype("float32")
        df[f"skeleton2d_{joint.name}_visible"] = df[f"skeleton2d_{joint.name}_visible"].astype("category")
        df[f"skeleton2d_{joint.name}_supported"] = df[f"skeleton2d_{joint.name}_visible"].astype("category")

        df[f"skeleton3d_{joint.name}_x"] = df[f"skeleton3d_{joint.name}_x"].astype("float32")
        df[f"skeleton3d_{joint.name}_y"] = df[f"skeleton3d_{joint.name}_y"].astype("float32")
        df[f"skeleton3d_{joint.name}_z"] = df[f"skeleton3d_{joint.name}_z"].astype("float32")
        df[f"skeleton3d_{joint.name}_score"] = df[f"skeleton3d_{joint.name}_score"].astype("float32")
        df[f"skeleton3d_{joint.name}_visible"] = df[f"skeleton3d_{joint.name}_visible"].astype("category")
        df[f"skeleton3d_{joint.name}_supported"] = df[f"skeleton3d_{joint.name}_visible"].astype("category")

        df["body_orientation_phi"] = df["body_orientation_phi"].astype("float32")
        df["body_orientation_theta"] = df["body_orientation_theta"].astype("float32")
        df["body_orientation_score"] = df["body_orientation_score"].astype("float32")
        df["body_orientation_visible"] = df["body_orientation_theta_supported"].astype("category")
        df["body_orientation_visible"] = df["body_orientation_phi_supported"].astype("category")

        df["head_orientation_phi"] = df["head_orientation_phi"].astype("float32")
        df["head_orientation_theta"] = df["head_orientation_theta"].astype("float32")
        df["head_orientation_score"] = df["head_orientation_score"].astype("float32")
        df["head_orientation_visible"] = df["head_orientation_theta_supported"].astype("category")
        df["head_orientation_visible"] = df["head_orientation_phi_supported"].astype("category")


def get_column_names():
    column_names = [
        "index"
    ]
    for joint in SKELETON_PEDREC_JOINTS:
        column_names.append(f"skeleton2d_{joint.name}_x")
        column_names.append(f"skeleton2d_{joint.name}_y")
        column_names.append(f"skeleton2d_{joint.name}_score")
        column_names.append(f"skeleton2d_{joint.name}_visible")
        column_names.append(f"skeleton2d_{joint.name}_supported")

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


def write_allframes(dataset_path: str, result_path: str, output_path: str):
    """
    Expands a result dataframe (which only contains the valid frames) to all frames of the dataset dataframe so that
    both share the same index (required by the temporal EHPI3D datasets). Missing frames are filled with zeros.
    """
    df = read_pedrec_df(dataset_path)
    result_df = read_pedrec_df(result_path)
    result_df = result_df.drop(columns=['index'])

    skeleton2d_visibles = [col for col in df if col.startswith('skeleton2d') and col.endswith('_visible')]
    df["visible_joints"] = df[skeleton2d_visibles].sum(axis=1)
    df["valid"] = (df['bb_score'] >= 1) & (df['visible_joints'] >= 3)
    df["original_index"] = df.index.astype('int32')
    y = df[df["valid"] == True]["original_index"]

    new_results_df = pd.DataFrame(np.zeros((df.shape[0], result_df.shape[1]), dtype=np.float32), columns=result_df.columns)
    new_results_df["original_index"] = df["original_index"]
    result_df["original_index"] = y.values

    new_results_df = pd.concat([new_results_df, result_df])
    new_results_df = new_results_df.drop_duplicates(['original_index'], keep='last')
    new_results_df = new_results_df.sort_values('original_index')
    set_df_dtypes(new_results_df)
    pd.to_pickle(new_results_df, output_path)
    print(f"Wrote {output_path}")


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Expands SIM-C01 result dataframes to all frames (*_allframes.pkl), "
                                                 "the input format of the EHPI3D training / evaluation.")
    parser.add_argument("--experiment", default="p2d3d_c_o_h36m_sim_mebow", help="Training stage name.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    return parser.parse_args(argv)


def cli(argv=None):
    import os
    from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
    args = parse_args(argv)
    experiment_paths = get_experiment_paths(args.data_dir)
    experiment_name = f"experiment_pedrec_{args.experiment}_0"
    if args.split == "train":
        dataset_dir, dataset_filename, prefix = experiment_paths.sim_c01_dir, experiment_paths.sim_c01_filename, "C01F_train_pred_df"
    else:
        dataset_dir, dataset_filename, prefix = experiment_paths.sim_c01_val_dir, experiment_paths.sim_c01_val_filename, "C01F_pred_df"
    write_allframes(os.path.join(dataset_dir, dataset_filename),
                    os.path.join(dataset_dir, "results", f"{prefix}_{experiment_name}.pkl"),
                    os.path.join(dataset_dir, f"{prefix}_{experiment_name}_allframes.pkl"))


if __name__ == '__main__':
    cli()
