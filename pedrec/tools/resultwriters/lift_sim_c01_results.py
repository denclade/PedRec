"""
Replaces the per frame 3D poses in the SIM-C01 result dataframes (``*_allframes.pkl``) with the poses of the temporal
3D lifter (``*_allframes_lifted.pkl``), so that the action recognition is trained on the same 3D poses the pipeline
produces at runtime.

    python pedrec/tools/resultwriters/lift_sim_c01_results.py --split train
"""
import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from pedrec.configs import default_paths
from pedrec.datasets.pose_sequence_dataset import PoseSequenceDataset
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.networks.net_pedrec.pose_lifter import TemporalPoseLifter, SKELETON_3D_RANGE
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.utils.pandas_helper import read_pedrec_df
from pedrec.utils.pedrec_dataset_helper import get_filter_skeleton3d
from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file
from pedrec.utils.torch_utils.torch_helper import get_device


@torch.inference_mode()
def lift_results(df_path: str, results_path: str, output_path: str, lifter: torch.nn.Module, device: torch.device,
                 batch_size: int = 1024) -> int:
    dataset = PoseSequenceDataset(df_path, results_path, train=False)
    indices = [i for i, position in enumerate(dataset.targets) if dataset.valid_result[dataset.order[position]]]
    lifted = dataset.result_3d.copy()
    lifter = lifter.to(device).eval()
    offset = 0
    for features, _, _ in DataLoader(Subset(dataset, indices), batch_size=batch_size):
        pred = lifter(features.to(device)).float().cpu().numpy() * SKELETON_3D_RANGE
        rows = dataset.order[dataset.targets[indices[offset:offset + len(pred)]]]
        lifted[rows, :, :3] = pred
        offset += len(pred)
    results = read_pedrec_df(results_path)
    columns = get_filter_skeleton3d(results)
    values = lifted.reshape(len(results), -1)
    for i, column in enumerate(columns):
        if column.endswith(("_x", "_y", "_z")):
            results[column] = values[:, i].astype(np.float32)
    results.to_pickle(output_path)
    return len(indices)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--weights", default=None, help=f"Default: <data-dir>/{default_paths.LIFTER_WEIGHTS}")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = get_experiment_paths(args.data_dir)
    if args.split == "train":
        directory, df, results, output = (paths.sim_c01_dir, paths.sim_c01_filename, paths.sim_c01_results_filename,
                                          paths.sim_c01_lifted_results_filename)
    else:
        directory, df, results, output = (paths.sim_c01_val_dir, paths.sim_c01_val_filename,
                                          paths.sim_c01_val_results_filename, paths.sim_c01_val_lifted_results_filename)
    lifter = TemporalPoseLifter(len(SKELETON_PEDREC_JOINTS))
    lifter.load_state_dict(load_state_dict_file(args.weights or default_paths.lifter_weights(args.data_dir)))
    output_path = os.path.join(directory, output)
    count = lift_results(os.path.join(directory, df), os.path.join(directory, results), output_path, lifter,
                         get_device(not args.cpu))
    print(f"Lifted {count} frames -> {output_path}")


if __name__ == "__main__":
    main()
