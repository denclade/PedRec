import sys

sys.path.append('.')  # allow running as a script from the repository root

from pedrec.networks.net_pedrec.pedrec_net import PedRecNet


from torch.utils.data import DataLoader

from pedrec.configs.dataset_configs import PedRecDatasetConfig, get_sim_val_dataset_cfg_default, \
    get_h36m_dataset_cfg_default, get_sim_dataset_cfg_default
from pedrec.datasets.pedrec_dataset import PedRecDataset
from pedrec.evaluations.eval_helper import get_total_coords
from pedrec.models.constants.dataset_constants import DatasetType
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.training.experiments.experiment_train_helper import init_experiment, \
    get_outputs_loss_mtl
from pedrec.utils.torch_utils.torch_helper import get_device, move_to_device
import pandas as pd
import numpy as np
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS


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


def get_preds_mtl(outputs: torch.Tensor):
    return {
        "skeleton": outputs[0].cpu().detach().numpy(),
        "skeleton_3d": outputs[1].cpu().detach().numpy(),
        "orientation": outputs[2].cpu().detach().numpy(),
    }


def main(output_path: str, dataset_cfg: PedRecDatasetConfig, pedrec_dataset_dir, pedrec_dataset_filename):
    # experiment_paths = get_experiment_paths_home()
    net_cfg = PedRecNet50Config()
    # sim_val_dataset_cfg: PedRecDatasetConfig = get_sim_val_dataset_cfg_default()
    # sim_val_dataset_cfg.subsample = 1
    init_experiment(42)
    device = get_device(use_gpu=True)

    ####################################################################################################################
    ############################################ Initialize Network ####################################################
    ####################################################################################################################
    net = PedRecNet(net_cfg)
    net.init_weights()
    net.to(device)
    net.load_state_dict(torch.load("data/models/pedrec/experiment_pedrec_direct_4_net.pth"))

    ####################################################################################################################
    ################################################# Datasets #########################################################
    ####################################################################################################################
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = PedRecDataset(pedrec_dataset_dir,
                            pedrec_dataset_filename,
                            DatasetType.VALIDATE,
                            dataset_cfg,
                            net_cfg.model.input_size,
                            trans)

    batch_size = 48
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    net.eval()
    result_rows = []
    # count = 0
    column_names = get_column_names()
    with torch.no_grad():
        for test_data in data_loader:
            # if count > 2:
            #     break
            # count += 1
            images, labels = test_data
            images = images.to(device)
            labels = move_to_device(labels, device)
            outputs = net(images)
            preds = get_preds_mtl(outputs)

            idxs = labels["idx"].cpu().detach().numpy()
            centers = labels["center"].cpu().detach().numpy()
            scales = labels["scale"].cpu().detach().numpy()
            rotations = labels["rotation"].cpu().detach().numpy()

            pose2d_preds = preds["skeleton"]
            pose2d_preds = get_total_coords(pose2d_preds, net_cfg.model.input_size, centers, scales, rotations)

            pose3d_preds = preds["skeleton_3d"]
            pose3d_preds[:, :, :3] = (pose3d_preds[:, :, :3] * 3000) - 1500  # to cm

            orientation_preds = preds["orientation"]

            for i in range(0, pose2d_preds.shape[0]):
                orientation_pred = orientation_preds[i]
                idx = idxs[i]
                pose2d_pred = pose2d_preds[i]
                pose3d_pred = pose3d_preds[i]
                visibles = (pose2d_pred[:, 2] > 0.5).astype(np.int32)
                supported = np.ones(pose2d_pred.shape[0])
                visible_supported = np.array([visibles, supported]).transpose(1, 0)
                pose2d_pred = np.concatenate((pose2d_pred, visible_supported), axis=1)
                pose3d_pred = np.concatenate((pose3d_pred, visible_supported), axis=1)
                pose2d_pred = pose2d_pred.reshape(-1).tolist()
                pose3d_pred = pose3d_pred.reshape(-1).tolist()
                orientation_pred_body = orientation_pred[0].reshape(-1).tolist()
                orientation_pred_head = orientation_pred[1].reshape(-1).tolist()
                result_rows.append([idx] + pose2d_pred + pose3d_pred + orientation_pred_body + [1, 1] + orientation_pred_head + [1, 1])

    df = pd.DataFrame(data=result_rows, columns=get_column_names())
    print(df.memory_usage(deep=True))
    set_df_dtypes(df)
    print(df.memory_usage(deep=True))
    df.to_pickle(output_path)

def write_allframes(dataset_path: str, result_path: str, output_path: str):
    """
    Expands a result dataframe (which only contains the valid frames) to all frames of the dataset dataframe so that
    both share the same index (required by the temporal EHPI3D datasets). Missing frames are filled with zeros.
    """
    df = pd.read_pickle(dataset_path)
    result_df = pd.read_pickle(result_path)
    result_df = result_df.drop(columns=['index'])

    skeleton2d_visibles = [col for col in df if col.startswith('skeleton2d') and col.endswith('_visible')]
    df["visible_joints"] = df[skeleton2d_visibles].sum(axis=1)
    df["valid"] = (df['bb_score'] >= 1) & (df['visible_joints'] >= 3)
    df["original_index"] = df.index.copy(dtype='int32')
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
