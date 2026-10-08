import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import os
from pathlib import Path

from pedrec.networks.net_pedrec.pedrec_net_factory import load_pedrec_net
from pedrec.datasets.pedrec_dataset import PedRecDataset
from pedrec.training.experiments.experiment_initializer import initialize_weights_with_same_name_and_shape


from torch.utils.data import DataLoader

from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.pedrec_stages import STAGES
from pedrec.configs.dataset_configs import get_h36m_val_dataset_cfg_default
from pedrec.evaluations.eval_helper import get_total_coords
from pedrec.models.constants.dataset_constants import DatasetType
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.training.experiments.experiment_train_helper import init_experiment
from pedrec.utils.torch_utils.torch_helper import get_device, move_to_device
import pandas as pd
import numpy as np
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS


def set_df_dtypes(df: pd.DataFrame):
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

def get_column_names():
    column_names = [
        "img_path"
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

    return column_names


def get_preds_mtl(outputs: torch.Tensor):
    return {
        "skeleton": outputs[0].cpu().detach().numpy(),
        "skeleton_3d": outputs[1].cpu().detach().numpy(),
        "orientation": outputs[2].cpu().detach().numpy(),
    }


def main(output_dir: str, output_postfix: str, net_cfg, h36m_val, weights_path: str):
    init_experiment(42)
    device = get_device(use_gpu=True)

    ####################################################################################################################
    ############################################ Initialize Network ####################################################
    ####################################################################################################################
    net = load_pedrec_net(weights_path, device)  # MTL checkpoint or *_net.pth
    net.to(device)

    batch_size = 48
    data_loader = DataLoader(h36m_val, batch_size=batch_size, shuffle=False, num_workers=12)
    net.eval()
    gt_rows = []
    result_rows = []
    # count = 0
    # column_names = get_column_names()
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
            pose2d_gts = labels["skeleton"].cpu().detach().numpy()
            pose3d_gts = labels["skeleton_3d"].cpu().detach().numpy()
            # orientation_gts = labels["orientation"].cpu().detach().numpy()
            img_paths = labels["img_path"]
            # img_sizes = labels["img_size"].cpu().detach().numpy().tolist()
            pose2d_gts = get_total_coords(pose2d_gts, net_cfg.model.input_size, centers, scales, rotations)
            
            pose2d_preds = preds["skeleton"]
            pose2d_preds = get_total_coords(pose2d_preds, net_cfg.model.input_size, centers, scales, rotations)
            pose3d_preds = preds["skeleton_3d"]
            pose3d_preds[:, :, :3] = (pose3d_preds[:, :, :3] * 3000) - 1500  # to cm

            pose3d_gts[:, :, :3] = (pose3d_gts[:, :, :3] * 3000) - 1500  # to cm
            # 
            # orientation_preds = preds["orientation"]

            for i in range(0, pose2d_preds.shape[0]):
                img_path = img_paths[i]
                idx = idxs[i]
                
                pose2d_pred = pose2d_preds[i]
                pose2d_gt = pose2d_gts[i]
                visibles = pose2d_gt[:, 3]
                supported = pose2d_gt[:, 4]
                visible_supported = np.array([visibles, supported]).transpose(1, 0)
                pose2d_pred = np.concatenate((pose2d_pred, visible_supported), axis=1)

                pose2d_pred = pose2d_pred.reshape(-1).tolist()
                pose2d_gt = pose2d_gt.reshape(-1).tolist()

                pose3d_pred = pose3d_preds[i]
                pose3d_gt = pose3d_gts[i]
                visibles = pose3d_gt[:, 4]
                supported = pose3d_gt[:, 5]
                visible_supported = np.array([visibles, supported]).transpose(1, 0)
                pose3d_pred = np.concatenate((pose3d_pred, visible_supported), axis=1)
                pose3d_pred = pose3d_pred.reshape(-1).tolist()
                pose3d_gt = pose3d_gt.reshape(-1).tolist()

                result_rows.append([img_path] + pose2d_pred + pose3d_pred)
                gt_rows.append([img_path] + pose2d_gt + pose3d_gt)

    df_pred = pd.DataFrame(data=result_rows, columns=get_column_names())
    set_df_dtypes(df_pred)
    output_path = os.path.join(output_dir, f"H36M_pred_df_{output_postfix}.pkl")
    df_pred.to_pickle(output_path)
    
    df_gt = pd.DataFrame(data=gt_rows, columns=get_column_names())
    set_df_dtypes(df_gt)
    output_path = os.path.join(output_dir, f"H36M_gt_df_{output_postfix}.pkl")
    df_gt.to_pickle(output_path)


DEFAULT_EXPERIMENTS = ["p2d3d_c_o"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Writes PedRecNet predictions on the Human3.6m validation set as "
                                                 "result dataframes for evaluations/h36m_eval_export.py.")
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS, choices=sorted(STAGES.keys()),
                        metavar="STAGE", help="Training stages (checkpoints) to run.")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--no-flipped", action="store_true", help="Skip the flipped (flip test) run.")
    return parser.parse_args(argv)


def cli(argv=None):
    args = parse_args(argv)
    experiment_paths = get_experiment_paths(args.data_dir)
    network_paths = [experiment_paths.get_stage_checkpoint_path(name) for name in args.experiments]
    output_dir = os.path.dirname(os.path.normpath(experiment_paths.h36m_val_dir))
    net_cfg = PedRecNetConfig()

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    h36m_val_dataset_cfg = get_h36m_val_dataset_cfg_default()
    h36m_val_dataset_cfg.subsample = 1
    h36m_val_dataset_cfg.flip = False
    h36m_val_dataset_cfg.scale_factor = 0
    h36m_val_dataset_cfg.rotation_factor = 0
    h36m_val_dataset_cfg.gt_result_ratio = 1

    h36m_val = PedRecDataset(experiment_paths.h36m_val_dir,
                             experiment_paths.h36m_val_filename,
                             DatasetType.VALIDATE, h36m_val_dataset_cfg,
                             net_cfg.model.input_size, trans, is_h36m=True)
    h36m_val_flipped = PedRecDataset(experiment_paths.h36m_val_dir,
                             experiment_paths.h36m_val_filename,
                             DatasetType.VALIDATE, h36m_val_dataset_cfg,
                             net_cfg.model.input_size, trans, is_h36m=True, flip_all=True)

    for net_path in network_paths:
        print(f"Working on {net_path}")
        main(output_dir=output_dir,
             output_postfix=Path(net_path).stem,
             net_cfg=net_cfg,
             h36m_val=h36m_val,
             weights_path=net_path)
        if not args.no_flipped:
            main(output_dir=output_dir,
                 output_postfix=f"{Path(net_path).stem}_flipped",
                 net_cfg=net_cfg,
                 h36m_val=h36m_val_flipped,
                 weights_path=net_path)


if __name__ == '__main__':
    cli()
