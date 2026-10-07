import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import csv
import logging

import math
import os.path
import os.path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torchvision import transforms

from pedrec.networks.net_pedrec.pedrec_net_factory import load_pedrec_net
from pedrec.configs import default_paths
from pedrec.configs.app_config import AppConfig
from pedrec.configs.dataset_configs import get_coco_dataset_cfg_default
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.datasets.coco_dataset import CocoDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.utils.augmentation_helper import get_affine_transforms
from pedrec.utils.bb_helper import get_center_bb_from_coord_bb, \
    bb_to_center_scale
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.utils.torch_utils.torch_helper import get_device

pose_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_angular_distance(phi_gt_deg, phi_pred_deg):
    dist_phi = np.abs(phi_pred_deg - phi_gt_deg)
    return np.minimum(360 - dist_phi, dist_phi)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Body orientation (phi) accuracy of a PedRecNet on the COCO "
                                                 "validation set with MEBOW annotations.")
    parser.add_argument("--weights", default=None,
                        help=f"PedRecNet *_net.pth (default: <data-dir>/{default_paths.PEDREC_NET_WEIGHTS}).")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = logging.getLogger(__name__)
    pedrecnet_weights = args.weights or default_paths.pedrec_net_weights(args.data_dir)
    experiment_paths = get_experiment_paths(args.data_dir)
    dataset_cfg = get_coco_dataset_cfg_default()
    dataset_cfg.use_mebow_orientation = True
    cfg = PedRecNet50Config()
    device = get_device(not args.cpu)
    val_set = CocoDataset(experiment_paths.coco_dir, DatasetType.VALIDATE, dataset_cfg,
                          cfg.model.input_size, pose_transform)
    net = load_pedrec_net(pedrecnet_weights, device)
    distances = []
    with torch.no_grad():
        for annotation in val_set:
            img, labels = annotation
            model_input = img.unsqueeze(0).to(device)
            output = net(model_input)
            orientation_pred = output[2].cpu().detach().numpy()
            orientation_pred[:, :, 0] *= math.pi
            orientation_pred[:, :, 1] *= 2 * math.pi

            orientation_gt = labels["orientation"]
            if orientation_gt[0, 4] != 1:
                continue
            orientation_gt[:, 0] *= math.pi
            orientation_gt[:, 1] *= 2 * math.pi

            body_phi_degree_gt = math.degrees(orientation_gt[0, 1])
            body_phi_degree_pred = math.degrees(orientation_pred[0, 0, 1])
            distances.append(get_angular_distance(body_phi_degree_gt, body_phi_degree_pred))

    distances = np.array(distances, dtype=np.float32)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    acc_5 = len(np.where(distances <= 5)[0]) / distances.shape[0]
    acc_15 = len(np.where(distances <= 15)[0]) / distances.shape[0]
    acc22_5 = len(np.where(distances <= 22.5)[0]) / distances.shape[0]
    acc_30 = len(np.where(distances <= 30)[0]) / distances.shape[0]
    acc45 = len(np.where(distances <= 45)[0]) / distances.shape[0]
    print(f"Acc5: {acc_5}, Acc15: {acc_15},  Acc22.5: {acc22_5}, Acc30: {acc_30}, Acc45: {acc45}, "
          f"Mean: {mean_distance}, std_distance: {std_distance}")


if __name__ == "__main__":
    main()
