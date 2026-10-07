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
from sklearn.metrics import average_precision_score, recall_score
from torchvision import transforms

from pedrec.networks.net_pedrec.pedrec_net_factory import load_pedrec_net
from pedrec.configs import default_paths
from pedrec.configs.app_config import AppConfig
from pedrec.configs.pedrec_net_config import PedRecNet50Config
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


def get_xml_data(path: str):
    src_dir = os.path.dirname(os.path.abspath(path))
    tree = ET.parse(path)
    annotationlist = tree.getroot()
    annotations = []
    orientation_degrees = {}
    with open(os.path.join(src_dir, "finalMeanAnnotationTUD.csv")) as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            parts = row[0].split('_')
            img_file_path = os.path.join(src_dir, parts[0], f"{parts[1]}.png")
            if img_file_path not in orientation_degrees:
                orientation_degrees[img_file_path] = []
            orientation_degrees[img_file_path].append(float(row[1]))

    for annotation in annotationlist:
        img_path = None
        for child in annotation:
            annot_num = 0
            if child.tag == "silhouette":
                orientation_bin = child.text
            if child.tag == "image":
                img_path = os.path.join(src_dir, child.find("name").text)
            if child.tag == "annorect":
                x_1 = int(child.find("x1").text)
                y_1 = int(child.find("y1").text)
                x_2 = int(child.find("x2").text)
                y_2 = int(child.find("y2").text)
                bb = [x_1, y_1, x_2, y_2, 1, 1]
                bb = get_center_bb_from_coord_bb(bb)
                orientation_bin = int(child.find("silhouette").find("id").text)
                if img_path == None:
                    raise("WTF")
                annotations.append({
                    "img_path": img_path,
                    "orientation_bin": orientation_bin,
                    "orientation": orientation_degrees[img_path][annot_num],
                    "bb": bb
                })
                annot_num += 1
    return annotations


def orientation_bin_from_orientation(orientation_phi_deg: float):
    if (337.5 <= orientation_phi_deg <= 0) \
            or (337.5 <= orientation_phi_deg <= 360) \
            or (22.5 >= orientation_phi_deg > 0):
        return 1
    if 22.5 <= orientation_phi_deg <= 67.5:
        return 2
    if 67.5 <= orientation_phi_deg <= 112.5:
        return 3
    if 112.5 <= orientation_phi_deg <= 157.5:
        return 4
    if 157.5 <= orientation_phi_deg <= 202.5:
        return 5
    if 202.5 <= orientation_phi_deg <= 247.5:
        return 6
    if 247.5 <= orientation_phi_deg <= 292.5:
        return 7
    if 292.5 <= orientation_phi_deg <= 337.5:
        return 8

    raise("WTF")


def get_angular_distance(phi_gt_deg, phi_pred_deg):
    dist_phi = np.abs(math.radians(phi_pred_deg) - math.radians(phi_gt_deg))
    return math.degrees(np.minimum(2 * math.pi - dist_phi, dist_phi))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Body orientation accuracy of a PedRecNet on the TUD multiview "
                                                 "pedestrians test set.")
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

    train_data = get_xml_data(os.path.join(experiment_paths.tud_dir, "viewpoints_test.al"))
    cfg = PedRecNet50Config()
    app_cfg = AppConfig()
    device = get_device(not args.cpu)

    # Pose
    pose_cfg = PedRecNet50Config()
    net = load_pedrec_net(pedrecnet_weights, device)
    corrects = 0
    distances = []
    bin_gt = {}
    bin_pred = {}
    for i in range(1, 9):
        bin_gt[i] = []
        bin_pred[i] = []
    for annotation in train_data:
        img = cv2.cvtColor(cv2.imread(annotation["img_path"]), cv2.COLOR_BGR2RGB)
        center, scale = bb_to_center_scale(annotation["bb"], cfg.model.input_size)
        rotation = 0
        trans, trans_inv = get_affine_transforms(center, scale, rotation, cfg.model.input_size, add_inv=True)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            img,
            trans,
            (int(cfg.model.input_size.width), int(cfg.model.input_size.height)),
            flags=cv2.INTER_LINEAR)

        model_input = pose_transform(model_input).unsqueeze(0).to(device)
        trans_invs = torch.tensor(trans_inv, dtype=torch.float32).unsqueeze(0).to(device)
        img_size = torch.tensor([[img.shape[1], img.shape[0]]], dtype=torch.float32).to(device)

        output = net(model_input)
        orientation_pred = output[2].cpu().detach().numpy()
        orientation_pred[:, :, 0] *= math.pi
        orientation_pred[:, :, 1] *= 2 * math.pi
        body_phi_degree = math.degrees(orientation_pred[0, 0, 1])
        if body_phi_degree > 360:
            print(body_phi_degree)
        body_phi_degree = min(math.degrees(orientation_pred[0, 0, 1]), 360)
        orientation_bin = orientation_bin_from_orientation(body_phi_degree)
        distances.append(get_angular_distance(annotation["orientation"], body_phi_degree))
        corrects += int(orientation_bin == annotation["orientation_bin"])
        for i in range (1, 9):
            gt = int(i == annotation["orientation_bin"])
            pred = int(i == orientation_bin)
            bin_gt[annotation["orientation_bin"]].append(gt)
            bin_pred[annotation["orientation_bin"]].append(pred)

    distances = np.array(distances, dtype=np.float32)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    acc22_5 = len(np.where(distances <= 22.5)[0]) / len(train_data)
    acc45 = len(np.where(distances <= 45)[0]) / len(train_data)
    print(f"Correct: {corrects}, total: {len(train_data)}, percentage: {corrects / len(train_data)}, Acc22_5: {acc22_5}, Acc45: {acc45} mean: {mean_distance}, std: {std_distance}")


if __name__ == "__main__":
    main()
