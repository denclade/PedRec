import csv
import math
import os
import xml.etree.ElementTree as ET
from random import random, sample

import cv2
import numpy as np
from torch.utils.data import Dataset

from pedrec.configs.dataset_configs import get_tud_dataset_cfg_default, \
    TudDatasetConfig
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.constants.sample_method import SAMPLE_METHOD
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.models.data_structures import ImageSize
from pedrec.utils.augmentation_helper import get_affine_transforms
from pedrec.utils.bb_helper import bb_to_center_scale, get_center_bb_from_coord_bb
from pedrec.utils.skeleton_helper_3d import flip_lr_orientation
import matplotlib.pyplot as plt

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

class TudDataset(Dataset):
    def __init__(self, dataset_path: str, mode: DatasetType, cfg: TudDatasetConfig,
                 input_size: ImageSize, transform):
        self.mode = mode
        self.cfg = cfg
        self.input_size = input_size
        self.dataset_path = dataset_path
        self.transform = transform

        self.annotations = []
        if mode == DatasetType.TRAIN:
            for i in range(1, 9):
                self.annotations += self.get_xml_data(os.path.join(dataset_path, f"viewpoints_train{i}.al"))
        elif mode == DatasetType.VALIDATE:
            self.annotations += self.get_xml_data(os.path.join(dataset_path, f"viewpoints_validate.al"))
        else:
            self.annotations += self.get_xml_data(os.path.join(dataset_path, f"viewpoints_test.al"))

        if self.cfg.subsample > 1:
            if self.cfg.subsampling_strategy == SAMPLE_METHOD.SYSTEMATIC:
                self.annotations = self.annotations[::self.cfg.subsample]
            else:
                self.annotations = sample(self.annotations, math.floor(len(self.annotations) / self.cfg.subsample))

        self.num_joints = len(SKELETON_PEDREC_JOINTS)

        self.pose2d_dummy = np.zeros((self.num_joints, 5), dtype=np.float32)
        self.pose3d_dummy = np.zeros((self.num_joints, 6), dtype=np.float32)
        self.env_position_dummy = np.array([0, 0, 0], dtype=np.float32)
        # self.threed_not_available_dummy = np.zeros((self.num_joints, 1), dtype=np.float32)
        self.is_real_img = np.ones((1), dtype=np.float32)
        self.__length = len(self.annotations)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        annotations = self.annotations[index]
        img_path = annotations["img_path"]
        # PoseResnet pretrain was trained on BGR, thus keep it for now
        # img = cv2.imread(img_path)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        orientation = annotations['orientations'].copy()

        center = annotations['center'].copy()
        scale = annotations['scale'].copy()
        rotation = 0

        if self.mode == DatasetType.TRAIN:
            scale = scale * np.clip(np.random.randn() * self.cfg.scale_factor + 1,
                                    1 - self.cfg.scale_factor,
                                    1 + self.cfg.scale_factor)
            if self.cfg.flip and random() <= 0.5:
                img = img[:, ::-1, :]
                center[0] = img.shape[1] - center[0] - 1
                orientation = flip_lr_orientation(orientation)
        trans, trans_inv = get_affine_transforms(center, scale, rotation, self.input_size, add_inv=True)

        model_input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size.width), int(self.input_size.height)),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            model_input = self.transform(model_input)

        return model_input, {
            "skeleton": self.pose2d_dummy,
            "skeleton_3d": self.pose3d_dummy,
            "center": center,
            "scale": scale,
            "rotation": rotation,
            "is_real_img": self.is_real_img,
            "orientation": orientation,
            "env_position_2d": self.env_position_dummy,
            "trans_inv": trans_inv.astype(np.float32),
            "img_path": img_path,
            "idx": index,
            "img_size": np.array([img.shape[1], img.shape[0]], dtype=np.float32)
        }

    def get_xml_data(self, path: str):
        tree = ET.parse(path)
        annotationlist = tree.getroot()
        annotations = []
        orientation_degrees = {}
        with open(os.path.join(self.dataset_path, "finalMeanAnnotationTUD.csv")) as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                parts = row[0].split('_')
                if parts[0] == "val":
                    parts[0] = "validate"
                if parts[0] == "train":
                    parts[1] = parts[1].split('.')[0]
                img_file_path = os.path.join(self.dataset_path, parts[0], f"{parts[1]}.png")
                if img_file_path not in orientation_degrees:
                    orientation_degrees[img_file_path] = []
                orientation_degrees[img_file_path].append(float(row[1]))

        for annotation in annotationlist:
            img_path = None
            is_train = False
            for child in annotation:
                annot_num = 0
                if child.tag == "image":
                    img_path = os.path.join(self.dataset_path, child.find("name").text)
                    if "/train/" in img_path:
                        is_train = True
                if child.tag == "annorect":
                    x_1 = int(child.find("x1").text)
                    y_1 = int(child.find("y1").text)
                    x_2 = int(child.find("x2").text)
                    y_2 = int(child.find("y2").text)
                    bb = [x_1, y_1, x_2, y_2, 1, 1]
                    bb = get_center_bb_from_coord_bb(bb)
                    center, scale = bb_to_center_scale(bb, self.input_size)
                    if is_train:
                        orientation_bin = int(path[-4])
                    else:
                        orientation_bin = int(child.find("silhouette").find("id").text)
                    orientation_deg = orientation_degrees[img_path][annot_num]
                    orientation_rad = math.radians(orientation_deg)
                    orientation_normalized = orientation_rad / (2 * math.pi)
                    orientations = np.array([[0, orientation_normalized, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=np.float32)
                    if img_path == None:
                        raise ("WTF")
                    annotations.append({
                        "img_path": img_path,
                        "orientation_bin": orientation_bin,
                        "orientations": orientations,
                        "bb": bb,
                        "center": center,
                        "scale": scale
                    })
                    annot_num += 1
        return annotations

if __name__ == "__main__":
    cfg = PedRecNet50Config()
    dataset_cfg = get_tud_dataset_cfg_default()
    dataset_cfg.subsample = 1
    dataset_cfg.subsampling_strategy = SAMPLE_METHOD

    # MS Coco
    dataset_name = "TUD"
    dataset = TudDataset("data/datasets/cvpr10_multiview_pedestrians/", DatasetType.TRAIN, dataset_cfg,
                          cfg.model.input_size, None)

    print(len(dataset))

    img, labels = dataset[6001]
    imgplot = plt.imshow(img)
    print(labels["orientation"])
    plt.show()
# #     a = 1
