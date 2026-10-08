import math
import os
from random import random
from typing import List, Dict

import cv2
import json
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from pedrec.configs.dataset_configs import CocoDatasetConfig, get_coco_dataset_cfg_default
from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.datasets.augmentations import half_body_center_scale, color_jitter, random_erasing
from pedrec.datasets.dataset_helper import get_skeleton_2d_affine_transform
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.models.data_structures import ImageSize
from pedrec.utils.augmentation_helper import get_affine_transforms, get_normalization_size
from pedrec.utils.bb_helper import get_center_bb_from_tl_bb, bb_to_center_scale
from pedrec.utils.skeleton_helper import flip_lr_joints
from pedrec.utils.skeleton_helper_3d import flip_lr_orientation


class CocoDataset(Dataset):
    def __init__(self, dataset_path: str, mode: DatasetType, cfg: CocoDatasetConfig,
                 input_size: ImageSize, transform, flip_all: bool = False):
        self.mode = mode
        self.flip_all: bool = flip_all
        self.cfg = cfg
        self.input_size = input_size
        self.dataset_path = dataset_path
        self.annotation_path = os.path.join(dataset_path, "annotations")
        if mode == DatasetType.TRAIN:
            self.annotation_path = os.path.join(self.annotation_path, "person_keypoints_train2017.json")
            self.img_dir = os.path.join(self.dataset_path, 'train2017')
        elif mode == DatasetType.VALIDATE:
            self.annotation_path = os.path.join(self.annotation_path, "person_keypoints_val2017.json")
            self.img_dir = os.path.join(self.dataset_path, 'val2017')
        else:
            self.annotation_path = None
            self.img_dir = os.path.join(self.dataset_path, 'test2017')

        self.mhbow_gt = self.get_mhbow_gt()
        self.coco = COCO(self.annotation_path)
        coco_classes = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]

        self.transform = transform
        self.classes = ['__background__'] + coco_classes
        self.num_classes = len(self.classes)
        self.num_non_humans = 0
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(coco_classes, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls],
                                             self._class_to_ind[cls])
                                            for cls in self.classes[1:]])
        self.num_joints = len(SKELETON_PEDREC_JOINTS)
        self.orientation_dummy = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
        self.no_human_joint_dummy = np.zeros((self.num_joints, 5), dtype=np.float32)
        self.no_human_joint_dummy[4] = 1
        self.threed_dummy = np.zeros((self.num_joints, 6), dtype=np.float32)
        self.env_position_dummy = np.array([0, 0, 0], dtype=np.float32)
        # self.threed_not_available_dummy = np.zeros((self.num_joints, 1), dtype=np.float32)
        self.is_real_img = np.ones((1), dtype=np.float32)
        self.load_gt()
        self.__length = len(self.annotations)

    def get_mhbow_gt(self):
        if self.mode == DatasetType.TRAIN:
            annot_path = os.path.join(self.dataset_path, "annotations", "train_hoe.json")
        else:
            annot_path = os.path.join(self.dataset_path, "annotations", "val_hoe.json")

        with open(annot_path, 'r') as obj_file:
            annot = json.load(obj_file)

        return annot

    def load_gt(self):
        self.annotations = []
        img_ids = self.coco.getImgIds()
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_size = np.array([img_info['width'], img_info['height']], dtype=np.float32)
            annotation_ids = self.coco.getAnnIds(img_id, iscrowd=False)
            annotations = self.coco.loadAnns(annotation_ids)
            person_annotations, non_person_annotations = self.get_clean_bb_annotations(annotations, img_size)
            for annotation in person_annotations:
                orientation = self.orientation_dummy

                if self.cfg.use_mebow_orientation:
                    mhbow_key = f"{img_id}_{annotation['id']}"
                    if mhbow_key in self.mhbow_gt:
                        orientation = self.orientation_dummy.copy()
                        degrees = self.mhbow_gt[mhbow_key] + 90
                        if degrees < 0:
                            degrees += 360
                        if degrees > 360:
                            degrees -= 360
                        orientation[0, 1] = math.radians(degrees) / (2 * math.pi)
                        orientation[0, 4] = 1
                # mhbow_orientation =
                filename = "{}.jpg".format(str(annotation["image_id"]).zfill(12))
                bb = annotation['clean_bb']
                bb = get_center_bb_from_tl_bb(bb)
                joints = self.get_joint_data(annotation)
                center, scale = bb_to_center_scale(bb, self.input_size)
                self.annotations.append({
                    "coco_id": annotation['id'],
                    "img_filename": filename,
                    "joints": joints,
                    "orientation": orientation,
                    "bb": bb,
                    "center": center,
                    "scale": scale,
                    "img_size": img_size,
                    "is_human": True
                })
            # for annotation in non_person_annotations:
            #     if (self.mode == DatasetType.TRAIN and self.num_non_humans > 50000) or \
            #             (self.mode == DatasetType.VALIDATE and self.num_non_humans > 10000):
            #         continue
            #     self.num_non_humans += 1
            #     filename = "{}.jpg".format(str(annotation["image_id"]).zfill(12))
            #     bb = annotation['clean_bb']
            #     bb = get_center_bb_from_tl_bb(bb)
            #     joints = self.get_joint_data(annotation)
            #     center, scale = bb_to_center_scale(bb, self.input_size)
            #     self.annotations.append({
            #         "img_filename": filename,
            #         "joints": joints,
            #         "bb": bb,
            #         "center": center,
            #         "scale": scale,
            #         "img_size": img_size,
            #         "is_human": False
            #     })

    def get_joint_data(self, annotation: Dict[str, any]):
        joint_data = annotation["keypoints"]
        joints = np.zeros((len(SKELETON_PEDREC_JOINTS), 5), dtype=np.float32)
        for joint_num in range(17):
            curr_idx = joint_num * 3
            joints[joint_num][0] = joint_data[curr_idx]
            joints[joint_num][1] = joint_data[curr_idx + 1]
            joints[joint_num][2] = 1
            visibility = joint_data[curr_idx + 2]
            if visibility == 0:
                joints[joint_num][0] = 0
                joints[joint_num][1] = 0
                joints[joint_num][2] = 0  # score
                joints[joint_num][3] = 0  # visibility
                joints[joint_num][4] = 1  # joint supported by dataset
            else:
                joints[joint_num][2] = 1  # score
                joints[joint_num][3] = 1  # visibility
                joints[joint_num][4] = 1  # joint supported by dataset
        return joints

    def get_clean_bb_annotations(self, annotations: List[Dict[str, any]], img_size: np.array):
        person_annotations = []
        non_person_annotations = []
        for annotation in annotations:
            x, y, w, h = annotation['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((img_size[0] - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((img_size[1] - 1, y1 + np.max((0, h - 1))))
            if x2 >= x1 and y2 >= y1:
                annotation['clean_bb'] = [x1, y1, x2 - x1, y2 - y1]
                if not self.is_valid_person_annotation(annotation):
                    non_person_annotations.append(annotation)
                    continue
                person_annotations.append(annotation)
            else:
                print("WAHHH")
        return person_annotations, non_person_annotations

    def is_valid_person_annotation(self, annotation: Dict[str, any]):
        cls = self._coco_ind_to_class_ind[annotation['category_id']]
        if cls != 1:
            return False
        if max(annotation['keypoints']) == 0:
            return False
        if annotation['area'] <= 0:
            return False
        return True

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        annotations = self.annotations[index]
        img_path = os.path.join(self.img_dir, annotations['img_filename'])
        # PoseResnet pretrain was trained on BGR, thus keep it for now
        # img = cv2.imread(img_path)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_orig = img.copy()
        orientation = annotations['orientation'].copy()
        skeleton = annotations['joints'].copy()

        center = annotations['center'].copy()
        scale = annotations['scale'].copy()
        rotation = 0

        if self.flip_all:
            img = img[:, ::-1, :]
            skeleton = flip_lr_joints(skeleton, img.shape[1])
            center[0] = img.shape[1] - center[0] - 1
            orientation = flip_lr_orientation(orientation)

        if self.mode == DatasetType.TRAIN:
            if random() < self.cfg.half_body_prob:
                half_body = half_body_center_scale(skeleton, self.input_size)
                if half_body is not None:
                    center, scale = half_body
            scale = scale * np.clip(np.random.randn() * self.cfg.scale_factor + 1,
                                    1 - self.cfg.scale_factor,
                                    1 + self.cfg.scale_factor)
            if random() <= 0.6:
                rotation = np.clip(np.random.randn() * self.cfg.rotation_factor,
                                   -self.cfg.rotation_factor * 2,
                                   self.cfg.rotation_factor * 2)
            if self.cfg.flip and random() <= 0.5:
                img = img[:, ::-1, :]
                skeleton = flip_lr_joints(skeleton, img.shape[1])
                center[0] = img.shape[1] - center[0] - 1
                orientation = flip_lr_orientation(orientation)
        trans, trans_inv = get_affine_transforms(center, scale, rotation, self.input_size, add_inv=True)
        # transx = get_affine_transform(center, scale, rotation, self.input_size)
        skeleton = get_skeleton_2d_affine_transform(skeleton, trans, self.input_size)
        if np.max(skeleton[:, 2]) == 0:  # augmentation screwed up, use unaugmented
            img = img_orig.copy()
            center = annotations['center'].copy()
            scale = annotations['scale'].copy()
            rotation = 0
            skeleton = annotations['joints'].copy()
            trans, trans_inv = get_affine_transforms(center, scale, rotation, self.input_size, add_inv=True)
            # trans = get_affine_transform(center, scale, rotation, self.input_size)
            skeleton = get_skeleton_2d_affine_transform(skeleton, trans, self.input_size)

        model_input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size.width), int(self.input_size.height)),
            flags=cv2.INTER_LINEAR)

        # if np.max(skeleton[:, 2]) == 0:  # still screwed
        # print(f"No visible skeleton COCO: {index} - {annotations['img_filename']}")

        if self.mode == DatasetType.TRAIN:
            model_input = color_jitter(model_input, self.cfg.color_jitter)
        skeleton[:, :2] /= get_normalization_size(self.input_size)

        if self.transform:
            model_input = self.transform(model_input)
            if self.mode == DatasetType.TRAIN:
                model_input = random_erasing(model_input, self.cfg.random_erasing_prob)

        if np.max(skeleton) > 1:
            raise ValueError("WTF")

        # skeleton = np.concatenate((skeleton[:, 0:2], self.threed_dummy, skeleton[:, 2:], self.threed_dummy), axis=1)
        # np.insert(skeleton, 2, 0, axis=0)

        # returns model_input, skeleton, center, scale, rotation, is_real_img
        return model_input, {
            "skeleton": skeleton,
            "skeleton_3d": self.threed_dummy,
            "center": center,
            "scale": scale,
            "rotation": rotation,
            "is_real_img": self.is_real_img,
            "orientation": orientation,
            "env_position_2d": self.env_position_dummy,
            "trans_inv": trans_inv.astype(np.float32),
            "img_path": img_path,
            "idx": index,
            "img_size": annotations["img_size"],
            # "coco_id": annotations["coco_id"]
        }

if __name__ == "__main__":
    cfg = PedRecNetConfig()
    dataset_cfg = get_coco_dataset_cfg_default()

    # MS Coco
    dataset_name = "MSCOCO"
    dataset = CocoDataset("data/datasets/COCO", DatasetType.VALIDATE, dataset_cfg,
                          cfg.model.input_size, None)

    test = dataset[49]
    a = 1
