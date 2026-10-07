import copy
import os
from random import random
from typing import Callable, List

import cv2
import numpy as np
from torch.utils.data import Dataset

from pedrec.configs.dataset_configs import PedRecDatasetConfig, get_h36m_dataset_cfg_default, \
    get_sim_dataset_cfg_default
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.datasets.augmentations import half_body_center_scale, color_jitter, random_erasing
from pedrec.datasets.dataset_helper import get_skeleton_2d_affine_transform
from pedrec.datasets.pedrec_df_loader import get_annotations_from_pedrec_df
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.data_structures import ImageSize
from pedrec.utils.augmentation_helper import get_affine_transform, get_affine_transforms
from pedrec.utils.skeleton_helper import flip_lr_joints
from pedrec.utils.skeleton_helper_3d import flip_lr_orientation, flip_lr_joints_3d


class PedRecDataset(Dataset):
    def __init__(self, dataset_root_path: str, dataset_filename: str, mode: DatasetType, cfg: PedRecDatasetConfig,
                 model_input_size: ImageSize, transform: Callable,  action_filters: List[str] = None,
                 is_h36m: bool = False, flip_all: bool = False):
        self.flip_all = flip_all
        self.action_filter = action_filters
        self.mode = mode
        self.cfg = cfg
        self.half_skeleton_range = cfg.skeleton_3d_range / 2
        self.model_input_size = model_input_size
        self.dataset_root_path = dataset_root_path
        self.dataset_hdf5_path = os.path.join(self.dataset_root_path, dataset_filename)
        self.orientation_dummy = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
        self.transform = transform
        self.index_mappings, self.annotations, self.info = get_annotations_from_pedrec_df(os.path.join(dataset_root_path, dataset_filename),
                                                                                          cfg,
                                                                                          self.model_input_size,
                                                                                          None,
                                                                                          self.action_filter,
                                                                                          is_h36m=is_h36m)
        self.index_mappings = list(self.index_mappings)
        self.__length = len(self.index_mappings)

    def load_image(self, img_path: str):
        # PoseResnet pretrain was trained on BGR, thus keep it for now
        # img = cv2.imread(os.path.join(self.dataset_root_path, img_path))
        img = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_root_path, img_path)), cv2.COLOR_BGR2RGB)
        return img

    def __len__(self):
        return self.__length

    def get_augmented_img(self, img: np.ndarray, trans, flip: bool):
        if flip:
            img = img[:, ::-1, :]
        return cv2.warpAffine(
            img,
            trans,
            (int(self.model_input_size.width), int(self.model_input_size.height)),
            flags=cv2.INTER_LINEAR)

    def get_augmented_skeleton(self, skeleton: np.ndarray, img_width: int, trans, flip: bool):
        if flip:
            skeleton = flip_lr_joints(skeleton, img_width)
        return get_skeleton_2d_affine_transform(skeleton, trans, self.model_input_size)

    def get_augmented_center(self, center: np.ndarray, img_width: int, flip: bool):
        if flip:
            center[0] = img_width - center[0] - 1
        return center

    def get_augmented_skeleton_3d(self, skeleton_3d: np.ndarray, flip: bool):
        if flip:
            skeleton_3d = flip_lr_joints_3d(skeleton_3d)
        return skeleton_3d

    def get_augmented_orientation(self, orientation: np.ndarray, flip: bool, rotation: int):
        if not self.info.provides_body_orientations or not self.info.provides_head_orientations:
            return orientation
        if flip:
            orientation = flip_lr_orientation(orientation)
        if rotation:
            raise NotImplementedError("Rotation is currently not supported for data with orientation gt")
        return orientation

    def normalize_skeleton(self, skeleton: np.ndarray):
        offset = 1 if getattr(self.cfg, "udp", False) else 0  # UDP: pixel centers, normalize by (size - 1)
        skeleton[:, 0] /= self.model_input_size.width - offset
        skeleton[:, 1] /= self.model_input_size.height - offset

    def normalize_skeleton_3d(self, skeleton_3d: np.ndarray):
        skeleton_3d[:, :3] += self.half_skeleton_range  # move negatives to positive, scale 0-2
        skeleton_3d[:, :3] /= self.cfg.skeleton_3d_range  # normalize to 0-1

    def __getitem__(self, index_in):
        index = self.index_mappings[index_in]
        annotations_orig = self.annotations.get_entry(index, self.cfg)
        annotations = copy.deepcopy(annotations_orig)
        img = self.load_image(annotations.img_path)
        img_orig = img.copy()
        img_size = np.array([img.shape[1], img.shape[0]], dtype=np.float32)
        flip = self.flip_all
        rotation = 0

        udp = getattr(self.cfg, "udp", False)
        if self.mode == DatasetType.TRAIN:
            """
            Randomize augmentation values, TODO: rotation support 4 orientation...
            """
            if getattr(self.cfg, "half_body_prob", 0) > 0 and random() < self.cfg.half_body_prob:
                half_body = half_body_center_scale(annotations.skeleton_2d, self.model_input_size)
                if half_body is not None:
                    annotations.center, annotations.scale = half_body
            annotations.scale = annotations.scale * np.clip(np.random.randn() * self.cfg.scale_factor + 1,
                                    1 - self.cfg.scale_factor,
                                    1 + self.cfg.scale_factor)
            flip = self.flip_all or (self.cfg.flip and random() <= 0.5)
            if self.cfg.rotation_factor > 0 and random() <= 0.6:
                rotation = np.clip(np.random.randn() * self.cfg.rotation_factor,
                                   -self.cfg.rotation_factor * 2,
                                   self.cfg.rotation_factor * 2)

        annotations.center = self.get_augmented_center(annotations.center, img_size[0], flip)
        trans, trans_inv = get_affine_transforms(annotations.center, annotations.scale, rotation, self.model_input_size, add_inv=True, udp=udp)
        annotations.skeleton_2d = self.get_augmented_skeleton(annotations.skeleton_2d, img_size[0], trans, flip)

        if np.max(annotations.skeleton_2d[:, 2]) == 0:  # augmentation screwed up, use unaugmented data
            annotations = copy.deepcopy(annotations_orig)
            img = img_orig.copy()
            rotation = 0
            flip = False
            trans, trans_inv = get_affine_transforms(annotations.center, annotations.scale, rotation, self.model_input_size, add_inv=True, udp=udp)
            annotations.skeleton_2d = self.get_augmented_skeleton(annotations.skeleton_2d, img_size[0], trans, flip)

        model_input = self.get_augmented_img(img, trans, flip)
        if self.mode == DatasetType.TRAIN:
            model_input = color_jitter(model_input, getattr(self.cfg, "color_jitter", 0))
        annotations.skeleton_3d = self.get_augmented_skeleton_3d(annotations.skeleton_3d, flip)
        annotations.body_orientation = self.get_augmented_orientation(annotations.body_orientation, flip, rotation)
        annotations.head_orientation = self.get_augmented_orientation(annotations.head_orientation, flip, rotation)
        env_position_2d = np.array([(annotations.env_position[0] + 5000) / 10000,
                                    (annotations.env_position[2] + 1000) / 30000,
                                    1], dtype=np.float32)
        self.normalize_skeleton(annotations.skeleton_2d)
        self.normalize_skeleton_3d(annotations.skeleton_3d)
        # if annotations.skeleton_3d[:, :3].any() > 1 or annotations.skeleton_3d[:, :3].any() < 0:
        #     raise ValueError("Normalization error 3D")

        if self.transform:
            model_input = self.transform(model_input)
            if self.mode == DatasetType.TRAIN:
                model_input = random_erasing(model_input, getattr(self.cfg, "random_erasing_prob", 0))

        orientation = np.zeros((2, 5), dtype=np.float32)
        orientation[0, :4] = annotations.body_orientation
        orientation[1, :4] = annotations.head_orientation
        orientation[:, 4] = orientation[:, 3]  # if supported == theta and phi are supported
        # orientation = np.concatenate((annotations.body_orientation, annotations.head_orientation), axis=0)
        is_real_img = np.array([annotations.is_real_img], dtype=np.float32)
        return model_input, {
            "skeleton": annotations.skeleton_2d,
            "skeleton_3d": annotations.skeleton_3d,
            "center": annotations.center,
            "scale": annotations.scale,
            "rotation": rotation,
            "is_real_img": is_real_img,
            "orientation": orientation,
            "img_path": annotations.img_path,
            "env_position_2d": env_position_2d,
            "trans_inv": trans_inv.astype(np.float32),
            "idx": index,
            "img_size": img_size
        }

#
if __name__ == "__main__":
    cfg = PedRecNet50Config()
    dataset_cfg = get_sim_dataset_cfg_default()
    dataset_root = "data/datasets/Human3.6m/train/"
    dataset_df_filename = "h36m_train_pedrec.pkl"
    dataset = PedRecDataset(dataset_root, dataset_df_filename, DatasetType.TRAIN, dataset_cfg, cfg.model.input_size,
                            None, is_h36m=True)
    # dataset_root = "data/datasets/Conti01/"
    # dataset_df_filename = "rt_conti_01_val_FIN.pkl"
    # dataset = PedRecDataset(dataset_root, dataset_df_filename, DatasetType.VALIDATE, dataset_cfg, cfg.model.input_size,
    #                         None)



    entry, labels = dataset[154252]
    x = 1
    # start = time.time()
    # for i in range(0, 1000):
    #     a = dataset[i]
    # end = time.time()
    # print(f"duration = {end-start}")
    b = 1