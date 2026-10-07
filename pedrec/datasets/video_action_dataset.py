import copy
import math
import os
import random
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from pedrec.configs.app_config import AppConfig
from pedrec.configs.dataset_configs import VideoActionDatasetConfig, get_video_action_dataset_cfg_default
from pedrec.datasets.dataset_helper import get_ehpi_skeleton_sequence_3d
from pedrec.datasets.video_action_df_loader import get_annotations_from_video_action_df
from pedrec.models.constants.action_mappings import ACTION
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.constants.sample_method import SAMPLE_METHOD
from pedrec.models.data_structures import ImageSize


class VideoActionDataset(Dataset):
    def __init__(self, dataset_root_path: str,
                 dataset_filename: str,
                 mode: DatasetType,
                 cfg: VideoActionDatasetConfig,
                 action_list: List[ACTION],
                 transform: Callable,
                 included_folders: List[str] = None,
                 excluded_folders: List[str] = ("SIM", "2019_ITS_Journal_Eval2")
                 ):
        """
        pose_results_file: Filename of results from an algorithm df, should contain pose2d / 3d results with same index
        as gt.
        - use_movement_instead_action: Uses the movement label used in RTSIMs instead of actions, movements are binary, thus it returns only the label for CE loss.
        """
        self.mode = mode
        self.cfg = copy.deepcopy(cfg)
        self.cfg.subsample = 1
        self.cfg.subsampling_strategy = SAMPLE_METHOD.SYSTEMATIC
        self.half_skeleton_range = self.cfg.skeleton_3d_range / 2
        self.dataset_root_path = dataset_root_path
        self.dataset_hdf5_path = os.path.join(self.dataset_root_path, dataset_filename)
        self.transform = transform

        self.annotations = get_annotations_from_video_action_df(os.path.join(dataset_root_path, dataset_filename),
                                                                self.cfg, include_folders=included_folders,
                                                                excluded_folders=excluded_folders)

        self.action_label_mapping = {}
        for action_idx, action in enumerate(action_list):
            self.action_label_mapping[action.value] = action_idx

        self.num_full_dataset = len(self.annotations.img_dirs)
        self.indexes = list(range(0, self.num_full_dataset))
        if cfg.subsample != 1:
            if cfg.subsampling_strategy == SAMPLE_METHOD.SYSTEMATIC:
                self.indexes = range(0, self.num_full_dataset, cfg.subsample)
            elif cfg.subsampling_strategy == SAMPLE_METHOD.RANDOM:
                self.indexes = np.random.choice(self.indexes, math.floor(self.num_full_dataset / cfg.subsample),
                                                replace=False)
            else:
                raise NotImplementedError(f"Sampling strategy {cfg.subsampling_strategy.name} is not implemented.")
        self.__length = len(self.indexes)

    def __len__(self):
        return self.__length

    def __getitem__(self, dataset_idx):
        index = self.indexes[dataset_idx]
        curr_frame_annotations_orig = self.annotations.get_entry(index)
        curr_frame_annotations = copy.deepcopy(curr_frame_annotations_orig)

        skeletons_3d = self.annotations.skeleton3ds
        skeletons_2d = self.annotations.skeleton2ds

        skeleton_sequence_3d, valid = get_ehpi_skeleton_sequence_3d(index=index,
                                                                    temporal_field=self.cfg.temporal_field,
                                                                    curr_frame_annotations=curr_frame_annotations,
                                                                    skeletons_3d=skeletons_3d,
                                                                    skeletons_2d=skeletons_2d,
                                                                    mode=self.mode,
                                                                    flip=self.cfg.flip,
                                                                    half_skeleton_range=self.half_skeleton_range,
                                                                    skeleton_3d_range=self.cfg.skeleton_3d_range,
                                                                    use_2d=self.cfg.add_2d,
                                                                    min_score=self.cfg.min_joint_score,
                                                                    use_unit_skeleton=self.cfg.use_unit_skeleton,
                                                                    frame_sampling=self.cfg.frame_sampling)

        label = np.zeros(len(self.action_label_mapping), dtype=np.float32)
        if valid:
            for action in curr_frame_annotations.actions:
                label[self.action_label_mapping[action]] = 1
            if ACTION.WAVE_CAR_OUT in curr_frame_annotations.actions or\
                    ACTION.IDLE in curr_frame_annotations.actions:
                # wave car out is always standing / idle in tue currently used data
                label[self.action_label_mapping[ACTION.STAND.value]] = 1

        ehpi = skeleton_sequence_3d * 255
        ehpi = ehpi.astype(np.uint8)

        if self.transform:
            ehpi = self.transform(ehpi)

        return ehpi, label


if __name__ == "__main__":
    app_cfg = AppConfig()
    dataset_cfg = get_video_action_dataset_cfg_default()
    dataset_name = "EHPIVIDS"
    dataset_cfg.gt_result_ratio = 0.5
    dataset = VideoActionDataset("data/videos/ehpi_videos/", "pedrec_results.pkl",
                                 DatasetType.TRAIN, dataset_cfg, app_cfg.inference.action_list, None)

    for i in range(0, 5):
        idx = random.randint(0, 10000)
        print(idx)
        ehpi, action = dataset[idx]
        plt.title(f"{idx} - {action}")
        imgplot = plt.imshow(ehpi)
        plt.show()

    # idx = 2529
    # ehpi, action = dataset[idx]
    # imgplot = plt.imshow(ehpi)
    # plt.show()

#     b = 1
