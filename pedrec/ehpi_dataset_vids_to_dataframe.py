import sys

sys.path.append('.')  # allow running as a script from the repository root

import logging
import os
import time
from typing import List

import cv2
import numpy as np
import pandas as pd

from pedrec.configs.app_config import AppConfig
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.configs.yolo_v4_config import YoloV4Config
from pedrec.models.constants.action_mappings import ACTION
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.networks.net_yolo_v4.yolo_v4_helper import do_detect
from pedrec.tracking.human_merger import HumanMerger
from pedrec.tracking.human_tracker import HumanTracker, add_undetected_bbs_from_tracking, bb_tracking, remove_duplicates
from pedrec.utils.bb_helper import split_human_bbs, get_bbs_above_score
from pedrec.utils.demo_helper import get_detector, init_pose_model
from pedrec.utils.human_helper import get_humans_from_pedrec_detections
from pedrec.utils.image_content_buffer import ImageContent, ImageContentBuffer
from pedrec.utils.input_providers.img_dir_provider import ImgDirProvider
from pedrec.utils.input_providers.input_provider_base import InputProviderBase
from pedrec.utils.log_helper import configure_logger
from pedrec.utils.pose_deconv_helper import (pedrec_recognizer, do_redetect_pose_recognition)
from pedrec.utils.time_helper import timed
from pedrec.utils.torch_utils.torch_helper import get_device


def set_df_dtypes(df: pd.DataFrame):
    df["path"] = df["path"].astype("category")
    df["frame"] = df["frame"].astype("int")
    df["uid"] = df["uid"].astype("int")
    df["action"] = df["action"].astype("category")

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
        df["body_orientation_visible"] = df["body_orientation_visible"].astype("category")

        df["head_orientation_phi"] = df["head_orientation_phi"].astype("float32")
        df["head_orientation_theta"] = df["head_orientation_theta"].astype("float32")
        df["head_orientation_score"] = df["head_orientation_score"].astype("float32")
        df["head_orientation_visible"] = df["head_orientation_visible"].astype("category")


def get_column_names():
    column_names = [
        "path",
        "frame",
        "uid",
        "action"
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

    column_names.append("body_orientation_phi")
    column_names.append("body_orientation_theta")
    column_names.append("body_orientation_score")
    column_names.append("body_orientation_visible")

    column_names.append("head_orientation_phi")
    column_names.append("head_orientation_theta")
    column_names.append("head_orientation_score")
    column_names.append("head_orientation_visible")

    return column_names


class PedRecNetDemoWorker(object):
    def __init__(self, input_provider: InputProviderBase, app_cfg: AppConfig,
                 model_file_path: str):
        super().__init__()
        configure_logger()
        self.logger = logging.getLogger(__name__)
        self.app_cfg = app_cfg
        self.device = get_device(self.app_cfg.cuda.use_gpu)

        yolo_weights = "data/models/yolo_v4/yolov4.pth"
        pedrecnet_weights = model_file_path

        # Detector
        self.yolo_cfg = YoloV4Config()
        self.detector = get_detector(self.yolo_cfg, yolo_weights, self.logger, self.device)

        # Pose
        self.pose_cfg = PedRecNet50Config()
        self.pose_recognizer = init_pose_model(PedRecNet(self.pose_cfg), pedrecnet_weights, self.logger, self.device)
        # self.action_list = [ACTION.IDLE,
        #                     ACTION.WALK,
        #                     ACTION.WAVE,
        #                     ACTION.KICK_BALL,
        #                     ACTION.THROW,
        #                     ACTION.LOOK_FOR_TRAFFIC,
        #                     ACTION.HITCHHIKE,
        #                     ACTION.TURN_AROUND,
        #                     ACTION.WORK,
        #                     ACTION.ARGUE,
        #                     ACTION.STUMBLE,
        #                     ACTION.OPEN_DOOR,
        #                     ACTION.FALL,
        #                     ACTION.STAND_UP,
        #                     ACTION.FIGHT]
        # self.movement_recognizer = Ehpi3DNet(15).to(self.device)
        # self.movement_recognizer.load_state_dict(torch.load("data/models/ehpi3d/ehpi3d_c01_test_01.pth"))
        # self.movement_recognizer.eval()
        # Tracking
        self.human_tracker = HumanTracker(img_size=self.app_cfg.inference.img_size)
        self.human_merger = HumanMerger(self.app_cfg.inference.img_size)
        self.image_content_buffer: ImageContentBuffer = ImageContentBuffer(
            buffer_size=self.app_cfg.inference.buffer_size)

        self.input_provider = input_provider
        self.dummy_human = Human(bb=[0, 0, 0, 0, 0, 0],
                                 skeleton_2d=np.zeros((len(SKELETON_PEDREC_JOINTS), 3), dtype=np.float32),
                                 skeleton_3d=np.zeros((len(SKELETON_PEDREC_JOINTS), 4), dtype=np.float32),
                                 orientation=np.array([[0, 0], [0, 0]], dtype=np.float32)
                                 )

    def run_impl(self, vid_path: str, action_label: ACTION):
        result_rows = []
        for frame_nr, img in enumerate(self.input_provider.get_data()):
            start = time.time()
            last_humans = self.image_content_buffer.get_last_humans()
            tracked_humans = self.human_tracker.get_humans_by_tracking(img, previous_humans=last_humans)

            human_bbs, other_bbs = self.get_bbs(img, tracked_humans)

            pose_time, pose_preds = timed(
                lambda: pedrec_recognizer(self.pose_recognizer, self.pose_cfg, img, human_bbs, self.device))
            humans = get_humans_from_pedrec_detections(human_bbs, pose_preds)
            humans = self.get_humans_with_tracking(img, humans, tracked_humans)
            humans = [human for human in humans if human.score > 0.65]  # remove low score humans and unnatural high

            best_human = None
            for human in humans:
                history = self.image_content_buffer.get_human_data_buffer_by_id(human.uid)
                self.smooth_data(human, history)
                if len(history) > 0:
                    best_human = human
                    break
                if best_human is None or best_human.score < human.score:
                    best_human = human
                # human.ehpi = get_ehpi_from_human_history(human, history)
                # ehpis.append(ehpi_transform(human.ehpi))

            orientation_vis_supp = [1, 1]
            if best_human is None:
                best_human = self.dummy_human
                orientation_vis_supp = [0, 0]

            pose2d_pred = best_human.skeleton_2d
            pose3d_pred = best_human.skeleton_3d
            orientation_pred = best_human.orientation
            visibles = (best_human.skeleton_2d[:, 2] > 0.5).astype(np.int32)
            supported = np.ones(best_human.skeleton_2d.shape[0])
            visible_supported = np.array([visibles, supported]).transpose(1, 0)
            pose2d_pred = np.concatenate((pose2d_pred, visible_supported), axis=1)
            pose3d_pred = np.concatenate((pose3d_pred, visible_supported), axis=1)
            pose2d_pred = pose2d_pred.reshape(-1).tolist()
            pose3d_pred = pose3d_pred.reshape(-1).tolist()
            orientation_pred_body = orientation_pred[0].reshape(-1).tolist()
            orientation_pred_head = orientation_pred[1].reshape(-1).tolist()
            result_rows.append(
                [vid_path, frame_nr, best_human.uid,
                 action_label.value] + pose2d_pred + pose3d_pred + orientation_pred_body + orientation_vis_supp + orientation_pred_head + orientation_vis_supp)

            image_content = ImageContent(humans=humans, objects=other_bbs)
            self.image_content_buffer.add(image_content)
        return result_rows

    def get_bbs(self, img: np.ndarray, tracked_humans: List[Human]):
        sized = cv2.resize(img, (self.yolo_cfg.model.input_size.width, self.yolo_cfg.model.input_size.height))
        detection_time, bbs = timed(
            lambda: do_detect(self.detector, sized, self.app_cfg.inference.img_size, 0.4, 0.6, self.device, self.logger,
                              tracked_humans))
        human_bbs, other_bbs = split_human_bbs(bbs[0])
        human_bbs = bb_tracking(human_bbs, tracked_humans)
        human_bbs = add_undetected_bbs_from_tracking(human_bbs, tracked_humans)
        human_bbs = get_bbs_above_score(human_bbs, 0.6)
        human_bbs = remove_duplicates(human_bbs)
        return human_bbs, other_bbs

    def get_humans_with_tracking(self, img: np.ndarray, humans: List[Human], tracked_humans: List[Human]):
        humans, undetected_humans = self.human_merger.merge_humans(humans, tracked_humans, assign_new_ids=True)
        redetected_humans = do_redetect_pose_recognition(self.pose_recognizer, self.pose_cfg, img, undetected_humans,
                                                         self.device)
        humans.extend(redetected_humans)
        return humans

    def smooth_data(self, human: Human, history: List[Human], num_smoothing: int = 2):
        if len(history) >= num_smoothing:
            for i in range(1, num_smoothing):
                human.skeleton_3d += history[-i].skeleton_3d
                human.orientation += history[-i].orientation
            human.orientation /= num_smoothing
            human.skeleton_3d /= num_smoothing


def get_action_from_str(action_str: str):
    if "wave" == action_str:
        return ACTION.WAVE_CAR_OUT
    elif "walk" == action_str:
        return ACTION.WALK
    elif "idle" == action_str:
        return ACTION.IDLE
    elif "sit" == action_str:
        return ACTION.SIT
    elif "jump" == action_str:
        return ACTION.JUMP

    raise ValueError(f"Action '{action_str}' not found.")


def get_dataset_data(src_dir):
    dataset_data = []
    for folder_name in os.listdir(src_dir):
        if folder_name == "2019_ITS_Journal_Eval2":
            continue
        full_path = os.path.join(src_dir, folder_name)
        if not os.path.isdir(full_path):
            continue
        if folder_name == src_dir:
            continue
        is_sim = False
        action_label = None
        if folder_name[0:3] == "SIM":
            is_sim = True
            if "wave" in folder_name:
                action_label = ACTION.WAVE_CAR_OUT
            elif "walk" in folder_name:
                action_label = ACTION.WALK
            elif "idle" in folder_name:
                action_label = ACTION.IDLE
            elif "sit" in folder_name:
                action_label = ACTION.SIT
            elif "jump" in folder_name:
                action_label = ACTION.JUMP
            else:
                continue
        img_dir = os.path.join(full_path, "imgs")
        for vid_folder in os.listdir(img_dir):
            vid_folder_path = os.path.join(img_dir, vid_folder)
            if not is_sim:
                action_label = get_action_from_str(vid_folder.split('_')[0])
            assert action_label is not None
            dataset_data.append((vid_folder_path, action_label))
    return dataset_data


if __name__ == '__main__':
    app_config = AppConfig()
    app_config.inference.img_size = ImageSize(1280, 720)
    model_file_path = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_net.pth"

    src_dir = "data/videos/ehpi_videos/"
    dataset_dirs = get_dataset_data(src_dir)
    result_rows = []
    for dataset_dir in dataset_dirs:
        vid_path, action_label = dataset_dir
        print(f"Working on: {vid_path} ({action_label.name})")
        input_provider = ImgDirProvider(vid_path)
        worker = PedRecNetDemoWorker(input_provider, app_config, model_file_path)
        result_rows += worker.run_impl(vid_path=vid_path.lstrip(src_dir), action_label=action_label)

    print("Create Pandas DF")
    df = pd.DataFrame(data=result_rows, columns=get_column_names())
    print("Set datatypes")
    set_df_dtypes(df)
    print("Save...")
    df.to_pickle("data/videos/ehpi_videos/pedrec_p2d3d_c_o_h36m_sim_mebow_0_results.pkl")
    print("Fin.")
