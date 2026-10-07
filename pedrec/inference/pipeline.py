"""
Framework independent PedRec inference pipeline.

The pipeline chains the individual components (all of them optional):

    YoloV4 detector -> PedRecNet (2D / 3D pose, orientation, joint confidence) -> tracking -> EHPI3D action recognition

It is used by the Qt demo (``pedrec/demo.py``), the headless runner and the dataset tools, so every component can be
used on its own (e.g. only the detector, or the pose net on full frames without a detector).
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from pedrec.configs import default_paths
from pedrec.configs.app_config import AppConfig
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.configs.yolo_v4_config import YoloV4Config
from pedrec.models.constants.action_mappings import ACTION
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.networks.net_pedrec.ehpi_3d_net import Ehpi3DNet
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.networks.net_yolo_v4.yolo_v4_helper import do_detect
from pedrec.tracking.human_merger import HumanMerger
from pedrec.tracking.human_tracker import HumanTracker, bb_tracking, add_undetected_bbs_from_tracking, \
    remove_duplicates
from pedrec.utils.bb_helper import split_human_bbs, get_bbs_above_score
from pedrec.utils.demo_helper import get_detector, init_pose_model
from pedrec.utils.ehpi_helper import get_ehpi_from_human_history, ehpi_transform
from pedrec.utils.human_helper import get_humans_from_pedrec_detections
from pedrec.utils.image_content_buffer import ImageContent, ImageContentBuffer
from pedrec.utils.pose_deconv_helper import pedrec_recognizer, do_redetect_pose_recognition
from pedrec.utils.time_helper import timed

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Which components run and where their weights are. ``None`` weight paths resolve to the defaults below the data
    root (``PEDREC_DATA_DIR``).
    """
    use_detector: bool = True  # YoloV4 human / object detection; if False the full frame is used as human bb
    use_pose: bool = True  # PedRecNet 2D / 3D pose + orientation + joint confidence
    use_tracking: bool = True  # optical flow tracking + id assignment (requires pose)
    use_action: bool = True  # EHPI3D action recognition (requires pose + tracking)

    yolo_weights: Optional[str] = None
    pedrec_weights: Optional[str] = None
    ehpi3d_weights: Optional[str] = None
    data_root: Optional[str] = None

    detector_conf_thresh: float = 0.4
    detector_nms_thresh: float = 0.6
    human_bb_min_score: float = 0.6
    human_min_score: float = 0.65  # humans with a lower mean joint score are dropped
    action_thresh: float = 0.7
    num_smoothing_frames: int = 2
    temporal_field: ImageSize = field(default_factory=lambda: ImageSize(width=64, height=32))

    def validate(self):
        if self.use_action and not (self.use_pose and self.use_tracking):
            raise ValueError("Action recognition requires pose estimation and tracking.")
        if self.use_tracking and not self.use_pose:
            raise ValueError("Tracking requires pose estimation.")


@dataclass
class FrameResult:
    frame_nr: int
    humans: List[Human]
    objects: List[np.ndarray]
    fps: int
    timings: Dict[str, float] = field(default_factory=dict)


def get_action_classes(action_probabilities: np.ndarray, action_list: List[ACTION], thresh: float = 0.7) -> List[ACTION]:
    return [action for probability, action in zip(action_probabilities, action_list) if probability > thresh]


def get_full_frame_bb(img_size: ImageSize) -> np.ndarray:
    """Center bb (center_x, center_y, width, height, confidence, class_idx) covering the full frame."""
    return np.array([img_size.width / 2, img_size.height / 2, img_size.width, img_size.height, 1.0, 0],
                    dtype=np.float32)


class PedRecPipeline:
    def __init__(self, cfg: PipelineConfig, app_cfg: AppConfig, device: torch.device):
        cfg.validate()
        self.cfg = cfg
        self.app_cfg = app_cfg
        self.device = device
        self.img_size: ImageSize = app_cfg.inference.img_size

        self.detector = None
        self.yolo_cfg = YoloV4Config()
        if cfg.use_detector:
            weights = cfg.yolo_weights or default_paths.yolo_v4_weights(cfg.data_root)
            self.detector = get_detector(self.yolo_cfg, weights, logger, device)

        self.pose_recognizer = None
        self.pose_cfg = PedRecNet50Config()
        if cfg.use_pose:
            weights = cfg.pedrec_weights or default_paths.pedrec_net_weights(cfg.data_root)
            self.pose_recognizer = init_pose_model(PedRecNet(self.pose_cfg), weights, logger, device)

        self.action_recognizer = None
        if cfg.use_action:
            weights = cfg.ehpi3d_weights or default_paths.ehpi3d_weights(cfg.data_root)
            self.action_recognizer = Ehpi3DNet(len(app_cfg.inference.action_list)).to(device)
            self.action_recognizer.load_state_dict(torch.load(weights, map_location=device))
            self.action_recognizer.eval()
            logger.info(f"Loaded EHPI3D (weights: {weights})")

        self.human_tracker = HumanTracker(img_size=self.img_size) if cfg.use_tracking else None
        self.human_merger = HumanMerger(self.img_size) if cfg.use_tracking else None
        self.image_content_buffer = ImageContentBuffer(buffer_size=app_cfg.inference.buffer_size)

    # ------------------------------------------------------------------------------------------------------ components
    def detect(self, img: np.ndarray, tracked_humans: List[Human]):
        """YoloV4 detection, returns (human bbs, other object bbs) in image coordinates."""
        if self.detector is None:
            human_bbs, other_bbs = [get_full_frame_bb(self.img_size)], []
        else:
            sized = cv2.resize(img, (self.yolo_cfg.model.input_size.width, self.yolo_cfg.model.input_size.height))
            bbs = do_detect(self.detector, sized, self.img_size, self.cfg.detector_conf_thresh,
                            self.cfg.detector_nms_thresh, self.device, logger, tracked_humans)
            human_bbs, other_bbs = split_human_bbs(bbs[0])
        if self.cfg.use_tracking:
            human_bbs = bb_tracking(human_bbs, tracked_humans)
            human_bbs = add_undetected_bbs_from_tracking(human_bbs, tracked_humans)
        human_bbs = get_bbs_above_score(human_bbs, self.cfg.human_bb_min_score)
        human_bbs = remove_duplicates(human_bbs)
        return human_bbs, other_bbs

    def estimate_poses(self, img: np.ndarray, human_bbs: List[np.ndarray]) -> List[Human]:
        """PedRecNet on the given human bbs."""
        pose_preds = pedrec_recognizer(self.pose_recognizer, self.pose_cfg, img, human_bbs, self.device)
        return get_humans_from_pedrec_detections(human_bbs, pose_preds)

    def track(self, img: np.ndarray, humans: List[Human], tracked_humans: List[Human]) -> List[Human]:
        humans, undetected_humans = self.human_merger.merge_humans(humans, tracked_humans, assign_new_ids=True)
        redetected_humans = do_redetect_pose_recognition(self.pose_recognizer, self.pose_cfg, img, undetected_humans,
                                                         self.device)
        humans.extend(redetected_humans)
        return humans

    def recognize_actions(self, humans: List[Human]):
        ehpis = []
        for human in humans:
            history = self.image_content_buffer.get_human_data_buffer_by_id(human.uid)
            self.smooth(human, history)
            human.ehpi = get_ehpi_from_human_history(human, history, temporal_field=self.cfg.temporal_field)
            ehpis.append(ehpi_transform(human.ehpi))
        if len(ehpis) == 0:
            return
        ehpis = torch.stack(ehpis).to(self.device)
        with torch.no_grad():
            action_probabilities = torch.sigmoid(self.action_recognizer(ehpis)).cpu().numpy()
        for human, action_probs in zip(humans, action_probabilities):
            human.action_probabilities = action_probs
            human.actions = get_action_classes(action_probs, self.app_cfg.inference.action_list,
                                               thresh=self.cfg.action_thresh)

    def smooth(self, human: Human, history: List[Human]):
        num_smoothing = self.cfg.num_smoothing_frames
        if num_smoothing > 1 and len(history) >= num_smoothing:
            for i in range(1, num_smoothing):
                human.skeleton_3d += history[-i].skeleton_3d
                human.orientation += history[-i].orientation
            human.orientation /= num_smoothing
            human.skeleton_3d /= num_smoothing

    # ------------------------------------------------------------------------------------------------------ pipeline
    def process(self, frame_nr: int, img: np.ndarray) -> FrameResult:
        """
        Runs the enabled components on one RGB frame of size ``app_cfg.inference.img_size``.
        """
        start = time.time()
        timings: Dict[str, float] = {}
        tracked_humans: List[Human] = []
        if self.human_tracker is not None:
            last_humans = self.image_content_buffer.get_last_humans()
            timings["tracking"], tracked_humans = timed(
                lambda: self.human_tracker.get_humans_by_tracking(img, previous_humans=last_humans))

        timings["detection"], (human_bbs, other_bbs) = timed(lambda: self.detect(img, tracked_humans))

        humans: List[Human] = []
        if self.cfg.use_pose:
            timings["pose"], humans = timed(lambda: self.estimate_poses(img, human_bbs))
            if self.cfg.use_tracking:
                humans = self.track(img, humans, tracked_humans)
            humans = [human for human in humans if human.score > self.cfg.human_min_score]
        else:
            empty_skeleton = np.zeros((self.pose_cfg.model.num_joints, 3), dtype=np.float32)
            humans = [Human(bb=bb, skeleton_2d=empty_skeleton.copy(), skeleton_3d=None, orientation=None, uid=-1)
                      for bb in human_bbs]

        if self.cfg.use_action:
            timings["action"], _ = timed(lambda: self.recognize_actions(humans))

        self.image_content_buffer.add(ImageContent(humans=humans, objects=other_bbs))

        required_time = time.time() - start
        timings["total"] = required_time
        fps = int(1.0 / required_time) if required_time > 0 else 0
        return FrameResult(frame_nr=frame_nr, humans=humans, objects=other_bbs, fps=fps, timings=timings)
