"""
PedRec inference pipeline:

    RT-DETRv2 detector -> PedRecNet (2D / 3D pose, orientation, joint confidence) -> ByteTrack
    -> temporal 3D lifter (per track, causal) + One Euro smoothing of the orientations -> ST-GCN action recognition

It is used by the Qt demo (``pedrec/demo.py``), the benchmark and the dataset tools. Every stage can be disabled (e.g.
only the detector, or the pose net on the full frame without a detector).

Performance relevant design decisions:

* everything runs under ``torch.inference_mode``
* the frame is uploaded once; detector resize, person crops (``grid_sample``), normalization and the
  back-transformation of the 2D poses run batched on the device (``pedrec.inference.gpu_ops``)
* PedRecNet runs exactly once per frame on all person crops
* optional fp16 autocast, channels_last memory format and ``torch.compile`` (see ``RuntimeConfig``)
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from pedrec.configs import default_paths
from pedrec.configs.app_config import AppConfig
from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.inference import gpu_ops
from pedrec.models.constants.action_mappings import ACTION
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.networks.net_detr.rtdetr_detector import RTDetrDetector
from pedrec.networks.net_pedrec.ehpi_stgcn import EhpiStGcn
from pedrec.networks.net_pedrec.pedrec_net_factory import load_pedrec_net
from pedrec.networks.net_pedrec.pose_lifter import TemporalPoseLifter, lifter_features
from pedrec.tracking.byte_tracker import ByteTracker
from pedrec.tracking.one_euro import HumanStateSmoother
from pedrec.utils.augmentation_helper import get_normalization_size
from pedrec.utils.bb_helper import split_human_bbs, get_bb_score
from pedrec.utils.ehpi_helper import get_ehpi_from_human_history
from pedrec.utils.human_helper import get_humans_from_pedrec_detections
from pedrec.utils.image_content_buffer import ImageContent, ImageContentBuffer
from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file

logger = logging.getLogger(__name__)

EHPI_MEAN = (0.400, 0.443, 0.401)
EHPI_STD = (0.203, 0.269, 0.203)
SKELETON_3D_RANGE = 3000.0  # mm, the 3D pose is predicted normalized in a cube of this size around the hip


@dataclass
class RuntimeConfig:
    """How the networks are executed. None of the options changes the weights."""
    half: bool = False  # fp16 autocast on CUDA (Tensor Cores, ~2x on RTX 30xx-50xx)
    channels_last: bool = False  # NHWC memory format, faster convolutions with Tensor Cores
    compile: bool = False  # torch.compile PedRecNet / ST-GCN (slow first frames, faster afterwards)


@dataclass
class PipelineConfig:
    """
    Which stages run and where their weights are. ``None`` weight paths resolve to the defaults below the data root
    (``PEDREC_DATA_DIR``).
    """
    use_detector: bool = True  # if False the full frame is used as the (single) human bb
    use_pose: bool = True  # PedRecNet 2D / 3D pose + orientation + joint confidence
    use_tracking: bool = True  # ByteTrack id assignment + One Euro smoothing (requires pose)
    use_lifter: bool = True  # temporal 3D lifting over the frames of each track (requires tracking)
    use_action: bool = True  # ST-GCN action recognition (requires pose + tracking)

    rtdetr_model: str = default_paths.RTDETR_MODEL  # Hugging Face id or local directory
    pedrec_weights: Optional[str] = None
    lifter_weights: Optional[str] = None
    ehpi3d_weights: Optional[str] = None
    data_root: Optional[str] = None

    detector_conf_thresh: float = 0.4  # objects (non humans) below this score are dropped
    person_high_thresh: float = 0.6  # ByteTrack: first association stage / new tracks; without tracking: min score
    person_low_thresh: float = 0.1  # ByteTrack: second association stage (continues existing tracks only)
    track_max_lost: int = 30  # frames a lost track can be recovered
    human_min_score: float = 0.65  # humans with a lower mean joint confidence are dropped
    action_thresh: float = 0.7
    temporal_field: ImageSize = field(default_factory=lambda: ImageSize(width=64, height=32))
    source_fps: float = 30.0  # frame rate of the input, used for smoothing and the EHPI temporal sampling
    model_fps: float = 30.0  # frame rate the action model was trained with (SIM-C01: 30 fps)

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self):
        if self.use_action and not (self.use_pose and self.use_tracking):
            raise ValueError("Action recognition requires pose estimation and tracking.")
        if self.use_tracking and not self.use_pose:
            raise ValueError("Tracking requires pose estimation.")
        if self.use_lifter and not self.use_tracking:
            raise ValueError("The temporal 3D lifter requires tracking.")


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


class _Runner:
    """Executes a network with the configured runtime options (autocast, memory format, torch.compile)."""

    def __init__(self, module: torch.nn.Module, device: torch.device, runtime: RuntimeConfig, images: bool = True):
        """:param images: the inputs are images (B x C x H x W), i.e. channels_last applies"""
        module = module.to(device).eval()
        runtime = RuntimeConfig(runtime.half, runtime.channels_last and images, runtime.compile)
        if runtime.channels_last:
            module = module.to(memory_format=torch.channels_last)
        if runtime.compile:
            module = torch.compile(module, dynamic=True)
        self.module = module
        self.device = device
        self.channels_last = runtime.channels_last
        self.autocast = runtime.half and device.type == "cuda"

    def __call__(self, inputs: torch.Tensor):
        if self.channels_last and inputs.dim() == 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.autocast):
            return self.module(inputs)


class PedRecPoseEstimator:
    """PedRecNet on a batch of person bbs of one frame."""

    def __init__(self, weights: str, device: torch.device, runtime: RuntimeConfig,
                 net: Optional[torch.nn.Module] = None):
        """:param net: an already constructed PedRecNet (tests / benchmarks), otherwise loaded from ``weights``"""
        self.device = device
        net = net if net is not None else load_pedrec_net(weights)
        self.input_size = net.cfg.model.input_size
        self.run = _Runner(net, device, runtime)
        self._input_scale = torch.from_numpy(get_normalization_size(self.input_size)).to(device)
        self._orientation_scale = torch.tensor([np.pi, 2 * np.pi], dtype=torch.float32, device=device)

    def __call__(self, frame: torch.Tensor, bbs: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        :return: dict with "skeletons" (B x J x 3: x, y in image pixels, confidence), "skeletons_3d" (B x J x 4:
            x, y, z in mm relative to the hip, confidence) and "orientations" (B x 2 x 2: body / head theta, phi in
            radians)
        """
        num = len(bbs)
        if num == 0:
            return {"skeletons": [], "skeletons_3d": [], "orientations": []}
        _, trans_invs = gpu_ops.get_crop_transforms(bbs, self.input_size)
        trans_invs = torch.from_numpy(trans_invs).to(self.device)
        crops = gpu_ops.crop_affine(frame, trans_invs, self.input_size)
        outputs = self.run(crops)
        pose_2d = outputs[0].float()
        pose_3d = outputs[1].float()
        orientation = outputs[2].float()

        xy = gpu_ops.transform_coords_2d(pose_2d[..., :2] * self._input_scale, trans_invs)
        pose_2d = torch.cat((xy, pose_2d[..., 2:]), dim=-1)
        pose_3d = torch.cat((pose_3d[..., :3] * SKELETON_3D_RANGE - SKELETON_3D_RANGE / 2, pose_3d[..., 3:]), dim=-1)
        orientation = orientation * self._orientation_scale

        joints = pose_2d.shape[1]
        flat = torch.cat((pose_2d.reshape(num, -1), pose_3d.reshape(num, -1), orientation.reshape(num, -1)), dim=1)
        flat = flat.cpu().numpy()  # single device -> host copy
        n2d, n3d = joints * pose_2d.shape[2], joints * pose_3d.shape[2]
        return {
            "skeletons": flat[:, :n2d].reshape(num, joints, -1),
            "skeletons_3d": flat[:, n2d:n2d + n3d].reshape(num, joints, -1),
            "orientations": flat[:, n2d + n3d:].reshape(num, 2, 2),
        }


class ActionRecognizer:
    """ST-GCN on the EHPI skeleton sequences (multi label, sigmoid)."""

    def __init__(self, weights: str, num_actions: int, device: torch.device, runtime: RuntimeConfig,
                 net: Optional[torch.nn.Module] = None):
        self.device = device
        if net is None:
            net = EhpiStGcn(num_actions)
            net.load_state_dict(load_state_dict_file(weights))
            logger.info(f"Loaded ST-GCN action recognition (weights: {weights})")
        self.run = _Runner(net, device, runtime)
        self._mean = torch.tensor(EHPI_MEAN, device=device).view(1, 3, 1, 1) * 255
        self._std = torch.tensor(EHPI_STD, device=device).view(1, 3, 1, 1) * 255

    def __call__(self, ehpis: Sequence[np.ndarray]) -> np.ndarray:
        """ehpis: list of rows x T x 3 uint8 EHPI images -> N x num_actions probabilities"""
        batch = torch.from_numpy(np.stack(ehpis)).to(self.device).permute(0, 3, 1, 2).float()
        batch = (batch - self._mean) / self._std  # == ToTensor + Normalize(EHPI_MEAN, EHPI_STD) as in training
        return torch.sigmoid(self.run(batch).float()).cpu().numpy()


class TemporalLifter:
    """
    Temporal 3D lifter (``pose_lifter.TemporalPoseLifter``) on the per frame PedRecNet outputs of each track. Frames
    in which a person was not seen repeat its last known state, as in the training.
    """

    def __init__(self, weights: str, num_joints: int, device: torch.device, runtime: RuntimeConfig,
                 history_stride: int = 1, net: Optional[torch.nn.Module] = None):
        if net is None:
            net = TemporalPoseLifter(num_joints)
            net.load_state_dict(load_state_dict_file(weights))
            logger.info(f"Loaded temporal 3D lifter (weights: {weights})")
        self.window = net.window
        self.stride = history_stride
        self.device = device
        self.run = _Runner(net, device, runtime, images=False)
        self.histories: Dict[int, deque] = {}

    def reset(self):
        self.histories.clear()

    def remove(self, uid: int):
        self.histories.pop(uid, None)

    def _window(self, frame_nr: int, history: deque) -> np.ndarray:
        frames = np.array([entry[0] for entry in history])
        targets = frame_nr - self.stride * np.arange(self.window - 1, -1, -1)
        # latest entry at or before each target frame; before the first entry the first one is repeated
        indices = np.clip(np.searchsorted(frames, targets, side="right") - 1, 0, None)
        return np.stack([history[i][1] for i in indices])

    @torch.inference_mode()
    def __call__(self, frame_nr: int, humans: List[Human]):
        """Replaces the 3D poses (x, y, z) of the tracked humans with the lifted poses."""
        if len(humans) == 0:
            return
        windows = []
        for human in humans:
            history = self.histories.setdefault(human.uid, deque(maxlen=self.window * self.stride + 1))
            history.append((frame_nr, human.lifter_features))
            windows.append(self._window(frame_nr, history))
        lifted = self.run(torch.from_numpy(np.stack(windows)).to(self.device)).float().cpu().numpy()
        for human, pose in zip(humans, lifted):
            skeleton_3d = human.skeleton_3d.copy()
            skeleton_3d[:, :3] = pose * SKELETON_3D_RANGE
            human.skeleton_3d = skeleton_3d


class PedRecPipeline:
    def __init__(self, cfg: PipelineConfig, app_cfg: AppConfig, device: torch.device,
                 detector: Optional[RTDetrDetector] = None, pose_net: Optional[torch.nn.Module] = None,
                 action_net: Optional[torch.nn.Module] = None, lifter_net: Optional[torch.nn.Module] = None):
        """
        :param detector, pose_net, action_net, lifter_net: already constructed models (tests / benchmarks); by
            default they are loaded from ``cfg``
        """
        cfg.validate()
        self.cfg = cfg
        self.app_cfg = app_cfg
        self.device = device
        self.img_size: ImageSize = app_cfg.inference.img_size
        runtime = cfg.runtime
        if device.type == "cuda":
            # no cudnn.benchmark: the number of person crops (= batch size) changes from frame to frame and every new
            # batch size would trigger a new benchmark of all convolutions (~1 s stalls in videos)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.detector = None
        if cfg.use_detector:
            self.detector = detector or RTDetrDetector(device, cfg.rtdetr_model, half=runtime.half,
                                                       channels_last=runtime.channels_last)
        self.pose_estimator = None
        if cfg.use_pose:
            self.pose_estimator = PedRecPoseEstimator(
                cfg.pedrec_weights or default_paths.pedrec_net_weights(cfg.data_root), device, runtime, pose_net)
        self.action_recognizer = None
        if cfg.use_action:
            self.action_recognizer = ActionRecognizer(
                cfg.ehpi3d_weights or default_paths.ehpi3d_weights(cfg.data_root),
                len(app_cfg.inference.action_list), device, runtime, action_net)

        self.tracker = None
        if cfg.use_tracking:
            self.tracker = ByteTracker(high_thresh=cfg.person_high_thresh, low_thresh=cfg.person_low_thresh,
                                       new_track_thresh=cfg.person_high_thresh, max_time_lost=cfg.track_max_lost)
        self.smoothers: Dict[int, HumanStateSmoother] = {}

        # EHPI / lifter temporal sampling: use every n-th frame if the source has a higher frame rate than the models
        self.history_stride = max(1, int(round(cfg.source_fps / cfg.model_fps)))
        self.lifter = None
        if cfg.use_lifter:
            self.lifter = TemporalLifter(cfg.lifter_weights or default_paths.lifter_weights(cfg.data_root),
                                         PedRecNetConfig().model.num_joints, device, runtime, self.history_stride,
                                         lifter_net)
        buffer_size = max(app_cfg.inference.buffer_size, self.history_stride * cfg.temporal_field.width)
        self.image_content_buffer = ImageContentBuffer(buffer_size=buffer_size)

    def reset(self):
        """Forgets the temporal state (tracks, smoothing, 3D / action history), e.g. after a jump in a video."""
        if self.tracker is not None:
            next_uid = self.tracker.next_uid  # keep the ids unique over the whole video
            self.tracker.reset()
            self.tracker.next_uid = next_uid
        if self.lifter is not None:
            self.lifter.reset()
        self.smoothers.clear()
        self.image_content_buffer = ImageContentBuffer(buffer_size=self.image_content_buffer.buffer_size)

    # ------------------------------------------------------------------------------------------------------ stages
    def detect(self, frame: Optional[torch.Tensor]):
        """Returns (human bbs, other object bbs) in image coordinates."""
        if self.detector is None:
            return [get_full_frame_bb(self.img_size)], []
        # with tracking, low score persons are kept for ByteTrack's second association stage
        conf_thresh = self.cfg.person_low_thresh if self.tracker is not None else self.cfg.detector_conf_thresh
        conf_thresh = min(conf_thresh, self.cfg.detector_conf_thresh)
        bbs = self.detector(frame, self.img_size, conf_thresh)
        human_bbs, other_bbs = split_human_bbs([np.asarray(bb, dtype=np.float32) for bb in bbs])
        other_bbs = [bb for bb in other_bbs if get_bb_score(bb) >= self.cfg.detector_conf_thresh]
        if self.tracker is None:
            human_bbs = [bb for bb in human_bbs if get_bb_score(bb) > self.cfg.person_high_thresh]
        return human_bbs, other_bbs

    def estimate_poses(self, frame: torch.Tensor, human_bbs: List[np.ndarray]) -> List[Human]:
        preds = self.pose_estimator(frame, human_bbs)
        humans = get_humans_from_pedrec_detections(human_bbs, preds)
        if self.lifter is not None:
            for human, bb in zip(humans, human_bbs):
                human.lifter_features = lifter_features(human.skeleton_2d, human.skeleton_3d, bb)
        return humans

    def track(self, humans: List[Human], human_bbs: List[np.ndarray]) -> List[Human]:
        """Assigns track ids; humans that are not (yet) part of a confirmed track are dropped."""
        if len(humans) == 0:
            result = self.tracker.update(np.zeros((0, 4)), np.zeros(0))
        else:
            bbs = np.array([np.asarray(bb, dtype=np.float64)[:4] for bb in human_bbs])
            scores = np.array([get_bb_score(bb) for bb in human_bbs], dtype=np.float64)
            result = self.tracker.update(bbs, scores, humans)
        for uid in result.removed_uids:
            self.smoothers.pop(uid, None)
            if self.lifter is not None:
                self.lifter.remove(uid)
        tracked = []
        for det_idx, uid in result.assignments.items():
            humans[det_idx].uid = uid
            tracked.append(humans[det_idx])
        return tracked

    def smooth(self, humans: List[Human]):
        dt = 1.0 / self.cfg.source_fps
        for human in humans:
            smoother = self.smoothers.setdefault(human.uid, HumanStateSmoother())
            if self.lifter is None:  # the lifted 3D poses are already temporally consistent
                human.skeleton_3d = smoother.smooth_skeleton_3d(human.skeleton_3d, dt)
            human.orientation = smoother.smooth_orientation(human.orientation, dt)

    def _get_ehpi_history(self, uid: int) -> List[Human]:
        history = list(self.image_content_buffer.get_human_data_buffer_by_id(uid))
        # frames in which the human was not visible are stored as None: repeat the last known state
        filled, last = [], None
        for entry in history:
            last = entry if entry is not None else last
            if last is not None:
                filled.append(last)
        if self.history_stride > 1:
            filled = filled[-self.history_stride::-self.history_stride][::-1]
        return filled

    def recognize_actions(self, humans: List[Human]):
        ehpis = []
        for human in humans:
            human.ehpi = get_ehpi_from_human_history(human, self._get_ehpi_history(human.uid),
                                                     temporal_field=self.cfg.temporal_field)
            ehpis.append(human.ehpi)
        if len(ehpis) == 0:
            return
        action_probabilities = self.action_recognizer(ehpis)
        for human, action_probs in zip(humans, action_probabilities):
            human.action_probabilities = action_probs
            human.actions = get_action_classes(action_probs, self.app_cfg.inference.action_list,
                                               thresh=self.cfg.action_thresh)

    # ------------------------------------------------------------------------------------------------------ pipeline
    @torch.inference_mode()
    def process(self, frame_nr: int, img: np.ndarray) -> FrameResult:
        """Runs the enabled stages on one RGB frame of size ``app_cfg.inference.img_size``."""
        start = time.perf_counter()
        timings: Dict[str, float] = {}

        def lap(name: str, t0: float) -> float:
            now = time.perf_counter()
            timings[name] = now - t0
            return now

        t = start
        frame = None
        if self.detector is not None or self.pose_estimator is not None:
            frame = gpu_ops.frame_to_tensor(img, self.device)
            t = lap("upload", t)

        human_bbs, other_bbs = self.detect(frame)
        t = lap("detection", t)

        if self.pose_estimator is not None:
            humans = self.estimate_poses(frame, human_bbs)
            t = lap("pose", t)
            if self.tracker is not None:
                humans = self.track(humans, human_bbs)
                t = lap("tracking", t)
            humans = [human for human in humans if human.score > self.cfg.human_min_score]
            if self.lifter is not None:
                self.lifter(frame_nr, humans)
                t = lap("lifting", t)
            if self.tracker is not None:
                self.smooth(humans)
        else:
            empty_skeleton = np.zeros((PedRecNetConfig().model.num_joints, 3), dtype=np.float32)
            humans = [Human(bb=bb, skeleton_2d=empty_skeleton.copy(), skeleton_3d=None, orientation=None, uid=-1)
                      for bb in human_bbs]

        if self.action_recognizer is not None:
            self.recognize_actions(humans)
            t = lap("action", t)

        self.image_content_buffer.add(ImageContent(humans=humans, objects=other_bbs))

        required_time = time.perf_counter() - start
        timings["total"] = required_time
        fps = int(1.0 / required_time) if required_time > 0 else 0
        return FrameResult(frame_nr=frame_nr, humans=humans, objects=other_bbs, fps=fps, timings=timings)
