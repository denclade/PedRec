"""
Framework independent PedRec inference pipeline.

The pipeline chains the individual components (all of them optional):

    YoloV4 detector -> PedRecNet (2D / 3D pose, orientation, joint confidence) -> tracking -> EHPI3D action recognition

It is used by the Qt demo (``pedrec/demo.py``), the headless runner, the benchmark and the dataset tools, so every
component can be used on its own (e.g. only the detector, or the pose net on full frames without a detector).

Performance relevant design decisions:

* everything runs under ``torch.inference_mode``
* the frame is uploaded once; detector resize, person crops (``grid_sample``), normalization, NMS and the
  back-transformation of the 2D poses run batched on the device (``pedrec.inference.gpu_ops``)
* PedRecNet runs exactly once per frame on all person crops. The legacy tracker used to run it a second time for
  tracked but undetected persons; their crops are part of the first batch, so the results are reused.
* optional fp16 autocast, channels_last memory format and ``torch.compile`` (see ``RuntimeConfig``)
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from pedrec.configs import default_paths
from pedrec.configs.app_config import AppConfig
from pedrec.configs.pedrec_net_config import PedRecNet50Config, PedRecNetConfig
from pedrec.configs.yolo_v4_config import YoloV4Config
from pedrec.inference import gpu_ops
from pedrec.models.constants.action_mappings import ACTION
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.networks.net_pedrec.ehpi_3d_net import Ehpi3DNet
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.tracking.byte_tracker import ByteTracker
from pedrec.tracking.human_merger import HumanMerger
from pedrec.tracking.human_tracker import HumanTracker, bb_tracking, add_undetected_bbs_from_tracking, \
    remove_duplicates
from pedrec.tracking.one_euro import HumanStateSmoother
from pedrec.utils.bb_helper import split_human_bbs, get_bb_score
from pedrec.utils.demo_helper import get_detector, init_pose_model
from pedrec.utils.ehpi_helper import get_ehpi_from_human_history
from pedrec.utils.human_helper import get_humans_from_pedrec_detections
from pedrec.utils.image_content_buffer import ImageContent, ImageContentBuffer
from pedrec.utils.skeleton_helper import get_skeleton_mean_score
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
    compile: bool = False  # torch.compile the networks (slow first frames, faster afterwards)
    backend: str = "torch"  # "torch" or "onnx" (onnxruntime, see pedrec/tools/networks/export_onnx.py)
    onnx_dir: Optional[str] = None  # directory with the exported *.onnx files (default <data-dir>/models/onnx)
    onnx_providers: Optional[Sequence[str]] = None  # onnxruntime execution providers, default TensorRT > CUDA > CPU


@dataclass
class PipelineConfig:
    """
    Which components run and where their weights are. ``None`` weight paths resolve to the defaults below the data
    root (``PEDREC_DATA_DIR``).
    """
    use_detector: bool = True  # YoloV4 human / object detection; if False the full frame is used as human bb
    use_pose: bool = True  # PedRecNet 2D / 3D pose + orientation + joint confidence
    use_tracking: bool = True  # id assignment over time (requires pose)
    use_action: bool = True  # EHPI3D action recognition (requires pose + tracking)
    tracker: str = "bytetrack"  # "bytetrack" (Kalman + two stage IoU association) or "legacy" (optical flow + merge)
    smoothing: str = "one_euro"  # "one_euro", "mean" (legacy 2 frame mean) or "none"

    yolo_weights: Optional[str] = None
    pedrec_weights: Optional[str] = None
    ehpi3d_weights: Optional[str] = None
    data_root: Optional[str] = None

    detector_conf_thresh: float = 0.4
    detector_nms_thresh: float = 0.6
    human_bb_min_score: float = 0.6
    human_min_score: float = 0.65  # humans with a lower mean joint score are dropped
    track_low_thresh: float = 0.1  # ByteTrack: lowest detection score used to continue existing tracks
    track_max_lost: int = 30  # ByteTrack: frames a lost track can be recovered
    action_thresh: float = 0.7
    num_smoothing_frames: int = 2  # smoothing == "mean"
    temporal_field: ImageSize = field(default_factory=lambda: ImageSize(width=64, height=32))
    source_fps: float = 30.0  # frame rate of the input, used for smoothing and the EHPI temporal sampling
    model_fps: float = 30.0  # frame rate the EHPI3D model was trained with (SIM-C01: 30 fps)

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self):
        if self.use_action and not (self.use_pose and self.use_tracking):
            raise ValueError("Action recognition requires pose estimation and tracking.")
        if self.use_tracking and not self.use_pose:
            raise ValueError("Tracking requires pose estimation.")
        if self.tracker not in ("bytetrack", "legacy"):
            raise ValueError(f"Unknown tracker '{self.tracker}'")
        if self.smoothing not in ("one_euro", "mean", "none"):
            raise ValueError(f"Unknown smoothing '{self.smoothing}'")


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


def _prepare_module(module: torch.nn.Module, device: torch.device, runtime: RuntimeConfig) -> torch.nn.Module:
    module = module.to(device).eval()
    if runtime.channels_last:
        module = module.to(memory_format=torch.channels_last)
    if runtime.compile:
        module = torch.compile(module, dynamic=True)
    return module


class _Runner:
    """Executes a network with the configured runtime options (autocast, memory format) or an ONNX session."""

    def __init__(self, module, device: torch.device, runtime: RuntimeConfig, onnx_session=None):
        self.module = module
        self.device = device
        self.runtime = runtime
        self.onnx_session = onnx_session
        self.autocast = runtime.half and device.type == "cuda"

    def __call__(self, inputs: torch.Tensor):
        if self.onnx_session is not None:
            return self.onnx_session(inputs)
        if self.runtime.channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.autocast):
            return self.module(inputs)


class YoloV4HumanDetector:
    def __init__(self, weights: str, device: torch.device, runtime: RuntimeConfig, onnx_session=None):
        self.cfg = YoloV4Config()
        self.device = device
        module = None
        if onnx_session is None:
            module = _prepare_module(get_detector(self.cfg, weights, logger, torch.device("cpu")), device, runtime)
        self.run = _Runner(module, device, runtime, onnx_session)

    def __call__(self, frame: torch.Tensor, img_size: ImageSize, conf_thresh: float, nms_thresh: float,
                 tracked_bbs: Optional[np.ndarray] = None) -> List[List[float]]:
        model_input = gpu_ops.resize_bilinear(frame, self.cfg.model.input_size) / 255.0
        output = self.run(model_input)
        return gpu_ops.yolo_postprocess(output, img_size, conf_thresh, nms_thresh, tracked_bbs)


class PedRecPoseEstimator:
    """PedRecNet on a batch of person bbs of one frame."""

    def __init__(self, weights: str, device: torch.device, runtime: RuntimeConfig, cfg: PedRecNetConfig = None,
                 onnx_session=None):
        self.cfg = cfg or PedRecNet50Config()
        self.device = device
        module = None
        if onnx_session is None:
            module = init_pose_model(PedRecNet(self.cfg), weights, logger, torch.device("cpu"))
            module = _prepare_module(module, device, runtime)
        self.run = _Runner(module, device, runtime, onnx_session)
        input_size = self.cfg.model.input_size
        self._input_scale = torch.tensor([input_size.width, input_size.height], dtype=torch.float32, device=device)
        self._orientation_scale = torch.tensor([np.pi, 2 * np.pi], dtype=torch.float32, device=device)

    def __call__(self, frame: torch.Tensor, bbs: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        :return: dict with "skeletons" (B x J x 3: x, y in image pixels, confidence), "skeletons_3d" (B x J x 4:
            x, y, z in mm relative to the hip, confidence) and "orientations" (B x 2 x 2: body / head theta, phi in
            radians), identical to ``pose_deconv_helper.pedrec_recognizer``
        """
        num = len(bbs)
        if num == 0:
            return {"skeletons": [], "skeletons_3d": [], "orientations": []}
        input_size = self.cfg.model.input_size
        _, trans_invs = gpu_ops.get_crop_transforms(bbs, input_size)
        trans_invs = torch.from_numpy(trans_invs).to(self.device)
        crops = gpu_ops.crop_affine(frame, trans_invs, input_size)
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


class Ehpi3DActionRecognizer:
    def __init__(self, weights: str, num_actions: int, device: torch.device, runtime: RuntimeConfig,
                 onnx_session=None):
        self.device = device
        module = None
        if onnx_session is None:
            module = Ehpi3DNet(num_actions)
            module.load_state_dict(load_state_dict_file(weights))
            module = _prepare_module(module, device, runtime)
            logger.info(f"Loaded EHPI3D (weights: {weights})")
        self.run = _Runner(module, device, runtime, onnx_session)
        self._mean = torch.tensor(EHPI_MEAN, device=device).view(1, 3, 1, 1) * 255
        self._std = torch.tensor(EHPI_STD, device=device).view(1, 3, 1, 1) * 255

    def __call__(self, ehpis: Sequence[np.ndarray]) -> np.ndarray:
        """ehpis: list of HxWx3 uint8 EHPI images -> N x num_actions probabilities"""
        batch = torch.from_numpy(np.stack(ehpis)).to(self.device).permute(0, 3, 1, 2).float()
        batch = (batch - self._mean) / self._std  # == ToTensor + Normalize(EHPI_MEAN, EHPI_STD)
        return torch.sigmoid(self.run(batch).float()).cpu().numpy()


def _load_onnx_session(runtime: RuntimeConfig, data_root: Optional[str], name: str, device: torch.device):
    if runtime.backend != "onnx":
        return None
    from pedrec.inference.onnx_runtime import OnnxModule, default_onnx_dir
    import os
    path = os.path.join(runtime.onnx_dir or default_onnx_dir(data_root), f"{name}.onnx")
    return OnnxModule(path, device, runtime.onnx_providers)


class PedRecPipeline:
    def __init__(self, cfg: PipelineConfig, app_cfg: AppConfig, device: torch.device):
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
            self.detector = YoloV4HumanDetector(cfg.yolo_weights or default_paths.yolo_v4_weights(cfg.data_root),
                                                device, runtime,
                                                _load_onnx_session(runtime, cfg.data_root, "yolov4", device))
        self.pose_estimator = None
        if cfg.use_pose:
            self.pose_estimator = PedRecPoseEstimator(
                cfg.pedrec_weights or default_paths.pedrec_net_weights(cfg.data_root), device, runtime,
                onnx_session=_load_onnx_session(runtime, cfg.data_root, "pedrecnet", device))
        self.action_recognizer = None
        if cfg.use_action:
            self.action_recognizer = Ehpi3DActionRecognizer(
                cfg.ehpi3d_weights or default_paths.ehpi3d_weights(cfg.data_root),
                len(app_cfg.inference.action_list), device, runtime,
                _load_onnx_session(runtime, cfg.data_root, "ehpi3d", device))

        # tracking
        self.legacy_tracker = None
        self.legacy_merger = None
        self.byte_tracker = None
        if cfg.use_tracking and cfg.tracker == "legacy":
            self.legacy_tracker = HumanTracker(img_size=self.img_size)
            self.legacy_merger = HumanMerger(self.img_size)
        elif cfg.use_tracking:
            self.byte_tracker = ByteTracker(high_thresh=cfg.human_bb_min_score, low_thresh=cfg.track_low_thresh,
                                            new_track_thresh=cfg.human_bb_min_score,
                                            max_time_lost=cfg.track_max_lost)
        self.smoothers: Dict[int, HumanStateSmoother] = {}

        # EHPI temporal sampling: use every n-th frame if the source has a higher frame rate than the model
        self.history_stride = max(1, int(round(cfg.source_fps / cfg.model_fps)))
        buffer_size = max(app_cfg.inference.buffer_size, self.history_stride * cfg.temporal_field.width)
        self.image_content_buffer = ImageContentBuffer(buffer_size=buffer_size)

    def reset(self):
        """Forgets the temporal state (tracks, smoothing, action history), e.g. after a jump in a video."""
        if self.byte_tracker is not None:
            next_uid = self.byte_tracker.next_uid  # keep the ids unique over the whole video
            self.byte_tracker.reset()
            self.byte_tracker.next_uid = next_uid
        if self.legacy_tracker is not None:
            next_uid = self.legacy_merger.next_human_uid
            self.legacy_tracker = HumanTracker(img_size=self.img_size)
            self.legacy_merger = HumanMerger(self.img_size)
            self.legacy_merger.next_human_uid = next_uid
        self.smoothers.clear()
        self.image_content_buffer = ImageContentBuffer(buffer_size=self.image_content_buffer.buffer_size)

    # ------------------------------------------------------------------------------------------------------ stages
    def detect(self, frame: Optional[torch.Tensor], tracked_humans: List[Human]):
        """Returns (human bbs, other object bbs) in image coordinates."""
        if self.detector is None:
            human_bbs, other_bbs = [get_full_frame_bb(self.img_size)], []
        else:
            conf_thresh = self.cfg.detector_conf_thresh
            if self.byte_tracker is not None:
                conf_thresh = min(conf_thresh, self.cfg.track_low_thresh)
            tracked_bbs = None
            if self.legacy_tracker is not None and len(tracked_humans) > 0:
                tracked_bbs = np.array([np.asarray(h.bb, dtype=np.float32)[:6] for h in tracked_humans])
            bbs = self.detector(frame, self.img_size, conf_thresh, self.cfg.detector_nms_thresh, tracked_bbs)
            human_bbs, other_bbs = split_human_bbs([np.asarray(bb, dtype=np.float32) for bb in bbs])
            other_bbs = [bb for bb in other_bbs if get_bb_score(bb) >= self.cfg.detector_conf_thresh]

        if self.legacy_tracker is not None:
            human_bbs = bb_tracking(human_bbs, tracked_humans)
            human_bbs = add_undetected_bbs_from_tracking(human_bbs, tracked_humans)
            # tracked bbs (uid != -1) are kept regardless of their score
            human_bbs = [bb for bb in human_bbs if bb[-1] != -1 or get_bb_score(bb) > self.cfg.human_bb_min_score]
            human_bbs = remove_duplicates(human_bbs)
        elif self.byte_tracker is None and self.detector is not None:
            human_bbs = [bb for bb in human_bbs if get_bb_score(bb) > self.cfg.human_bb_min_score]
        return human_bbs, other_bbs

    def estimate_poses(self, frame: torch.Tensor, human_bbs: List[np.ndarray]) -> List[Human]:
        preds = self.pose_estimator(frame, human_bbs)
        return get_humans_from_pedrec_detections(human_bbs, preds)

    def _track_legacy(self, humans: List[Human], human_bbs: List[np.ndarray],
                      tracked_humans: List[Human]) -> List[Human]:
        # PedRecNet results of the tracked bbs from this frame's (single) batch, keyed by uid
        cache = {int(bb[-1]): human for bb, human in zip(human_bbs, humans) if len(bb) > 6 and bb[-1] != -1}
        humans, undetected_humans = self.legacy_merger.merge_humans(humans, tracked_humans, assign_new_ids=True)
        for human in undetected_humans:
            pred = cache.get(human.uid)
            if pred is None or get_skeleton_mean_score(pred.skeleton_2d) < 0.4:
                continue
            humans.append(Human(bb=human.bb, skeleton_2d=pred.skeleton_2d.copy(), skeleton_3d=pred.skeleton_3d.copy(),
                                orientation=pred.orientation.copy(), uid=human.uid))
        return humans

    def _track_bytetrack(self, humans: List[Human], human_bbs: List[np.ndarray]) -> List[Human]:
        if len(humans) == 0:
            result = self.byte_tracker.update(np.zeros((0, 4)), np.zeros(0))
        else:
            bbs = np.array([np.asarray(bb, dtype=np.float64)[:4] for bb in human_bbs])
            scores = np.array([get_bb_score(bb) for bb in human_bbs], dtype=np.float64)
            result = self.byte_tracker.update(bbs, scores, humans)
        for uid in result.removed_uids:
            self.smoothers.pop(uid, None)
        tracked = []
        for det_idx, uid in result.assignments.items():
            humans[det_idx].uid = uid
            tracked.append(humans[det_idx])
        return tracked

    def smooth(self, humans: List[Human]):
        if self.cfg.smoothing == "none":
            return
        if self.cfg.smoothing == "mean":
            num = self.cfg.num_smoothing_frames
            for human in humans:
                history = [h for h in self.image_content_buffer.get_human_data_buffer_by_id(human.uid) if h is not None]
                if num > 1 and len(history) >= num:
                    for i in range(1, num):
                        human.skeleton_3d += history[-i].skeleton_3d
                        human.orientation += history[-i].orientation
                    human.orientation /= num
                    human.skeleton_3d /= num
            return
        dt = 1.0 / self.cfg.source_fps
        for human in humans:
            if human.uid == -1:
                continue
            smoother = self.smoothers.setdefault(human.uid, HumanStateSmoother())
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
        """
        Runs the enabled components on one RGB frame of size ``app_cfg.inference.img_size``.
        """
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

        tracked_humans: List[Human] = []
        if self.legacy_tracker is not None:
            last_humans = self.image_content_buffer.get_last_humans()
            tracked_humans = self.legacy_tracker.get_humans_by_tracking(img, previous_humans=last_humans)
            t = lap("optical_flow", t)

        human_bbs, other_bbs = self.detect(frame, tracked_humans)
        t = lap("detection", t)

        humans: List[Human] = []
        if self.pose_estimator is not None:
            humans = self.estimate_poses(frame, human_bbs)
            t = lap("pose", t)
            if self.legacy_tracker is not None:
                humans = self._track_legacy(humans, human_bbs, tracked_humans)
            elif self.byte_tracker is not None:
                humans = self._track_bytetrack(humans, human_bbs)
            t = lap("tracking", t)
            humans = [human for human in humans if human.score > self.cfg.human_min_score]
            self.smooth(humans)
        else:
            empty_skeleton = np.zeros((PedRecNet50Config().model.num_joints, 3), dtype=np.float32)
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
