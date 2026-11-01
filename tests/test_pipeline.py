"""End to end pipeline runs with random weights (CPU)."""
import cv2
import numpy as np
import pytest
import torch

from pedrec.configs.app_config import AppConfig
from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.inference.pipeline import PedRecPipeline, PipelineConfig
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.networks.net_pedrec.ehpi_stgcn import EhpiStGcn
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.networks.net_pedrec.pose_lifter import TemporalPoseLifter

IMG_SIZE = ImageSize(320, 240)


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    root = tmp_path_factory.mktemp("weights")
    torch.manual_seed(0)
    paths = {"pedrec": root / "pedrec.pth", "action": root / "action.pth", "lifter": root / "lifter.pth"}
    torch.save(PedRecNet(PedRecNetConfig()).state_dict(), paths["pedrec"])
    torch.save(EhpiStGcn(len(AppConfig().inference.action_list)).state_dict(), paths["action"])
    torch.save(TemporalPoseLifter(PedRecNetConfig().model.num_joints).state_dict(), paths["lifter"])
    return {k: str(v) for k, v in paths.items()}


def _app_cfg():
    app_cfg = AppConfig()
    app_cfg.inference.img_size = IMG_SIZE
    return app_cfg


def _frames(num=4):
    for i in range(num):
        img = np.full((IMG_SIZE.height, IMG_SIZE.width, 3), 40, dtype=np.uint8)
        cv2.rectangle(img, (60 + 5 * i, 30), (120 + 5 * i, 220), (200, 180, 160), -1)
        yield img


class FakeDetector:
    """Returns one person moving to the right and one object (center bbs in pixels)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame, img_size, conf_thresh):
        self.calls += 1
        x = 90 + 5 * (self.calls - 1)
        return [[x, 125, 60, 190, 0.9, 0], [250, 200, 40, 30, 0.8, 2]]


def test_full_pipeline_single_pose_pass(weights):
    cfg = PipelineConfig(human_min_score=0.0, pedrec_weights=weights["pedrec"], ehpi3d_weights=weights["action"],
                         lifter_weights=weights["lifter"])
    detector = FakeDetector()
    pipeline = PedRecPipeline(cfg, _app_cfg(), torch.device("cpu"), detector=detector)
    calls = []
    estimator = pipeline.pose_estimator

    class Counting:
        def __call__(self, frame, bbs):
            calls.append(len(bbs))
            return estimator(frame, bbs)
    pipeline.pose_estimator = Counting()
    uids = set()
    for nr, img in enumerate(_frames(), start=1):
        result = pipeline.process(nr, img)
        assert all(h.action_probabilities is not None for h in result.humans)
        assert len(result.objects) == 1
        uids |= {h.uid for h in result.humans}
    assert calls == [1, 1, 1, 1]  # exactly one PedRecNet batch per frame
    assert detector.calls == 4
    assert len(result.humans) == 1 and len(uids) == 1  # one track over all frames
    assert set(result.timings) >= {"upload", "detection", "pose", "tracking", "lifting", "action", "total"}


def test_pose_only_on_full_frame(weights):
    cfg = PipelineConfig(use_detector=False, use_tracking=False, use_lifter=False, use_action=False,
                         human_min_score=0.0,
                         pedrec_weights=weights["pedrec"])
    pipeline = PedRecPipeline(cfg, _app_cfg(), torch.device("cpu"))
    result = pipeline.process(1, next(_frames()))
    assert len(result.humans) == 1
    assert result.humans[0].skeleton_3d.shape == (PedRecNetConfig().model.num_joints, 4)


def test_detector_only_with_rtdetr():
    transformers = pytest.importorskip("transformers")
    from pedrec.networks.net_detr.rtdetr_detector import RTDetrDetector
    torch.manual_seed(0)
    model = transformers.RTDetrV2ForObjectDetection(transformers.RTDetrV2Config(num_labels=80))
    detector = RTDetrDetector(torch.device("cpu"), "unused", model=model)
    cfg = PipelineConfig(use_pose=False, use_tracking=False, use_lifter=False, use_action=False)
    pipeline = PedRecPipeline(cfg, _app_cfg(), torch.device("cpu"), detector=detector)
    result = pipeline.process(1, next(_frames()))
    assert set(result.timings) >= {"upload", "detection", "total"}


def test_invalid_stage_combination():
    with pytest.raises(ValueError):
        PipelineConfig(use_pose=False, use_tracking=True, use_action=False).validate()
    with pytest.raises(ValueError):
        PipelineConfig(use_tracking=False, use_lifter=True, use_action=False).validate()


def _tracking_pipeline(weights, visible, scores, recovery_frames=3):
    class Detector(FakeDetector):
        def __call__(self, *args):
            boxes = super().__call__(*args)
            return boxes[:1] if visible[self.calls - 1] else []

    class PoseEstimator:
        calls = 0

        def __call__(self, frame, bbs):
            self.calls += 1
            skeletons = np.zeros((len(bbs), 26, 3), dtype=np.float32)
            poses = np.zeros((len(bbs), 26, 4), dtype=np.float32)
            for i, bb in enumerate(bbs):
                skeletons[i, :, 0] = bb[0] + np.linspace(-10, 10, 26)
                skeletons[i, :, 1] = bb[1] + np.linspace(-30, 30, 26)
                skeletons[i, :, 2] = scores[detector.calls - 1]
                poses[i, :, 0] = self.calls  # fresh estimates, not frozen previous skeletons
                poses[i, :, 3] = skeletons[i, :, 2]
            return dict(skeletons=skeletons, skeletons_3d=poses,
                        orientations=np.zeros((len(bbs), 2, 2), dtype=np.float32))

    detector = Detector()
    cfg = PipelineConfig(use_lifter=False, use_action=False, pedrec_weights=weights["pedrec"],
                         track_recovery_frames=recovery_frames)
    pipeline = PedRecPipeline(cfg, _app_cfg(), torch.device("cpu"), detector=detector)
    pipeline.pose_estimator = PoseEstimator()
    return pipeline


def test_short_detector_gaps_get_fresh_poses_and_keep_id(weights):
    pipeline = _tracking_pipeline(weights, [True, False, False, False, False, True], [0.9] * 6)
    results = [pipeline.process(i, img) for i, img in enumerate(_frames(6))]
    assert [len(result.humans) for result in results] == [1, 1, 1, 1, 0, 1]
    humans = [result.humans[0] for result in results if result.humans]
    assert len({human.uid for human in humans}) == 1
    assert len({float(human.skeleton_3d[0, 0]) for human in humans}) == len(humans)
    assert len(pipeline.tracker.tracks) == 1
    assert pipeline.tracker.next_uid == 1  # recovery must not create additional tracks


def test_pose_confidence_hysteresis_only_for_accepted_tracks(weights):
    pipeline = _tracking_pipeline(weights, [True] * 5, [0.5, 0.9, 0.5, 0.2, 0.9])
    results = [pipeline.process(i, img) for i, img in enumerate(_frames(5))]
    assert [len(result.humans) for result in results] == [0, 1, 1, 0, 1]
    assert len({result.humans[0].uid for result in results if result.humans}) == 1


@pytest.mark.parametrize("score,recovery_frames", [(0.2, 3), (0.9, 0)])
def test_gap_recovery_requires_pose_confidence_and_can_be_disabled(weights, score, recovery_frames):
    pipeline = _tracking_pipeline(weights, [True, False], [0.9, score], recovery_frames)
    assert len(pipeline.process(0, next(_frames())).humans) == 1
    assert pipeline.process(1, next(_frames())).humans == []


def test_gap_recovery_does_not_duplicate_overlapping_detection(weights):
    pipeline = _tracking_pipeline(weights, [True, True], [0.9, 0.9])
    assert len(pipeline.process(0, next(_frames())).humans) == 1
    # Force an association failure despite an overlapping observation.
    pipeline.tracker.match_thresh = 0.0
    result = pipeline.process(1, next(_frames()))
    assert len(result.humans) == 1


@pytest.mark.parametrize("use_lifter", [False, True])
def test_3d_smoothing_reduces_stationary_jitter_with_or_without_lifter(use_lifter):
    # Exercise the live postprocessing on a stationary pose with alternating estimation noise.
    pipeline = PedRecPipeline.__new__(PedRecPipeline)
    pipeline.cfg = PipelineConfig(source_fps=30, use_lifter=use_lifter)
    pipeline.lifter = object() if use_lifter else None
    pipeline.smoothers = {}
    raw, filtered = [], []
    for frame in range(60):
        pose = np.zeros((26, 4), dtype=np.float32)
        pose[:, 0] = 10.0 if frame % 2 else -10.0
        pose[:, 3] = 0.9
        original = pose.copy()
        human = Human(bb=[], skeleton_2d=None, skeleton_3d=pose,
                      orientation=np.zeros((2, 2), dtype=np.float32), uid=7)
        pipeline.smooth([human])
        raw.append(original[0, 0])
        filtered.append(human.skeleton_3d[0, 0])
        np.testing.assert_array_equal(human.skeleton_3d[:, 3], original[:, 3])
        np.testing.assert_array_equal(pose, original)  # no feedback into raw lifter inputs
    assert filtered[0] == raw[0]
    assert np.mean(np.abs(np.diff(filtered[10:]))) < 0.5 * np.mean(np.abs(np.diff(raw[10:])))
