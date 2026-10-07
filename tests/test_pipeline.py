"""End to end pipeline runs with random weights (CPU)."""
import cv2
import numpy as np
import pytest
import torch

from pedrec.configs.app_config import AppConfig
from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.inference.pipeline import PedRecPipeline, PipelineConfig
from pedrec.models.data_structures import ImageSize
from pedrec.networks.net_pedrec.ehpi_stgcn import EhpiStGcn
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet

IMG_SIZE = ImageSize(320, 240)


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    root = tmp_path_factory.mktemp("weights")
    torch.manual_seed(0)
    paths = {"pedrec": root / "pedrec.pth", "action": root / "action.pth"}
    torch.save(PedRecNet(PedRecNetConfig()).state_dict(), paths["pedrec"])
    torch.save(EhpiStGcn(len(AppConfig().inference.action_list)).state_dict(), paths["action"])
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
    cfg = PipelineConfig(human_min_score=0.0, pedrec_weights=weights["pedrec"], ehpi3d_weights=weights["action"])
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
    assert set(result.timings) >= {"upload", "detection", "pose", "tracking", "action", "total"}


def test_pose_only_on_full_frame(weights):
    cfg = PipelineConfig(use_detector=False, use_tracking=False, use_action=False, human_min_score=0.0,
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
    cfg = PipelineConfig(use_pose=False, use_tracking=False, use_action=False)
    pipeline = PedRecPipeline(cfg, _app_cfg(), torch.device("cpu"), detector=detector)
    result = pipeline.process(1, next(_frames()))
    assert set(result.timings) >= {"upload", "detection", "total"}


def test_invalid_stage_combination():
    with pytest.raises(ValueError):
        PipelineConfig(use_pose=False, use_tracking=True, use_action=False).validate()
