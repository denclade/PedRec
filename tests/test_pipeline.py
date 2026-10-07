"""End to end pipeline runs with random weights (CPU)."""
import cv2
import numpy as np
import pytest
import torch

from pedrec.configs.app_config import AppConfig
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.configs.yolo_v4_config import YoloV4Config
from pedrec.inference.pipeline import PedRecPipeline, PipelineConfig
from pedrec.models.data_structures import ImageSize
from pedrec.networks.net_pedrec.ehpi_3d_net import Ehpi3DNet
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.networks.net_yolo_v4.yolov4 import YoloV4


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    root = tmp_path_factory.mktemp("weights")
    torch.manual_seed(0)
    paths = {"yolo": root / "yolo.pth", "pedrec": root / "pedrec.pth", "ehpi": root / "ehpi.pth"}
    torch.save(YoloV4(YoloV4Config(), inference=True).state_dict(), paths["yolo"])
    torch.save(PedRecNet(PedRecNet50Config()).state_dict(), paths["pedrec"])
    torch.save(Ehpi3DNet(len(AppConfig().inference.action_list)).state_dict(), paths["ehpi"])
    return {k: str(v) for k, v in paths.items()}


def _frames(num=4, size=ImageSize(320, 240)):
    for i in range(num):
        img = np.full((size.height, size.width, 3), 40, dtype=np.uint8)
        cv2.rectangle(img, (60 + 5 * i, 30), (120 + 5 * i, 220), (200, 180, 160), -1)
        yield img


@pytest.mark.parametrize("tracker,smoothing", [("bytetrack", "one_euro"), ("legacy", "mean"), ("bytetrack", "none")])
def test_full_pipeline_single_pose_pass(weights, tracker, smoothing):
    app_cfg = AppConfig()
    app_cfg.inference.img_size = ImageSize(320, 240)
    cfg = PipelineConfig(use_detector=False, tracker=tracker, smoothing=smoothing, human_min_score=0.0,
                         yolo_weights=weights["yolo"], pedrec_weights=weights["pedrec"], ehpi3d_weights=weights["ehpi"])
    pipeline = PedRecPipeline(cfg, app_cfg, torch.device("cpu"))
    calls = []
    estimator = pipeline.pose_estimator

    class Counting:
        def __call__(self, frame, bbs):
            calls.append(len(bbs))
            return estimator(frame, bbs)
    pipeline.pose_estimator = Counting()
    for nr, img in enumerate(_frames(), start=1):
        result = pipeline.process(nr, img)
        assert all(h.action_probabilities is not None for h in result.humans)
    assert len(calls) == 4  # exactly one PedRecNet batch per frame
    if tracker == "bytetrack":
        assert len({h.uid for h in result.humans}) == len(result.humans) == 1


def test_detector_only(weights):
    app_cfg = AppConfig()
    app_cfg.inference.img_size = ImageSize(320, 240)
    cfg = PipelineConfig(use_pose=False, use_tracking=False, use_action=False, yolo_weights=weights["yolo"])
    pipeline = PedRecPipeline(cfg, app_cfg, torch.device("cpu"))
    result = pipeline.process(1, next(_frames()))
    assert set(result.timings) >= {"upload", "detection", "total"}
