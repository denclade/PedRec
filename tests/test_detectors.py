import pytest
import torch

from pedrec.models.data_structures import ImageSize


def test_rtdetr_detector_interface():
    transformers = pytest.importorskip("transformers")
    from pedrec.networks.net_detr.rtdetr_detector import RTDetrDetector
    torch.manual_seed(0)
    model = transformers.RTDetrV2ForObjectDetection(transformers.RTDetrV2Config(num_labels=80))
    detector = RTDetrDetector(torch.device("cpu"), "unused", model=model)
    frame = torch.rand(1, 3, 360, 640) * 255
    with torch.inference_mode():
        bbs = detector(frame, ImageSize(640, 360), conf_thresh=0.0)
    assert len(bbs) > 0 and all(len(bb) == 6 for bb in bbs)
    assert all(0 <= bb[5] < 80 for bb in bbs)
    assert all(bb[0] <= 640 and bb[1] <= 360 for bb in bbs)
    with torch.inference_mode():
        assert detector(frame, ImageSize(640, 360), conf_thresh=1.0) == []
