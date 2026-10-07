"""
Person / object detection with RT-DETRv2 (Lv et al., "RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time
Detection Transformer", 2024) via Hugging Face transformers (Apache 2.0 weights, e.g. ``PekingU/rtdetr_v2_r18vd``).
DETR style detectors predict a set of boxes, so no NMS is needed.

The frame is resized to a long side of 640 px keeping its aspect ratio (multiples of 32, e.g. 640 x 384 for 16:9):
persons are not squeezed and a 16:9 frame needs ~40% less computation than the 640 x 640 default.
"""
import logging
from typing import List, Optional

import torch

from pedrec.inference import gpu_ops
from pedrec.models.data_structures import ImageSize

logger = logging.getLogger(__name__)


def get_detector_input_size(img_size: ImageSize, long_side: int = 640, multiple: int = 32) -> ImageSize:
    scale = long_side / max(img_size.width, img_size.height)
    width = max(multiple, int(round(img_size.width * scale / multiple)) * multiple)
    height = max(multiple, int(round(img_size.height * scale / multiple)) * multiple)
    return ImageSize(width=width, height=height)


class RTDetrDetector:
    def __init__(self, device: torch.device, model_name: str, half: bool = False,
                 model: Optional[torch.nn.Module] = None, channels_last: bool = False, long_side: int = 640):
        """
        :param model_name: Hugging Face hub id or local directory (``from_pretrained``)
        :param model: an already constructed transformers RT-DETR model (tests / offline use)
        """
        if model is None:
            from transformers import AutoModelForObjectDetection
            model = AutoModelForObjectDetection.from_pretrained(model_name)
            logger.info(f"Loaded RT-DETR {model_name}")
        # positional embeddings / anchors computed for the actual feature map size (identical for 640 x 640)
        model.config.anchor_image_size = None
        for module in model.modules():
            if hasattr(module, "eval_size"):
                module.eval_size = None
        model = model.to(device).eval()
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        self.model = model
        self.device = device
        self.long_side = long_side
        self.channels_last = channels_last
        self.autocast = half and device.type == "cuda"

    def __call__(self, frame: torch.Tensor, img_size: ImageSize, conf_thresh: float) -> List[List[float]]:
        """
        :param frame: 1 x 3 x H x W RGB frame with values in [0, 255] (see gpu_ops.frame_to_tensor)
        :return: bbs (center x, center y, width, height, confidence, COCO class idx, 0 = person) in image pixels
        """
        input_size = get_detector_input_size(img_size, self.long_side)
        pixel_values = gpu_ops.resize_bilinear(frame, input_size) / 255.0  # RT-DETR: no mean / std
        if self.channels_last:
            pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.autocast):
            outputs = self.model(pixel_values=pixel_values)
        scores, labels = torch.sigmoid(outputs.logits[0].float()).max(dim=1)
        keep = scores > conf_thresh
        boxes = outputs.pred_boxes[0].float()[keep]  # center x, center y, w, h normalized to [0, 1]
        scale = torch.tensor([img_size.width, img_size.height, img_size.width, img_size.height], device=boxes.device)
        result = torch.cat((boxes * scale, scores[keep, None], labels[keep, None].float()), dim=1)
        return result.cpu().tolist()
