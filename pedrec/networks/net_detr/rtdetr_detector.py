"""
Person / object detection with RT-DETRv2 (Lv et al., "RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time
Detection Transformer", 2024) via Hugging Face transformers (Apache 2.0 weights, e.g. ``PekingU/rtdetr_v2_r18vd``,
``_r34vd``, ``_r50vd``, ``_r101vd``). DETR style detectors predict a set of boxes, so no NMS is needed.
"""
import logging
from typing import List, Optional

import torch

from pedrec.inference import gpu_ops
from pedrec.models.data_structures import ImageSize

logger = logging.getLogger(__name__)


class RTDetrDetector:
    def __init__(self, device: torch.device, model_name: str, half: bool = False,
                 model: Optional[torch.nn.Module] = None, input_size: ImageSize = ImageSize(640, 640)):
        """
        :param model_name: Hugging Face hub id or local directory (``from_pretrained``)
        :param model: an already constructed transformers RT-DETR model (tests / offline use)
        """
        if model is None:
            from transformers import AutoModelForObjectDetection
            model = AutoModelForObjectDetection.from_pretrained(model_name)
            logger.info(f"Loaded RT-DETR {model_name}")
        self.model = model.to(device).eval()
        self.device = device
        self.input_size = input_size
        self.autocast = half and device.type == "cuda"

    def __call__(self, frame: torch.Tensor, img_size: ImageSize, conf_thresh: float) -> List[List[float]]:
        """
        :param frame: 1 x 3 x H x W RGB frame with values in [0, 255] (see gpu_ops.frame_to_tensor)
        :return: bbs (center x, center y, width, height, confidence, COCO class idx, 0 = person) in image pixels
        """
        pixel_values = gpu_ops.resize_bilinear(frame, self.input_size) / 255.0  # RT-DETR: no mean / std
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.autocast):
            outputs = self.model(pixel_values=pixel_values)
        scores, labels = torch.sigmoid(outputs.logits[0].float()).max(dim=1)
        keep = scores > conf_thresh
        boxes = outputs.pred_boxes[0].float()[keep]  # center x, center y, w, h normalized to [0, 1]
        scale = torch.tensor([img_size.width, img_size.height, img_size.width, img_size.height], device=boxes.device)
        result = torch.cat((boxes * scale, scores[keep, None], labels[keep, None].float()), dim=1)
        return result.cpu().tolist()
