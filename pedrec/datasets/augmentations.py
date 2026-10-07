"""
Additional training augmentations (all disabled by default, enabled by ``--augmentation strong``):

* half body crops (Wang et al. / MMPose): crop to the visible upper or lower body joints, trains truncated persons
* random erasing (Zhong et al. 2020) on the normalized model input: simulates occlusions
* color jitter: brightness, contrast and saturation
"""
from random import random
from typing import Optional, Tuple

import numpy as np
import torch

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT as J
from pedrec.models.data_structures import ImageSize
from pedrec.utils.bb_helper import bb_to_center_scale

UPPER_BODY = [J.nose, J.left_eye, J.right_eye, J.left_ear, J.right_ear, J.left_shoulder, J.right_shoulder,
              J.left_elbow, J.right_elbow, J.left_wrist, J.right_wrist, J.neck, J.head_lower, J.head_upper,
              J.spine_center, J.left_hand_end, J.right_hand_end]
LOWER_BODY = [J.left_hip, J.right_hip, J.left_knee, J.right_knee, J.left_ankle, J.right_ankle, J.hip_center,
              J.left_foot_end, J.right_foot_end]
UPPER_IDX = np.array([j.value for j in UPPER_BODY])
LOWER_IDX = np.array([j.value for j in LOWER_BODY])


def half_body_center_scale(skeleton_2d: np.ndarray, input_size: ImageSize, min_visible: int = 8,
                           padding: float = 1.2) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    :param skeleton_2d: J x >=4 joints in image coordinates (x, y, score, visible, ...)
    :return: (center, scale) of the upper or lower body, or None if not enough joints are visible
    """
    visible = skeleton_2d[:, 3] > 0
    if visible.sum() < min_visible:
        return None
    upper = UPPER_IDX[visible[UPPER_IDX]]
    lower = LOWER_IDX[visible[LOWER_IDX]]
    selected = upper if (random() < 0.5 and len(upper) > 2) or len(lower) <= 2 else lower
    if len(selected) <= 2:
        return None
    points = skeleton_2d[selected, :2]
    top_left, bottom_right = points.min(axis=0), points.max(axis=0)
    width, height = bottom_right - top_left
    if width < 2 or height < 2:
        return None
    bb = np.array([*(top_left + bottom_right) / 2, width * padding, height * padding, 1, 0], dtype=np.float32)
    return bb_to_center_scale(bb, input_size)


def color_jitter(img: np.ndarray, strength: float) -> np.ndarray:
    """Random brightness / contrast / saturation change of an RGB uint8 image by up to +-strength."""
    if strength <= 0:
        return img
    out = img.astype(np.float32)
    brightness = 1 + (random() * 2 - 1) * strength
    contrast = 1 + (random() * 2 - 1) * strength
    saturation = 1 + (random() * 2 - 1) * strength
    out *= brightness
    mean = out.mean()
    out = (out - mean) * contrast + mean
    gray = out.mean(axis=2, keepdims=True)
    out = (out - gray) * saturation + gray
    return np.clip(out, 0, 255).astype(np.uint8)


def random_erasing(model_input: torch.Tensor, probability: float, area: Tuple[float, float] = (0.02, 0.2),
                   aspect: Tuple[float, float] = (0.3, 3.3)) -> torch.Tensor:
    """Erases a random rectangle of the (normalized) C x H x W model input with random values."""
    if probability <= 0 or random() > probability:
        return model_input
    _, height, width = model_input.shape
    for _ in range(10):
        target_area = (area[0] + random() * (area[1] - area[0])) * height * width
        ratio = np.exp(np.log(aspect[0]) + random() * (np.log(aspect[1]) - np.log(aspect[0])))
        h = int(round(np.sqrt(target_area * ratio)))
        w = int(round(np.sqrt(target_area / ratio)))
        if 0 < h < height and 0 < w < width:
            top = int(random() * (height - h))
            left = int(random() * (width - w))
            model_input = model_input.clone()
            model_input[:, top:top + h, left:left + w] = torch.randn(model_input.shape[0], h, w)
            return model_input
    return model_input
