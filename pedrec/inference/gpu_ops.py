"""
Batched, device side replacements for the per person OpenCV / numpy pre- and post-processing of the inference path.

All functions work on CPU and CUDA tensors. They reproduce the original operations:

* ``crop_affine``           == ``cv2.warpAffine(img, trans, size, flags=cv2.INTER_LINEAR)`` (border value 0) per person,
                               followed by ``ToTensor`` + ``Normalize`` (done after sampling, so the zero border is
                               normalized exactly as before)
* ``resize_bilinear``       == ``cv2.resize(img, size)`` (INTER_LINEAR, half pixel centers, no antialiasing)
* ``transform_coords_2d``   == ``affine_transform_coords_2d`` for all persons at once
* ``yolo_postprocess``      == ``yolo_v4_helper.post_processing`` with a GPU NMS
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from pedrec.models.data_structures import ImageSize
from pedrec.utils.augmentation_helper import get_affine_transforms
from pedrec.utils.bb_helper import bb_to_center_scale

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_base_grid_cache = {}
_norm_cache = {}


def frame_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """HWC uint8 RGB frame -> 1x3xHxW float32 tensor with values in [0, 255] on ``device``."""
    tensor = torch.from_numpy(np.ascontiguousarray(img))
    if device.type == "cuda":
        tensor = tensor.pin_memory().to(device, non_blocking=True)
    return tensor.permute(2, 0, 1).unsqueeze(0).float()


def resize_bilinear(frame: torch.Tensor, size: ImageSize) -> torch.Tensor:
    """Bilinear resize with the same sampling positions as cv2.INTER_LINEAR."""
    return F.interpolate(frame, size=(size.height, size.width), mode="bilinear", align_corners=False,
                         antialias=False)


def _get_base_grid(out_size: ImageSize, device: torch.device) -> torch.Tensor:
    key = (out_size.width, out_size.height, str(device))
    if key not in _base_grid_cache:
        ys, xs = torch.meshgrid(torch.arange(out_size.height, device=device, dtype=torch.float32),
                                torch.arange(out_size.width, device=device, dtype=torch.float32), indexing="ij")
        # homogeneous output pixel coordinates (pixel centers at integer positions, as in OpenCV)
        _base_grid_cache[key] = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1).view(-1, 3)
    return _base_grid_cache[key]


def _get_norm(device: torch.device, mean: Sequence[float], std: Sequence[float]):
    key = (str(device), tuple(mean), tuple(std))
    if key not in _norm_cache:
        mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1) * 255.0
        std_t = torch.tensor(std, device=device).view(1, 3, 1, 1) * 255.0
        _norm_cache[key] = (mean_t, std_t)
    return _norm_cache[key]


def crop_affine(frame: torch.Tensor, trans_invs: torch.Tensor, out_size: ImageSize,
                mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD) -> torch.Tensor:
    """
    Crops ``B`` persons out of ``frame`` (1x3xHxW, values 0..255).

    :param trans_invs: Bx2x3 affine transforms mapping crop pixel coordinates to frame pixel coordinates (the inverse
        of the transform that would be passed to cv2.warpAffine)
    :return: Bx3x out_size.height x out_size.width normalized model input
    """
    batch = trans_invs.shape[0]
    _, _, height, width = frame.shape
    base_grid = _get_base_grid(out_size, frame.device)  # (h*w, 3)
    src = torch.matmul(base_grid.unsqueeze(0), trans_invs.transpose(1, 2))  # (B, h*w, 2) frame pixel coordinates
    # pixel coordinates -> normalized grid coordinates for align_corners=False
    scale = torch.tensor([2.0 / width, 2.0 / height], device=frame.device)
    grid = (src * scale + scale / 2 - 1).view(batch, out_size.height, out_size.width, 2)
    crops = F.grid_sample(frame.expand(batch, -1, -1, -1), grid, mode="bilinear", padding_mode="zeros",
                          align_corners=False)
    mean_t, std_t = _get_norm(frame.device, mean, std)
    return (crops - mean_t) / std_t


def get_crop_transforms(bbs: Sequence[np.ndarray], input_size: ImageSize,
                        udp: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Affine transforms (frame -> crop and crop -> frame, each Bx2x3) for the human bbs, identical to the transforms
    used during training / in the original inference code (bb_to_center_scale + get_affine_transforms).
    """
    trans, trans_invs = [], []
    for bb in bbs:
        center, scale = bb_to_center_scale(bb, input_size)
        t, t_inv = get_affine_transforms(center, scale, 0, input_size, add_inv=True, udp=udp)
        trans.append(t)
        trans_invs.append(t_inv)
    return np.asarray(trans, dtype=np.float32), np.asarray(trans_invs, dtype=np.float32)


def transform_coords_2d(coords: torch.Tensor, trans_invs: torch.Tensor) -> torch.Tensor:
    """Applies per person 2x3 affine transforms to Bx J x 2 coordinates."""
    ones = torch.ones_like(coords[..., :1])
    return torch.matmul(torch.cat((coords, ones), dim=-1), trans_invs.transpose(1, 2))


def yolo_postprocess(output: torch.Tensor, img_size: ImageSize, conf_thresh: float, nms_thresh: float,
                     tracked_bbs: Optional[np.ndarray] = None) -> List[List[float]]:
    """
    Post-processing of the YoloV4 output (1 x N x (4 + num_classes)) for a single image, on the device.

    :param tracked_bbs: optional Kx6 array with bbs of tracked humans (center x, center y, w, h, score, class idx) in
        image coordinates, added as candidates before the NMS (as in the original implementation)
    :return: list of bbs (center x, center y, width, height, confidence, class idx) in image coordinates
    """
    boxes = output[0, :, :4].float()
    max_conf, max_id = output[0, :, 4:].float().max(dim=1)
    if tracked_bbs is not None and len(tracked_bbs) > 0:
        tracked = torch.as_tensor(np.asarray(tracked_bbs, dtype=np.float32), device=boxes.device)
        norm = torch.tensor([img_size.width, img_size.height, img_size.width, img_size.height], device=boxes.device)
        boxes = torch.cat((boxes, tracked[:, :4] / norm), dim=0)
        max_conf = torch.cat((max_conf, tracked[:, 4]), dim=0)
        max_id = torch.cat((max_id, tracked[:, 5].long()), dim=0)
    keep = max_conf > conf_thresh
    boxes, max_conf, max_id = boxes[keep], max_conf[keep], max_id[keep]
    if boxes.shape[0] == 0:
        return []
    # class agnostic NMS (as before) on corner coordinates
    xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1)
    keep_idx = torchvision.ops.nms(xyxy, max_conf, nms_thresh)
    scale = torch.tensor([img_size.width, img_size.height, img_size.width, img_size.height], device=boxes.device)
    result = torch.cat((boxes[keep_idx] * scale, max_conf[keep_idx, None], max_id[keep_idx, None].float()), dim=1)
    return result.cpu().tolist()
