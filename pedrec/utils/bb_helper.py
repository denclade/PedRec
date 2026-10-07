import sys
from typing import Union, Tuple, List

import numpy as np
import torch
from torch import Tensor

from pedrec.models.data_structures import ImageSize

"""
center bb: center_x, center_y, width, height, confidence, class_idx
tl bb: tl_x, tl_y, width, height, confidence, class_idx
coord bb: tl_x, tl_y, br_x, br_y, confidence, class_idx
default: center bb
-1 to coordinates because we work on pixels, thus 0 counts as 1 and needs to be added to numeric values.
"""

# TODO: The iou code still uses coord bb as default, should be switched to center based

bb_type = Union[torch.tensor, np.ndarray, List[Union[float, int]]]


def split_human_bbs(bbs: List[bb_type]) -> Tuple[List[bb_type], List[bb_type]]:
    human_bbs = []
    other_bbs = []
    for bb in bbs:
        if len(bb) == 0 or get_bb_class_idx(bb) != 0:
            other_bbs.append(bb)
        else:
            human_bbs.append(bb)
    return human_bbs, other_bbs


def get_bb_class_idx(bb: bb_type) -> int:
    """
    Returns the class index of the bounding box
    :param bb:
    :return:
    """
    return int(bb[5])


def get_bb_score(bb: bb_type) -> float:
    """
    Returns the confidence value of a bounding box
    :param bb:
    :return:
    """
    return bb[4]


def get_bb_center(bb: bb_type) -> Tuple[float, float]:
    return bb[0], bb[1]


def get_bb_width(bb: bb_type) -> float:
    return bb[2]


def get_bb_height(bb: bb_type) -> float:
    return bb[3]


def get_denormalized_bb(bb: bb_type, img_size: ImageSize) -> bb_type:
    """
    Returns a bounding box with percentual values to a bounding box with actual image coordinates
    :param bb:
    :param img_size:
    :return:
    """
    assert min(bb) >= 0 and max(bb) <= 1
    bb_new = bb.copy()
    bb_new[0] = bb[0] * img_size.width
    bb_new[1] = bb[1] * img_size.height
    bb_new[2] = bb[2] * img_size.width
    bb_new[3] = bb[3] * img_size.height
    return bb_new


def get_empty_bb_of_same_type(bb: bb_type) -> bb_type:
    if isinstance(bb, np.ndarray):
        return np.zeros((6, 1), dtype=np.float32)
    elif isinstance(bb, Tensor):
        return torch.zeros((6, 1), dtype=torch.float32)
    else:
        return [0] * 6


def get_empty_bbs_of_same_type(bb: bb_type) -> bb_type:
    if isinstance(bb, np.ndarray):
        return np.zeros(bb.shape, dtype=np.float32)
    elif isinstance(bb, Tensor):
        return torch.zeros(bb.shape, dtype=torch.float)
    else:
        raise ValueError("bbs type only support a numpy array or tensor")


def get_img_coordinates_from_bb(bb: bb_type) -> (int, int, int, int):
    """
    returns bb1_x_tl, bb1_y_tl, bb1_x_br, bb1_y_br from center based bb (center_x, center_y, width, height)
    :param bb: center based bb
    :return:
    """
    x1, y1, x2, y2 = get_float_coordinates_from_bb(bb)
    return int(x1), int(y1), int(x2), int(y2)


def get_float_coordinates_from_bb(bb: bb_type) -> (float, float, float, float):
    x1 = bb[0] - bb[2] / 2.0
    y1 = bb[1] - bb[3] / 2.0
    x2 = bb[0] + bb[2] / 2.0
    y2 = bb[1] + bb[3] / 2.0
    return x1, y1, x2, y2


def get_float_coordinates_from_bbs(bbs: bb_type) -> (float, float, float, float):
    x1 = bbs[:, 0] - bbs[:, 2] / 2.0
    y1 = bbs[:, 1] - bbs[:, 3] / 2.0
    x2 = bbs[:, 0] + bbs[:, 2] / 2.0
    y2 = bbs[:, 1] + bbs[:, 3] / 2.0
    return x1, y1, x2, y2


def get_coord_bb_from_center_bb(bb: bb_type) -> bb_type:
    """
    returns coord bb
    :param bb: center bb
    :return:
    """
    coord_bb = get_empty_bb_of_same_type(bb)
    coord_bb[0] = bb[0] - bb[2] / 2.0
    coord_bb[1] = bb[1] - bb[3] / 2.0
    coord_bb[2] = bb[0] + bb[2] / 2.0
    coord_bb[3] = bb[1] + bb[3] / 2.0
    coord_bb[4] = bb[4]
    coord_bb[5] = bb[5]
    return coord_bb


def get_center_bb_from_tl_bb(bb: bb_type) -> bb_type:
    """
    returns coord bb
    :param bb: center bb
    :return:
    """
    bb_len = len(bb)
    center_bb = get_empty_bb_of_same_type(bb)
    center_bb[0] = bb[0] + (bb[2] / 2.0)
    center_bb[1] = bb[1] + (bb[3] / 2.0)
    center_bb[2] = bb[2]
    center_bb[3] = bb[3]
    if bb_len > 4:
        center_bb[4] = bb[4]
    if bb_len > 5:
        center_bb[5] = bb[5]
    return center_bb


def get_center_bb_from_coord_bb(bb: bb_type) -> bb_type:
    """
    returns coord bb
    :param bb: center bb
    :return:
    """
    bb_len = len(bb)
    center_bb = get_empty_bb_of_same_type(bb)
    width = abs(bb[2] - bb[0])
    height = abs(bb[3] - bb[1])
    center_bb[0] = bb[0] + (width / 2.0)
    center_bb[1] = bb[1] + (height / 2.0)
    center_bb[2] = width
    center_bb[3] = height
    if bb_len > 4:
        center_bb[4] = bb[4]
    if bb_len > 5:
        center_bb[5] = bb[5]
    return center_bb


def get_coord_bb_from_center_bbs(bbs: bb_type) -> bb_type:
    """
    returns coord bb
    :param bb: center bb
    :return:
    """
    coord_bbs = get_empty_bbs_of_same_type(bbs)
    coord_bbs[:, 0] = bbs[:, 0] - bbs[:, 2] / 2.0
    coord_bbs[:, 1] = bbs[:, 1] - bbs[:, 3] / 2.0
    coord_bbs[:, 2] = bbs[:, 0] + bbs[:, 2] / 2.0
    coord_bbs[:, 3] = bbs[:, 1] + bbs[:, 3] / 2.0
    coord_bbs[:, 4] = bbs[:, 4]
    coord_bbs[:, 5] = bbs[:, 5]
    return coord_bbs


def get_bb_coords(bb: bb_type) -> (float, float, float, float):
    """
    returns bb1_x_tl, bb1_y_tl, bb1_x_br, bb1_y_br
    :param bb: the bounding box with shape [4]
    :return:
    """
    return get_float_coordinates_from_bb(bb)


def get_bbs_coords(bbs: bb_type) -> (float, float, float, float):
    """
    returns bb1_x_tl, bb1_y_tl, bb1_x_br, bb1_y_br
    :param bb: the bounding boxes with shape [batch_size, 4]
    :return:
    """
    return get_float_coordinates_from_bbs(bbs)


def bbox_iou_torch(bb1: torch.Tensor, bb2: torch.Tensor, device: torch.device) -> torch.tensor:
    """
    Returns the IoU of two bounding boxes.
    :param bb1: torch tensor with shape [4] 
    :param bb2: torch tensor with shape [4] 
    :param device: torch device to work on
    """
    return __bb_iou_torch_impl(*get_bb_coords(bb1), *get_bb_coords(bb2), device=device)


def bbox_ious_torch(bbs1: torch.Tensor, bbs2: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Returns the IoU of two bounding boxes for the given batches.
    :param bbs1: torch tensor with shape [batch_size, 4]
    :param bbs2: torch tensor with shape [batch_size, 4]
    :param device: torch device to work on
    """
    return __bb_iou_torch_impl(*get_bbs_coords(bbs1), *get_bbs_coords(bbs2), device=device)


def __bb_iou_torch_impl(bb1_x_tl: torch.Tensor, bb1_y_tl: torch.Tensor,
                        bb1_x_br: torch.Tensor, bb1_y_br: torch.Tensor,
                        bb2_x_tl: torch.Tensor, bb2_y_tl: torch.Tensor,
                        bb2_x_br: torch.Tensor, bb2_y_br: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
    # get the coordinates of the intersection box
    inter_box_x_tl = torch.max(bb1_x_tl, bb2_x_tl)
    inter_box_y_tl = torch.max(bb1_y_tl, bb2_y_tl)
    inter_box_x_br = torch.min(bb1_x_br, bb2_x_br)
    inter_box_y_br = torch.min(bb1_y_br, bb2_y_br)

    zeroes = torch.zeros(inter_box_x_br.shape).to(device)

    inter_area = torch.max(inter_box_x_br - inter_box_x_tl + 1, zeroes) * \
                 torch.max(inter_box_y_br - inter_box_y_tl + 1, zeroes)

    # Union Area
    bb1_area = (bb1_x_br - bb1_x_tl + 1) * (bb1_y_br - bb1_y_tl + 1)
    bb2_area = (bb2_x_br - bb2_x_tl + 1) * (bb2_y_br - bb2_y_tl + 1)
    union_area = bb1_area + bb2_area - inter_area

    iou = inter_area / union_area

    return iou


def bb_iou_numpy(bb1: np.ndarray, bb2: np.ndarray) -> float:
    """
    Returns the IoU of two bounding boxes.
    :param bb1: numpy array with shape [4]
    :param bb2: numpy array with shape [4]
    """
    return __bb_iou_numpy(*get_bb_coords(bb1), *get_bb_coords(bb2))


def bb_ious_numpy(bbs1: np.ndarray, bbs2: np.ndarray) -> Union[float, np.ndarray]:
    """
    Returns the IoU of two bounding boxes.
    :param bbs1: numpy array with shape [batch_size, 4]
    :param bbs2: numpy array with shape [batch_size, 4]
    """
    return __bb_iou_numpy(*get_bbs_coords(bbs1), *get_bbs_coords(bbs2))


def __bb_iou_numpy(bb1_x_tl: Union[float, int, np.ndarray], bb1_y_tl: Union[float, int, np.ndarray],
                   bb1_x_br: Union[float, int, np.ndarray], bb1_y_br: Union[float, int, np.ndarray],
                   bb2_x_tl: Union[float, int, np.ndarray], bb2_y_tl: Union[float, int, np.ndarray],
                   bb2_x_br: Union[float, int, np.ndarray], bb2_y_br: Union[float, int, np.ndarray]) -> float:
    # determine the (x, y)-coordinates of the inter rectangle
    inter_box_x_tl = max(bb1_x_tl, bb2_x_tl)
    inter_box_y_tl = max(bb1_y_tl, bb2_y_tl)
    inter_box_x_br = min(bb1_x_br, bb2_x_br)
    inter_box_y_br = min(bb1_y_br, bb2_y_br)
    # compute the area of inter rectangle
    inter_area = max(inter_box_x_br - inter_box_x_tl + 1, 0) * max(inter_box_y_br - inter_box_y_tl + 1, 0)
    # compute the area of both the prediction and ground-truth
    # rectangles
    bb1_area = (bb1_x_br - bb1_x_tl + 1) * (bb1_y_br - bb1_y_tl + 1)
    bb2_area = (bb2_x_br - bb2_x_tl + 1) * (bb2_y_br - bb2_y_tl + 1)
    union_area = bb1_area + bb2_area - inter_area
    # compute the inter over union by taking the inter
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / union_area
    # return the inter over union value
    return iou


def get_region_boxes(boxes_and_confs: torch.Tensor) -> torch.Tensor:
    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)

    output = torch.cat((boxes, confs), dim=2)

    return output


def get_human_bb_from_joints(joints: np.ndarray, max_x_val: int = sys.maxsize,
                             max_y_val: int = sys.maxsize, confidence: float = 0, class_idx: int = None, expand: float = 0.15):
    """
    Center bb around all joints (x, y in the first two columns), expanded by ``expand`` of the width / height and
    clipped to the image.
    """
    min_x = min(float(np.min(joints[:, 0])), sys.maxsize)
    min_y = min(float(np.min(joints[:, 1])), sys.maxsize)
    max_x = max(float(np.max(joints[:, 0])), 0)
    max_y = max(float(np.max(joints[:, 1])), 0)

    bb_expand_width = expand * (max_x - min_x)
    bb_expand_height = expand * (max_y - min_y)

    min_x = max(min_x - bb_expand_width, 0)
    min_y = max(min_y - bb_expand_height, 0)
    max_x = min(max_x + bb_expand_width, max_x_val)
    max_y = min(max_y + bb_expand_height, max_y_val)

    width = max_x - min_x
    height = max_y - min_y
    return [min_x + (width / 2), min_y + (height / 2), width, height, confidence, class_idx]


def bbs_to_centers_scales(bbs_orig: np.ndarray, input_size: ImageSize):
    """
    vectorized form of bb_to_center_scale
    """
    bbs = bbs_orig.copy()
    # box_widths = bbs[:, 2]
    # box_heights = bbs[:, 3]

    aspect_ratio = input_size.width * 1.0 / input_size.height
    width_larger_filter = bbs[:, 2] > aspect_ratio * bbs[:, 3]
    width_smaller_filter = bbs[:, 2] < aspect_ratio * bbs[:, 3]
    # bbs[bbs[:, 2] > aspect_ratio * box_heights][:, 3] = bbs[bbs[:, 2] > aspect_ratio * box_heights][:, 2] / aspect_ratio
    # bbs[bbs[:, 2] < aspect_ratio * box_heights][:, 2] = bbs[bbs[:, 2] < aspect_ratio * box_heights][:, 3] * aspect_ratio
    bb_width_filtered = bbs[width_larger_filter]
    bb_height_filtered = bbs[width_smaller_filter]
    bb_width_filtered[:, 3] = bb_width_filtered[:, 2] / aspect_ratio
    bb_height_filtered[:, 2] = bb_height_filtered[:, 3] * aspect_ratio
    bbs[width_larger_filter] = bb_width_filtered
    bbs[width_smaller_filter] = bb_height_filtered
    # box_heights[box_widths > aspect_ratio * box_heights] = box_widths
    # if box_widths > aspect_ratio * box_heights:
    #     box_heights = box_widths / aspect_ratio
    # else:
    #     box_widths = box_heights * aspect_ratio
    centers = bbs[:, 0:2]
    scales = bbs[:, 2:4] * 1.25
    # bbs[:, 2:] *= 1.25
    return centers, scales

def bb_to_center_scale(bb: np.ndarray, input_size: ImageSize):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    bb : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    # center = np.zeros((2), dtype=np.float32)

    box_width = get_bb_width(bb)
    box_height = get_bb_height(bb)
    center = np.array(get_bb_center(bb))

    aspect_ratio = input_size.width / input_size.height

    if box_width > aspect_ratio * box_height:
        box_height = box_width / aspect_ratio
    else:
        box_width = box_height * aspect_ratio
    scale = np.array([box_width, box_height], dtype=np.float32) * 1.25

    return center, scale


def bb_nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def get_bbs_above_score(human_bbs, score: float):
    return [human_bb for human_bb in human_bbs if
            human_bb[-1] != -1 or get_bb_score(
                human_bb) > score]  # only keep humans with high score, because we have tracking


if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print('Working on GPU: {}'.format(torch.cuda.get_device_name(0)))
    # else:

    device = torch.device("cpu")
    b1 = [2, 1, 5, 5]  # x_tl, y_tl, x_br, y_br
    b2 = [3, 2, 6, 7]
    iou_t1 = bbox_iou_torch(torch.FloatTensor(b1).to(device), torch.FloatTensor(b2).to(device), device)
    iou_t2 = bbox_ious_torch(torch.FloatTensor([b1, b1]).to(device), torch.FloatTensor([b2, b2]).to(device), device)
    iou_n1 = bb_iou_numpy(np.array(b1), np.array(b2))
    iou_n2 = bb_ious_numpy(np.array([b1, b1]), np.array([b2, b2]))
    a = 1
    # start = time.time()
    # for i in range(0, 100000):
    #     # iou_t1 = bbox_iou_torch(torch.FloatTensor([b1]).to(device), torch.FloatTensor([b2]).to(device), device)
    #     test2 = bb_iou_numpy(np.array(b1), np.array(b2))
    # end = time.time()
    # print("Required: {}".format(end-start))

    a = 1
