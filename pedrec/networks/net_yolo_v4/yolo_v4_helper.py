import logging
import time
from typing import List

import numpy as np
import torch

from pedrec.configs.yolo_v4_config import YoloV4Config
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.networks.net_yolo_v4.yolov4 import YoloV4
from pedrec.utils.bb_helper import bb_nms_cpu

logger = logging.getLogger(__name__)


def post_processing(img_size: ImageSize, conf_thresh: float, nms_thresh: float, output: torch.FloatTensor,
                    tracked_bbs: np.ndarray = None):
    t1 = time.time()

    if type(output).__name__ != 'ndarray':
        output = output.cpu().detach().numpy()

    # [batch, num, 4]
    box_array = output[:, :, :4]

    # [batch, num, num_classes]
    confs = output[:, :, 4:]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)
    # uids = np.ones(max_id.shape, dtype=np.int) * -1

    if tracked_bbs is not None:
        box_array = np.append(box_array, tracked_bbs[:, :, :4], axis=1)
        max_conf = np.append(max_conf, tracked_bbs[:, :, 4], axis=1)
        max_id = np.append(max_id, tracked_bbs[:, :, 5].astype(np.int64), axis=1)
    #     uids = np.append(uids, tracked_bbs[:, :, 6].astype(np.int), axis=1)
    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]
        # l_uid = uids[i, argwhere]

        keep = bb_nms_cpu(l_box_array, l_max_conf, nms_thresh)

        bboxes = []
        if keep.size > 0:
            l_box_array = l_box_array[keep, :]
            l_max_conf = l_max_conf[keep]
            l_max_id = l_max_id[keep]
            # l_uid = l_uid[keep_uid]

            for j in range(l_box_array.shape[0]):
                bboxes.append(
                    [l_box_array[j, 0] * img_size.width,
                     l_box_array[j, 1] * img_size.height,
                     l_box_array[j, 2] * img_size.width,
                     l_box_array[j, 3] * img_size.height,
                     l_max_conf[j],
                     l_max_id[j]])

        bboxes_batch.append(bboxes)

    t3 = time.time()

    logger.debug('       max and argmax : %f' % (t2 - t1))
    logger.debug('                  nms : %f' % (t3 - t2))
    logger.debug('Post processing total : %f' % (t3 - t1))

    return bboxes_batch


def get_yolo_v4_zero_img(cfg: YoloV4Config, device: torch.device):
    img = np.zeros((cfg.model.input_size.width, cfg.model.input_size.height, 3), dtype=np.uint8)
    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose((0, 3, 1, 2))).float().div(255.0)
    else:
        logger.error("unknow image type")
        exit(-1)

    img = img.to(device)
    return img


def do_detect(model: YoloV4,
              img: np.ndarray,
              img_size: ImageSize,
              conf_thresh: float,
              nms_thresh: float,
              device: torch.device,
              logger: logging.Logger,
              tracked_humans: List[Human] = []):
    t0 = time.time()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose((0, 3, 1, 2))).float().div(255.0)
    else:
        logger.error("unknow image type")
        exit(-1)

    img = img.to(device)

    t1 = time.time()

    output = model(img)

    t2 = time.time()

    logger.debug('           Preprocess : %f' % (t1 - t0))
    logger.debug('      Model Inference : %f' % (t2 - t1))

    if len(tracked_humans) > 0:
        tracked_bbs = np.array([np.array(human.bb, dtype=np.float32) for human in tracked_humans])
        tracked_bbs[:, 0] = tracked_bbs[:, 0] / img_size.width
        tracked_bbs[:, 1] = tracked_bbs[:, 1] / img_size.height
        tracked_bbs[:, 2] = tracked_bbs[:, 2] / img_size.width
        tracked_bbs[:, 3] = tracked_bbs[:, 3] / img_size.height
        tracked_bbs = np.expand_dims(tracked_bbs, axis=0)
    else:
        tracked_bbs = None
    return post_processing(img_size, conf_thresh, nms_thresh, output, tracked_bbs)
