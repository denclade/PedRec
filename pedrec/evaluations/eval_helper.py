import numpy as np

from pedrec.models.constants.skeleton_coco import SKELETON_COCO_JOINTS
from pedrec.models.constants.skeleton_h36m import SKELETON_H36M_JOINTS, SKELETON_H36M_HANDFOOTENDS_JOINTS
from pedrec.models.data_structures import ImageSize
from pedrec.utils.augmentation_helper import get_affine_transform, affine_transform_pt


def get_total_coords(coords_orig: np.ndarray, model_input_size: ImageSize, centers, scales, rotations,
                     udp: bool = False):
    """
    Inverts the affine transformation used on the GT img (scale, rotation, bb cut, ...) and returns
    the coordinates in the GT image.
    :param coords:
    :param cfg:
    :param centers:
    :param scales:
    :param rotations:
    :return:
    """
    coords = coords_orig.copy()
    offset = 1 if udp else 0
    coords[:, :, 0] *= model_input_size.width - offset
    coords[:, :, 1] *= model_input_size.height - offset
    for i in range(coords.shape[0]):
        trans = get_affine_transform(centers[i], scales[i], rotations[i], model_input_size, inv=1, udp=udp)
        for p in range(coords[i].shape[0]):
            coords[i, p, 0:2] = affine_transform_pt(coords[i, p, 0:2], trans)
    return coords

def get_skel_h36m(skeleton_2ds):
    skel_h36m = np.zeros((skeleton_2ds.shape[0], 17, 5), dtype=np.float32)
    for idx, joint in enumerate(SKELETON_H36M_JOINTS):
        skel_h36m[:, idx, :] = skeleton_2ds[:, joint.value, :]
    return skel_h36m

def get_skel_h36m_handfootends(skeleton_2ds):
    skel_h36m = np.zeros((skeleton_2ds.shape[0], 21, 5), dtype=np.float32)
    for idx, joint in enumerate(SKELETON_H36M_HANDFOOTENDS_JOINTS):
        skel_h36m[:, idx, :] = skeleton_2ds[:, joint.value, :]
    return skel_h36m

def get_skel_coco(skeleton_2ds):
    skel_coco = np.zeros((skeleton_2ds.shape[0], 17, 5), dtype=np.float32)
    for idx, joint in enumerate(SKELETON_COCO_JOINTS):
        skel_coco[:, idx, :] = skeleton_2ds[:, joint.value, :]
    return skel_coco