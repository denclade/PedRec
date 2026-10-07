import numpy as np

from pedrec.models.constants.skeleton_coco import SKELETON_COCO_JOINTS
from pedrec.models.constants.skeleton_h36m import SKELETON_H36M_JOINTS, SKELETON_H36M_HANDFOOTENDS_JOINTS
from pedrec.models.data_structures import ImageSize
from pedrec.utils.augmentation_helper import get_affine_transform, affine_transform_pt, get_normalization_size


def get_total_coords(coords_orig: np.ndarray, model_input_size: ImageSize, centers, scales, rotations):
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
    coords[:, :, :2] *= get_normalization_size(model_input_size)
    for i in range(coords.shape[0]):
        trans = get_affine_transform(centers[i], scales[i], rotations[i], model_input_size, inv=1)
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