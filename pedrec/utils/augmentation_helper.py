import cv2
import numpy as np

from pedrec.models.data_structures import ImageSize


def get_normalization_size(size: ImageSize) -> np.ndarray:
    """
    Unbiased data processing (UDP): normalized coordinates refer to the first / last pixel center, i.e. a crop of
    width W spans [0, W - 1]. Pixel coordinates are divided by this size for the labels and multiplied for the
    predictions.
    """
    return np.array([size.width - 1, size.height - 1], dtype=np.float32)


def get_affine_transforms(center, scale, rot, output_size: ImageSize, add_inv: bool = False):
    """
    Affine transform (and optionally its inverse) from the image region given by ``center`` / ``scale`` (width,
    height in pixels) and rotation ``rot`` (degrees) to a crop of ``output_size``, following the unbiased data
    processing of Huang et al. ("The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose
    Estimation", CVPR 2020): pixel centers are aligned and the region is mapped onto (output_size - 1), so resizing
    and flipping do not shift the coordinates.
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    rot_rad = np.deg2rad(-rot)
    cos, sin = np.cos(rot_rad), np.sin(rot_rad)
    scale_x = (output_size.width - 1) / scale[0]
    scale_y = (output_size.height - 1) / scale[1]
    trans = np.zeros((2, 3), dtype=np.float64)
    trans[0, 0] = cos * scale_x
    trans[0, 1] = -sin * scale_x
    trans[0, 2] = scale_x * (-center[0] * cos + center[1] * sin + 0.5 * scale[0])
    trans[1, 0] = sin * scale_y
    trans[1, 1] = cos * scale_y
    trans[1, 2] = scale_y * (-center[0] * sin - center[1] * cos + 0.5 * scale[1])
    return trans, (cv2.invertAffineTransform(trans) if add_inv else None)


def get_affine_transform(center, scale, rot, output_size: ImageSize, inv=0):
    trans, trans_inv = get_affine_transforms(center, scale, rot, output_size, add_inv=bool(inv))
    return trans_inv if inv else trans


def affine_transform_pt(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def affine_transform_pts(pts, t):
    """
    expects pts_in: num_poses, num_joints, n (x, y, [...])
    """
    new_pts = np.ones((pts.shape[0], 3), dtype=np.float32)
    new_pts[:, :2] = pts[:, :2]
    # new_pts = np.array(pts.expand_dims())
    # new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pts = np.dot(t, new_pts.T).T
    return new_pts
