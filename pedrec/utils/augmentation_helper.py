import cv2
import numpy as np

from pedrec.models.data_structures import ImageSize

def get_udp_affine_transform(center, scale, rot, output_size: ImageSize) -> np.ndarray:
    """
    Unbiased data processing (Huang et al., "The Devil is in the Details: Delving into Unbiased Data Processing for
    Human Pose Estimation", CVPR 2020): pixel centers are aligned, the source region is mapped onto
    (output_size - 1) instead of output_size, so flipping and resizing do not shift the coordinates.
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    rot_rad = np.deg2rad(-rot)  # same rotation direction as get_affine_transforms
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
    return trans


def get_affine_transforms(
        center, scale, rot, output_size: ImageSize,
        shift=np.array([0, 0], dtype=np.float32), add_inv: bool = False, udp: bool = False
):
    if udp:
        trans = get_udp_affine_transform(center, scale, rot, output_size)
        return trans, (cv2.invertAffineTransform(trans) if add_inv else None)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size.width
    dst_h = output_size.height

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    trans_inv = None
    if add_inv:
        trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, trans_inv

def get_affine_transform(
        center, scale, rot, output_size: ImageSize,
        shift=np.array([0, 0], dtype=np.float32), inv=0, udp: bool = False
):
    if udp:
        trans = get_udp_affine_transform(center, scale, rot, output_size)
        return cv2.invertAffineTransform(trans) if inv else trans
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size.width
    dst_h = output_size.height

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


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


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img
