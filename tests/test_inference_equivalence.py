"""
The batched inference path (pedrec.inference) must reproduce a straightforward OpenCV / numpy implementation with the
transforms used in training (UDP).
"""
import cv2
import numpy as np
import torch
from torchvision import transforms

from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.inference import gpu_ops
from pedrec.inference.pipeline import PedRecPoseEstimator, RuntimeConfig, SKELETON_3D_RANGE
from pedrec.models.data_structures import ImageSize
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.utils.augmentation_helper import get_affine_transforms, get_normalization_size
from pedrec.utils.bb_helper import bb_to_center_scale

IMAGENET = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def _random_frame(height=360, width=640, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((height, width, 3)) * 255).astype(np.uint8), (7, 7), 2)
    cv2.rectangle(img, (100, 60), (220, 320), (200, 180, 160), -1)
    return img


BBS = [np.array([160, 190, 130, 270, 0.9, 0], dtype=np.float32),
       np.array([400, 200, 60, 150, 0.8, 0], dtype=np.float32),
       np.array([620, 340, 200, 300, 0.7, 0], dtype=np.float32)]  # partially outside the image


def _cv2_crop(img, bb, input_size):
    center, scale = bb_to_center_scale(bb, input_size)
    trans, trans_inv = get_affine_transforms(center, scale, 0, input_size, add_inv=True)
    crop = cv2.warpAffine(img, trans, (input_size.width, input_size.height), flags=cv2.INTER_LINEAR)
    return IMAGENET(crop), trans_inv


def test_crop_affine_matches_cv2_warp_affine():
    img = _random_frame()
    input_size = ImageSize(width=192, height=256)
    frame = gpu_ops.frame_to_tensor(img, torch.device("cpu"))
    _, trans_invs = gpu_ops.get_crop_transforms(BBS, input_size)
    crops = gpu_ops.crop_affine(frame, torch.from_numpy(trans_invs), input_size)
    for i, bb in enumerate(BBS):
        expected, _ = _cv2_crop(img, bb, input_size)
        diff = (crops[i] - expected).abs()
        # cv2 uses 5 bit fixed point interpolation weights and rounds to uint8
        assert diff.mean() < 0.01, diff.mean()
        assert torch.quantile(diff.flatten(), 0.999) < 0.1


def test_resize_matches_cv2():
    img = _random_frame(1080, 1920)
    size = ImageSize(640, 640)  # RT-DETR input
    expected = cv2.resize(img, (size.width, size.height)).astype(np.float32)
    actual = gpu_ops.resize_bilinear(gpu_ops.frame_to_tensor(img, torch.device("cpu")), size)
    actual = actual[0].permute(1, 2, 0).numpy()
    assert np.abs(actual - expected).mean() < 0.6  # cv2 rounds to uint8


def _reference_poses(net, img, bbs, input_size):
    """Per person: cv2 crop -> PedRecNet -> UDP de-normalization -> inverse affine transform."""
    skeletons, skeletons_3d, orientations = [], [], []
    for bb in bbs:
        crop, trans_inv = _cv2_crop(img, bb, input_size)
        with torch.no_grad():
            outputs = net(crop[None])
        pose_2d = outputs[0][0].numpy()
        xy = pose_2d[:, :2] * get_normalization_size(input_size)
        xy = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=1) @ trans_inv.T
        skeletons.append(np.concatenate((xy, pose_2d[:, 2:]), axis=1))
        pose_3d = outputs[1][0].numpy()
        skeletons_3d.append(np.concatenate((pose_3d[:, :3] * SKELETON_3D_RANGE - SKELETON_3D_RANGE / 2,
                                            pose_3d[:, 3:]), axis=1))
        orientations.append(outputs[2][0].numpy() * np.array([np.pi, 2 * np.pi]))
    return np.stack(skeletons), np.stack(skeletons_3d), np.stack(orientations)


def test_pose_estimator_matches_reference():
    torch.manual_seed(0)
    net = PedRecNet(PedRecNet50Config())
    net.init_weights()
    net.eval()
    img = _random_frame()
    estimator = PedRecPoseEstimator("unused", torch.device("cpu"), RuntimeConfig(), net=net)
    with torch.inference_mode():
        new = estimator(gpu_ops.frame_to_tensor(img, torch.device("cpu")), BBS)
    skeletons, skeletons_3d, orientations = _reference_poses(net, img, BBS, net.cfg.model.input_size)
    # 2D in image pixels, 3D in mm, orientation in radians; differences only from the crop interpolation
    assert np.abs(new["skeletons"][..., :2] - skeletons[..., :2]).max() < 1.0
    assert np.abs(new["skeletons"][..., 2] - skeletons[..., 2]).max() < 0.02
    assert np.abs(new["skeletons_3d"][..., :3] - skeletons_3d[..., :3]).max() < 5.0
    assert np.abs(new["orientations"] - orientations).max() < 0.05
