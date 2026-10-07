"""
The optimized inference path (pedrec.inference) must reproduce the original OpenCV / numpy implementation.
"""
import cv2
import numpy as np
import pytest
import torch
from torchvision import transforms

from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.configs.yolo_v4_config import YoloV4Config
from pedrec.inference import gpu_ops
from pedrec.inference.pipeline import PedRecPoseEstimator, RuntimeConfig
from pedrec.models.data_structures import ImageSize
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.networks.net_yolo_v4.yolo_layer import YoloLayer
from pedrec.networks.net_yolo_v4.yolo_v4_helper import post_processing
from pedrec.utils.augmentation_helper import get_affine_transforms
from pedrec.utils.bb_helper import bb_to_center_scale
from pedrec.utils.pose_deconv_helper import pedrec_recognizer

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


def test_crop_affine_matches_cv2_warp_affine():
    img = _random_frame()
    input_size = ImageSize(width=192, height=256)
    frame = gpu_ops.frame_to_tensor(img, torch.device("cpu"))
    _, trans_invs = gpu_ops.get_crop_transforms(BBS, input_size)
    crops = gpu_ops.crop_affine(frame, torch.from_numpy(trans_invs), input_size)
    for i, bb in enumerate(BBS):
        center, scale = bb_to_center_scale(bb, input_size)
        trans, _ = get_affine_transforms(center, scale, 0, input_size, add_inv=True)
        expected = IMAGENET(cv2.warpAffine(img, trans, (input_size.width, input_size.height), flags=cv2.INTER_LINEAR))
        diff = (crops[i] - expected).abs()
        # cv2 uses 5 bit fixed point interpolation weights and rounds to uint8
        assert diff.mean() < 0.01, diff.mean()
        assert torch.quantile(diff.flatten(), 0.999) < 0.1


def test_resize_matches_cv2():
    img = _random_frame(1080, 1920)
    size = YoloV4Config().model.input_size
    expected = cv2.resize(img, (size.width, size.height)).astype(np.float32)
    actual = gpu_ops.resize_bilinear(gpu_ops.frame_to_tensor(img, torch.device("cpu")), size)
    actual = actual[0].permute(1, 2, 0).numpy()
    assert np.abs(actual - expected).mean() < 0.6  # cv2 rounds to uint8


def test_pose_estimator_matches_legacy_recognizer():
    torch.manual_seed(0)
    cfg = PedRecNet50Config()
    net = PedRecNet(cfg)
    net.init_weights()
    net.eval()
    img = _random_frame()
    with torch.no_grad():
        legacy = pedrec_recognizer(net, cfg, img, BBS, torch.device("cpu"))
    estimator = PedRecPoseEstimator.__new__(PedRecPoseEstimator)
    estimator.cfg = cfg
    estimator.device = torch.device("cpu")
    estimator.run = lambda x: net(x)
    estimator._input_scale = torch.tensor([cfg.model.input_size.width, cfg.model.input_size.height],
                                          dtype=torch.float32)
    estimator._orientation_scale = torch.tensor([np.pi, 2 * np.pi], dtype=torch.float32)
    with torch.inference_mode():
        new = estimator(gpu_ops.frame_to_tensor(img, torch.device("cpu")), BBS)
    # 2D in image pixels, 3D in mm, orientation in radians; differences only from the crop interpolation
    assert np.abs(new["skeletons"][..., :2] - legacy["skeletons"][..., :2]).max() < 1.0
    assert np.abs(new["skeletons"][..., 2] - legacy["skeletons"][..., 2]).max() < 0.02
    assert np.abs(new["skeletons_3d"][..., :3] - legacy["skeletons_3d"][..., :3]).max() < 5.0
    assert np.abs(new["orientations"] - legacy["orientations"]).max() < 0.05


def _reference_yolo_decode(output, num_classes, anchors, num_anchors):
    """The original numpy based implementation of YoloLayer.yolo_forward_alternative."""
    batch, _, H, W = output.shape
    grid_x = np.expand_dims(np.linspace(0, W - 1, W), axis=0).repeat(H, 0).reshape(1, 1, H * W).repeat(batch, 0) \
        .repeat(num_anchors, 1)
    grid_y = np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(W, 1).reshape(1, 1, H * W).repeat(batch, 0) \
        .repeat(num_anchors, 1)
    anchor_array = np.expand_dims(np.array(anchors).reshape(1, num_anchors, 2).repeat(batch, 0), axis=3) \
        .repeat(H * W, 3)
    normal = np.array([1.0 / W, 1.0 / H, 1.0 / W, 1.0 / H], dtype=np.float32).reshape(1, 1, 4)
    bxy_list, bwh_list, det_list, cls_list = [], [], [], []
    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        bxy_list.append(output[:, begin: begin + 2])
        bwh_list.append(output[:, begin + 2: begin + 4])
        det_list.append(output[:, begin + 4: begin + 5])
        cls_list.append(output[:, begin + 5: (i + 1) * (5 + num_classes)])
    bxy = torch.sigmoid(torch.cat(bxy_list, 1)).view(batch, num_anchors, 2, H * W)
    bwh = torch.exp(torch.cat(bwh_list, 1)).view(batch, num_anchors, 2, H * W)
    det = torch.sigmoid(torch.cat(det_list, 1).view(batch, num_anchors * H * W))
    cls = torch.cat(cls_list, 1).view(batch, num_anchors, num_classes, H * W).permute(0, 1, 3, 2) \
        .reshape(batch, num_anchors * H * W, num_classes)
    cls = torch.softmax(cls, 2)
    bxy[:, :, 0] += torch.tensor(grid_x, dtype=torch.float32)
    bxy[:, :, 1] += torch.tensor(grid_y, dtype=torch.float32)
    bwh *= torch.tensor(anchor_array, dtype=torch.float32)
    boxes = torch.cat((bxy, bwh), 2).permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, 4) \
        * torch.tensor(normal)
    return boxes, cls * det.view(batch, num_anchors * H * W, 1)


def test_yolo_decode_matches_original():
    layer = YoloLayer(anchor_mask=[0, 1, 2], num_classes=80,
                      anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                      num_anchors=9, stride=8).eval()
    output = torch.randn(1, 3 * 85, 40, 76)
    boxes, confs = layer(output)
    anchors = [a / 8 for a in [12, 16, 19, 36, 40, 28]]
    ref_boxes, ref_confs = _reference_yolo_decode(output, 80, anchors, 3)
    assert torch.allclose(boxes, ref_boxes, atol=1e-5)
    assert torch.allclose(confs, ref_confs, atol=1e-6)


def test_yolo_postprocess_matches_original():
    rng = np.random.default_rng(1)
    num = 500
    centers = rng.random((num, 2))
    sizes = np.full((num, 2), 0.1)  # equal sizes: the original NMS used shifted boxes, identical IoU in this case
    confs = rng.random((num, 80)) * 0.5
    confs[:20, 0] = 0.9 + rng.random(20) * 0.1
    output = np.concatenate((centers, sizes, confs), axis=1)[None].astype(np.float32)
    img_size = ImageSize(1920, 1080)
    expected = sorted(map(tuple, np.round(np.array(post_processing(img_size, 0.4, 0.6, output)[0]), 3)))
    actual = sorted(map(tuple, np.round(np.array(gpu_ops.yolo_postprocess(torch.from_numpy(output), img_size, 0.4,
                                                                          0.6)), 3)))
    assert expected == actual
