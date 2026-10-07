import sys

sys.path.append('.')  # allow running as a script from the repository root

import math
import os

from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file
from pedrec.networks.net_pedrec.pose_resnet import PoseResNet
from pedrec.training.experiments.experiment_initializer import initialize_weights_with_same_name_and_shape

import cv2
from torch.utils.data import DataLoader

from pedrec.datasets.coco_dataset import CocoDataset
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths_home
from pedrec.configs.dataset_configs import get_coco_dataset_cfg_default
from pedrec.evaluations.eval_helper import get_total_coords
from pedrec.models.constants.dataset_constants import DatasetType
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.training.experiments.experiment_train_helper import init_experiment
from pedrec.utils.torch_utils.torch_helper import get_device, move_to_device
import pandas as pd
import numpy as np
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_final_preds(post_process, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if post_process:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals


def set_df_dtypes(df: pd.DataFrame):
    for joint in SKELETON_PEDREC_JOINTS:
        df[f"skeleton2d_{joint.name}_x"] = df[f"skeleton2d_{joint.name}_x"].astype("float32")
        df[f"skeleton2d_{joint.name}_y"] = df[f"skeleton2d_{joint.name}_y"].astype("float32")
        df[f"skeleton2d_{joint.name}_score"] = df[f"skeleton2d_{joint.name}_score"].astype("float32")
        df[f"skeleton2d_{joint.name}_visible"] = df[f"skeleton2d_{joint.name}_visible"].astype("category")
        df[f"skeleton2d_{joint.name}_supported"] = df[f"skeleton2d_{joint.name}_visible"].astype("category")

    df["body_orientation_phi"] = df["body_orientation_phi"].astype("float32")
    df["body_orientation_theta"] = df["body_orientation_theta"].astype("float32")
    df["body_orientation_score"] = df["body_orientation_score"].astype("float32")
    df["body_orientation_visible"] = df["body_orientation_visible"].astype("category")


def get_column_names():
    column_names = [
        "img_path",
        "img_size_w",
        "img_size_h",
        "center_x",
        "center_y",
        "scale_x",
        "scale_y"
    ]
    for joint in SKELETON_PEDREC_JOINTS:
        column_names.append(f"skeleton2d_{joint.name}_x")
        column_names.append(f"skeleton2d_{joint.name}_y")
        column_names.append(f"skeleton2d_{joint.name}_score")
        column_names.append(f"skeleton2d_{joint.name}_visible")
        column_names.append(f"skeleton2d_{joint.name}_supported")

    column_names.append("body_orientation_theta")
    column_names.append("body_orientation_phi")
    column_names.append("body_orientation_score")
    column_names.append("body_orientation_visible")

    return column_names


def initialize_pose_resnet(net, pose_resnet_weights_path: str):
    pose_resnet_state_dict = load_state_dict_file(pose_resnet_weights_path)
    net_weights = net.state_dict()
    for name, param in pose_resnet_state_dict.items():
        if name.startswith("final"):
            net_name = name.replace("final_layer.", "head_pose_2d.pose_heatmap_layer.")
            # net_name = f"head.pose_3d_heatmap_layer.{name}"
        elif name.startswith("deconv_layers.6") or name.startswith("deconv_layers.7"):
            net_name = f"head_pose_2d.{name.replace('deconv_layers', 'deconv_head').replace('6', '0').replace('7', '1')}"
        elif name.startswith("deconv"):
            net_name = f"conv_transpose_shared.{name}"
        else:
            net_name = f"feature_extractor.{name}"
        if net_name in net_weights:
            if net_weights[net_name].shape != param.shape and len(param.shape) == 1:
                net_weights[net_name][:17] = param
            elif net_weights[net_name].shape != param.shape and len(param.shape) == 4:
                net_weights[net_name][:17, :, :, :] = param
            elif net_weights[net_name].shape != param.shape:
                print(f"Shape mismatch in {net_name}: {net_weights[net_name].shape} <-> {param.shape}")
            else:
                net_weights[net_name] = param
        else:
            print(f"Skipped: {name}, tried: {net_name}")
    net.load_state_dict(net_weights)

def get_preds_mtl(outputs: torch.Tensor):
    return {
        "skeleton": outputs[0].cpu().detach().numpy(),
        # "skeleton_3d": outputs[1].cpu().detach().numpy(),
        # "orientation": outputs[2].cpu().detach().numpy(),
    }


def main(output_dir: str, output_postfix: str, net_cfg, coco_val: CocoDataset):
    init_experiment(42)
    device = get_device(use_gpu=True)

    ####################################################################################################################
    ############################################ Initialize Network ####################################################
    ####################################################################################################################
    net = PoseResNet(net_cfg)
    net.init_weights()
    initialize_pose_resnet(net, "data/models/human_pose_baseline/pose_resnet_50_256x192.pth.tar")

    net.to(device)
    # net.load_state_dict(torch.load("data/models/pedrec/experiment_pedrec_direct_4_net.pth"))

    batch_size = 1
    data_loader = DataLoader(coco_val, batch_size=batch_size, shuffle=False, num_workers=12)
    net.eval()
    gt_rows = []
    result_rows = []
    # count = 0
    column_names = get_column_names()
    with torch.no_grad():
        for test_data in data_loader:
            # if count > 2:
            #     break
            # count += 1
            images, labels = test_data
            images = images.to(device)
            labels = move_to_device(labels, device)
            outputs = net(images)
            # preds = get_preds_mtl(outputs)
            centers = labels["center"].cpu().detach().numpy()
            scales = labels["scale"].cpu().detach().numpy()

            pose_2d_preds, maxvals = get_final_preds(
                True, outputs.clone().cpu().numpy(), centers, scales)

            pose2d_preds = np.concatenate((pose_2d_preds, maxvals), axis=2)

            idxs = labels["idx"].cpu().detach().numpy()
            centers = labels["center"].cpu().detach().numpy()
            scales = labels["scale"].cpu().detach().numpy()
            rotations = labels["rotation"].cpu().detach().numpy()
            pose2d_gts = labels["skeleton"].cpu().detach().numpy()
            orientation_gts = labels["orientation"].cpu().detach().numpy()
            img_paths = labels["img_path"]
            img_sizes = labels["img_size"].cpu().detach().numpy()

            pose2d_gts = get_total_coords(pose2d_gts, net_cfg.model.input_size, centers, scales, rotations)


            # pose2d_preds = preds["skeleton"]
            # pose2d_preds = get_total_coords(pose2d_preds, net_cfg.model.input_size, centers, scales, rotations)

            # orientation_preds = preds["orientation"]

            for i in range(0, pose2d_preds.shape[0]):
                img_path = img_paths[i]
                img_size = img_sizes[i]
                idx = idxs[i]
                
                pose2d_pred = pose2d_preds[i]

                visibles = (pose2d_pred[:, 2] > 0.5).astype(np.int32)
                supported = np.ones(pose2d_pred.shape[0])
                visible_supported = np.array([visibles, supported]).transpose(1, 0)
                pose2d_pred = np.concatenate((pose2d_pred, visible_supported), axis=1)
                pose2d_pred = pose2d_pred.reshape(-1).tolist()
                
                pose2d_gt = pose2d_gts[i]
                pose2d_gt = pose2d_gt.reshape(-1).tolist()
                
                # orientation_pred = orientation_preds[i]
                # orientation_pred_body = orientation_pred[0].reshape(-1).tolist()
                
                orientation_gt = orientation_gts[i]
                orientation_gt_body = orientation_gt[0].reshape(-1).tolist()

                result_rows.append([img_path, img_size[0], img_size[1], centers[i, 0], centers[i, 1], scales[i, 0] / 200, scales[i, 1] / 200] + pose2d_pred + [0, 0, 0, 0])
                gt_rows.append([img_path, img_size[0], img_size[1], centers[i, 0], centers[i, 1], scales[i, 0] / 200, scales[i, 1] / 200] + pose2d_gt + [0, 0, 0, 0])

    df_pred = pd.DataFrame(data=result_rows, columns=get_column_names())
    set_df_dtypes(df_pred)
    output_path = os.path.join(output_dir, f"COCO_pred_df_{output_postfix}.pkl")
    df_pred.to_pickle(output_path)
    
    df_gt = pd.DataFrame(data=gt_rows, columns=get_column_names())
    set_df_dtypes(df_gt)
    output_path = os.path.join(output_dir, f"COCO_gt_df_{output_postfix}.pkl")
    df_gt.to_pickle(output_path)


if __name__ == '__main__':
    experiment_paths = get_experiment_paths_home()
    net_cfg = PedRecNet50Config()
    coco_val_dataset_cfg = get_coco_dataset_cfg_default()

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    coco_val_dataset_cfg.flip = False
    coco_val_dataset_cfg.scale_factor = 0
    coco_val_dataset_cfg.rotation_factor = 0
    coco_val_dataset_cfg.use_mebow_orientation = True
    coco_val = CocoDataset(experiment_paths.coco_dir, DatasetType.VALIDATE,
                           coco_val_dataset_cfg,
                           net_cfg.model.input_size,
                           trans)

    main(output_dir="data/datasets/COCO/", output_postfix="val_resnet", net_cfg=net_cfg, coco_val=coco_val)

    # coco_val = CocoDataset(experiment_paths.coco_dir, DatasetType.TRAIN,
    #                        coco_val_dataset_cfg,
    #                        net_cfg.model.input_size,
    #                        trans)
    #
    # main(output_dir="data/datasets/COCO/", output_postfix="train_direct_4_resnet", net_cfg=net_cfg,
    #      coco_val=coco_val)

