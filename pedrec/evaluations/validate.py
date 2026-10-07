import math
import time
from typing import List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from tqdm import tqdm

from pedrec.evaluations.eval_helper import get_total_coords
from pedrec.evaluations.eval_np.eval_angular import get_angular_error_statistics
from pedrec.evaluations.eval_np.eval_classification import get_binary_multi_label_accuracy, get_joint_label_accuracy
from pedrec.evaluations.eval_np.eval_pose_2d import get_pck_results
from pedrec.evaluations.eval_np.eval_pose_3d import get_dim_distances, get_mpjpe, get_relative_correct_depth, \
    get_relative_correct_joint_positions
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT, SKELETON_PEDREC_PARENT_CHILD_PAIRS
from pedrec.models.data_structures import ImageSize
from pedrec.models.validation.env_position_validation_results import EnvPositionValidationResults
from pedrec.models.validation.orientation_validation_results import OrientationValidationResult, \
    OrientationValidationResults
from pedrec.models.validation.pose_2d_validation_conf_results import Pose2DValidationConfResults
from pedrec.models.validation.pose_2d_validation_pck_results import Pose2DValidationPCKResults
from pedrec.models.validation.pose_3d_validation_results import Pose3DValidationResults
from pedrec.models.validation.validation_results import ValidationResults
from pedrec.utils.print_helper import get_heading
from pedrec.utils.torch_utils.torch_helper import move_to_device


def get_pedrec_preds(outputs: torch.Tensor):
    return {
        "skeleton": outputs[0].cpu().detach().numpy(),
        "skeleton_3d": outputs[1].cpu().detach().numpy(),
        "orientation": outputs[2].cpu().detach().numpy()
    }


def get_outputs_loss_pedrec(net: nn.Module, model_input: torch.Tensor, labels: torch.Tensor):
    outputs = net(model_input)
    loss = 0  # just dummy loss, because MTL net is not available
    return outputs, loss


def get_outputs_loss_mtl(net: nn.Module, model_input: torch.Tensor, labels: torch.Tensor):
    return net(model_input, labels)


# def main(pose_weights: str):
#     configure_logger()
#     logger = logging.getLogger(__name__)
#     app_cfg = AppConfig()
#     dataset_cfg_coco = get_coco_dataset_cfg_default()
#     dataset_cfg_sim = get_sim_dataset_cfg_default()
#     dataset_cfg_h36m = get_h36m_dataset_cfg_default()
#     dataset_cfg_h36m.rotation_factor = 30
#     net_cfg = PedRecNet50Config()
#     device = get_device(app_cfg.cuda.use_gpu)
#
#     cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#     # Data loading code
#     trans = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     net: PedRecNet = init_pose_model(PedRecNet(net_cfg), pose_weights, logger, device)
#     net.to(device)
#
#     val_set = PedRecDataset("data/datasets/Human3.6m/val", "h36m_val", DatasetType.VALIDATE,
#                             dataset_cfg_h36m, net_cfg.model.input_size, ImageSize(1000, 1000), trans)
#
#     print(get_heading("RT3DROM!!", 1))
#     val_set = PedRecDataset("data/datasets/ROM/", "rt_rom_01",
#                             DatasetType.TRAIN, dataset_cfg_sim, net_cfg.model.input_size, ImageSize(1920, 1080),
#                             trans)
#     val_loader = DataLoader(val_set, batch_size=64, num_workers=12)
#     print_results(validate(net, val_loader, get_outputs_loss_pedrec, get_pedrec_preds, device, net_cfg.model.input_size))
#
#     print(get_heading("MS_COCO", 1))
#     val_set = CocoDataset("data/datasets/COCO",
#                           DatasetType.VALIDATE,
#                           dataset_cfg_coco,
#                           net_cfg.model.input_size,
#                           trans)
#     val_loader = DataLoader(val_set, batch_size=64, num_workers=12)
#     print_results(
#         validate(net, val_loader, get_outputs_loss_pedrec, get_pedrec_preds, device, net_cfg.model.input_size, validate_3D=False, validate_orientation=False))
#
#     print(get_heading("Human 3.6M", 1))
#     val_set = PedRecDataset("data/datasets/Human3.6m/val", "h36m_val", DatasetType.VALIDATE,
#                             dataset_cfg_h36m, net_cfg.model.input_size, ImageSize(1000, 1000), trans)
#     val_loader = DataLoader(val_set, batch_size=64, num_workers=12)
#     joints_not_available = [SKELETON_PEDREC_JOINT.nose.value,
#                             SKELETON_PEDREC_JOINT.left_ear.value,
#                             SKELETON_PEDREC_JOINT.right_ear.value,
#                             SKELETON_PEDREC_JOINT.left_eye.value,
#                             SKELETON_PEDREC_JOINT.right_eye.value]
#     print_results(validate(net, val_loader, get_outputs_loss_pedrec, get_pedrec_preds, device, net_cfg.model.input_size, validate_orientation=False),
#                   joints_not_available=joints_not_available)
#
#     print(get_heading("RT3DValidate", 1))
#     val_set = PedRecDataset("data/datasets/RT3DValidate/", "rt_validate_3d",
#                             DatasetType.VALIDATE, dataset_cfg_sim, net_cfg.model.input_size, ImageSize(1920, 1080),
#                             trans)
#     val_loader = DataLoader(val_set, batch_size=64, num_workers=12)
#     print_results(validate(net, val_loader, get_outputs_loss_pedrec, get_pedrec_preds, device, net_cfg.model.input_size))


def get_pair_mask(joints_to_mask: List[int]):
    mask = []
    for pair in SKELETON_PEDREC_PARENT_CHILD_PAIRS:
        if pair[0] in joints_to_mask or pair[1] in joints_to_mask:
            mask.append(0)
            continue
        mask.append(1)
    return mask


def validate(net: nn.Module, val_loader: DataLoader,
             get_outputs_loss_func: Callable,
             get_preds_func: Callable,
             device: torch.device,
             model_input_size: ImageSize,
             skeleton_3d_range: int = 3000,
             validate_2D: bool = True,
             validate_3D: bool = True,
             validate_orientation: bool = True,
             validate_pose_conf: bool = True,
             validate_env_position: bool = True) -> ValidationResults:
    start = time.time()
    loss_total = 0.0
    net.eval()
    pose2d_gts = []
    pose2d_preds = []
    pose3d_gts = []
    pose3d_preds = []
    orientation_gts = []
    orientation_preds = []
    env_position_gts = []
    env_position_preds = []
    count = 0
    with torch.inference_mode():
        for test_data in tqdm(val_loader):
            # if count > 2:
            #     break
            # count += 1
            images, labels = test_data
            images = images.to(device, non_blocking=True)
            labels = move_to_device(labels, device)
            outputs, loss = get_outputs_loss_func(net, images, labels)
            preds = get_preds_func(outputs)
            loss_total += float(loss)

            # GTs
            pose2d_gt = labels["skeleton"].cpu().detach().numpy()
            centers = labels["center"].cpu().detach().numpy()
            scales = labels["scale"].cpu().detach().numpy()
            rotations = labels["rotation"].cpu().detach().numpy()
            pose2d_gt = get_total_coords(pose2d_gt, model_input_size, centers, scales, rotations)

            pose3d_gt = labels["skeleton_3d"].cpu().detach().numpy()
            pose3d_gt[:, :, :3] = pose3d_gt[:, :, :3] * skeleton_3d_range - (skeleton_3d_range / 2) # to cm

            orientation_gt = labels["orientation"].cpu().detach().numpy()

            if validate_2D:
                pose2d_pred = preds["skeleton"]
                pose2d_pred = get_total_coords(pose2d_pred, model_input_size, centers, scales, rotations)
                pose2d_gts.append(pose2d_gt)
                pose2d_preds.append(pose2d_pred)
            if validate_3D:
                pose3d_pred = preds["skeleton_3d"]
                pose3d_pred[:, :, :3] = pose3d_pred[:, :, :3] * skeleton_3d_range - (skeleton_3d_range / 2)  # to cm
                pose3d_gts.append(pose3d_gt)
                pose3d_preds.append(pose3d_pred)

            if validate_orientation:
                orientation_pred = preds["orientation"]
                orientation_gts.append(orientation_gt)
                orientation_preds.append(orientation_pred)

            if validate_env_position:
                env_position_gt = labels["env_position_2d"].cpu().detach().numpy()
                env_position_pred = preds["env_position"]

                env_position_gts.append(env_position_gt)
                env_position_preds.append(env_position_pred)

    def stack(arrays):
        return np.concatenate(arrays, 0) if len(arrays) > 0 else None
    pose2d_gts, pose2d_preds = stack(pose2d_gts), stack(pose2d_preds)
    pose3d_gts, pose3d_preds = stack(pose3d_gts), stack(pose3d_preds)
    orientation_gts, orientation_preds = stack(orientation_gts), stack(orientation_preds)
    env_position_gts, env_position_preds = stack(env_position_gts), stack(env_position_preds)
    results = ValidationResults(loss=loss_total / len(val_loader), val_duration=time.time() - start)
    # remove foot / hand end
    # pose2d_gts = pose2d_gts[:, :22, :]
    # pose2d_preds = pose2d_preds[:, :22, :]
    # pose3d_gts = pose3d_gts[:, :22, :]
    # pose3d_preds = pose3d_preds[:, :22, :]
    if validate_2D:
        results.pose2d_pck = get_2d_pose_pck_results(pose2d_gts, pose2d_preds)
    if validate_pose_conf:
        results.pose2d_conf = get_2d_pose_conf_results(pose2d_gts, pose2d_preds)
    if validate_3D:
        results.pose3d = get_3d_pose_results(pose3d_gts, pose3d_preds)
    if validate_orientation:
        results.orientation = get_orientation_results(orientation_gts, orientation_preds)
    if validate_env_position:
        results.env_position = get_env_position_results(env_position_gts, env_position_preds)
    return results


def get_2d_pose_conf_results(target: np.ndarray, pred: np.ndarray) -> Pose2DValidationConfResults:

    conf_acc, conf_per_joint_acc = get_joint_label_accuracy(target, pred)
    return Pose2DValidationConfResults(
        conf_acc=conf_acc,
        conf_per_joint_acc=conf_per_joint_acc
    )


def get_2d_pose_pck_results(target: np.ndarray, pred: np.ndarray) -> Pose2DValidationPCKResults:
    pck_05 = get_pck_results(target[:, :, 0:2], pred[:, :, 0:2], target[:, :, 3], 0.05)
    pck_05_wo_nans = pck_05[~np.isnan(pck_05)]
    pck_05_mean = np.sum(pck_05_wo_nans) / len(pck_05_wo_nans)
    pck_2 = get_pck_results(target[:, :, 0:2], pred[:, :, 0:2], target[:, :, 3], 0.2)
    pck_2_wo_nans = pck_2[~np.isnan(pck_2)]
    pck_2_mean = np.sum(pck_2_wo_nans) / len(pck_2_wo_nans)
    return Pose2DValidationPCKResults(
        pck_05=pck_05,
        pck_2=pck_2,
        pck_05_mean=pck_05_mean,
        pck_2_mean=pck_2_mean
    )


# def get_2d_pose_results(target: np.ndarray, pred: np.ndarray) -> Pose2DValidationResults:
#     pck_results = get_2d_pose_pck_results(target, pred)
#     conf_results = get_2d_pose_conf_results(target, pred)
#     return Pose2DValidationResults(
#         pck=pck_results,
#         conf=conf_results
#     )


def get_3d_pose_results(target: np.ndarray, pred: np.ndarray) -> Pose3DValidationResults:
    mpjpe = get_mpjpe(target, pred)
    mean_joint_depth_distances = get_dim_distances(target, pred, 2)
    mean_joint_x_distances = get_dim_distances(target, pred, 0)
    mean_joint_y_distances = get_dim_distances(target, pred, 1)
    pct_correct_depth_per_pair, pct_correct_depth_mean = get_relative_correct_depth(target, pred)
    pct_correct_joint_positions_per_pair, pct_correct_joint_positions_mean = get_relative_correct_joint_positions(target, pred)
    mpjpe_wo_nan = mpjpe[~np.isnan(mpjpe)]
    mpjpe_mean = np.sum(mpjpe_wo_nan) / mpjpe_wo_nan.shape[0]
    return Pose3DValidationResults(
        mpjpe=mpjpe,
        mpjpe_mean=mpjpe_mean,
        mean_joint_depth_distances=mean_joint_depth_distances,
        mean_joint_x_distances=mean_joint_x_distances,
        mean_joint_y_distances=mean_joint_y_distances,
        pct_correct_depth_per_pair=pct_correct_depth_per_pair,
        pct_correct_depth_mean=pct_correct_depth_mean,
        pct_correct_joint_position_per_pair=pct_correct_joint_positions_per_pair,
        pct_correct_joint_position_mean=pct_correct_joint_positions_mean,
        num_examples=target.shape[0]
    )

def get_env_position_results(env_position_gts, env_position_preds) -> EnvPositionValidationResults:
    env_position_gts[:, 0] *= 10000
    env_position_gts[:, 1] *= 30000
    env_position_preds[:, 0] *= 10000
    env_position_preds[:, 1] *= 30000
    env_position_distance = np.linalg.norm(env_position_gts[:, :2] - env_position_preds[:, :2])
    num_visibles = np.sum(env_position_gts[:, 2], axis=0)
    return EnvPositionValidationResults(
        env_position_distance / num_visibles
    )


def get_orientation_results(orientation_gts_in, orientation_preds_in) -> OrientationValidationResults:
    """
    input: batch, body_part, [theta, phi], [x, y]
    """
    # orientation_preds_cartesian = (orientation_preds_cartesian * 2) - 1
    # phi = np.arctan2(orientation_preds_cartesian[:, 1::2, 1], orientation_preds_cartesian[:, 1::2, 0])
    # phi[phi < 0] += 2 * math.pi
    # phi = np.expand_dims(phi, axis=2)
    #
    # theta = np.arctan2(orientation_preds_cartesian[:, ::2, 1], orientation_preds_cartesian[:, ::2, 0])
    # theta[theta < 0] += 2 * math.pi
    # theta = np.expand_dims(theta, axis=2)
    # orientation_preds = np.concatenate([theta, phi], axis=2)
    orientation_gts = orientation_gts_in.copy()
    orientation_preds_cartesian = orientation_preds_in.copy()
    orientation_gts[:, :, 0] *= math.pi
    orientation_gts[:, :, 1] *= 2 * math.pi

    orientation_preds_cartesian[:, :, 0] *= math.pi
    orientation_preds_cartesian[:, :, 1] *= 2 * math.pi
    orientation_preds = orientation_preds_cartesian
    angle_error_phi_body, angle_error_theta_body, spherical_distance_body = get_angular_error_statistics(
        orientation_gts[:, 0, :],
        orientation_preds[:, 0, :])
    angle_error_phi_head, angle_error_theta_head, spherical_distance_head = get_angular_error_statistics(
        orientation_gts[:, 1, :],
        orientation_preds[:, 1, :])
    return OrientationValidationResults(
        body=OrientationValidationResult(
            angle_error_phi=angle_error_phi_body,
            angle_error_theta=angle_error_theta_body,
            spherical_distance=spherical_distance_body
        ),
        head=OrientationValidationResult(
            angle_error_phi=angle_error_phi_head,
            angle_error_theta=angle_error_theta_head,
            spherical_distance=spherical_distance_head
        )
    )


def print_results_pose_2d_pck(pck_05_mean: float, pck_2_mean: float):
    print(get_heading("PCK (Torso)", 3))
    print(f"@0.05: {pck_05_mean:.2f}")
    print(f"@0.2: {pck_2_mean:.2f}")


def print_results_pose_2d_joint_accuracy(conf_acc: float):
    print(get_heading("Joint Confidence Accuracy (%)", 3))
    print(f"Joint Confidence Accuracy: {conf_acc * 100:.2f}")


def print_results_pose_3d(mpjpe: np.ndarray, joints_not_available: List[SKELETON_PEDREC_JOINT]):
    print(get_heading("MPJPE (cm)", 3))
    dists = []
    for joint in SKELETON_PEDREC_JOINT:
        if joint.value in joints_not_available:
            continue
        dist = mpjpe[joint.value]
        dists.append(np.abs(dist))
        print(f"{joint.name}: {dist:.2f}")
    dists = np.array(dists)
    print("---")
    print(f"MPJPE (mean): {np.sum(dists) / dists.shape[0]:.2f}")


def print_results_depth_distance(mean_joint_depth_distances: np.ndarray,
                                 joints_not_available: List[SKELETON_PEDREC_JOINT]):
    print(get_heading("Depth distances (cm)", 3))
    dists = []
    for joint in SKELETON_PEDREC_JOINT:
        if joint.value in joints_not_available:
            continue
        dist = mean_joint_depth_distances[joint.value]
        dists.append(np.abs(dist))
        print(f"{joint.name}: {dist:.2f}")
    dists = np.array(dists)
    print("---")
    print(f"Mean joint distance over all: {np.sum(dists) / dists.shape[0]:.2f}")


def print_results_relative_correct_depth(corrects_per_pair: np.ndarray, num_examples: int,
                                         joints_not_available: List[SKELETON_PEDREC_JOINT]):
    print(get_heading("Relative Correct Depth", 3))
    pair_mask = get_pair_mask(joints_not_available)
    corrects_pcts = []
    for pair_idx, pair in enumerate(SKELETON_PEDREC_PARENT_CHILD_PAIRS):
        if pair_mask[pair_idx] == 0:
            continue
        correct_pct = corrects_per_pair[pair_idx] / num_examples
        corrects_pcts.append(correct_pct)
        print(f"{SKELETON_PEDREC_JOINT(pair[0]).name}-{SKELETON_PEDREC_JOINT(pair[1]).name}: {correct_pct * 100:.2f}%")
    corrects_pcts = np.array(corrects_pcts)
    print("---")
    print(f"Relative correct depth (all joints): {(np.sum(corrects_pcts) / corrects_pcts.shape[0]) * 100:.2f}%")


def print_results(resultsx: ValidationResults, joints_not_available: List[SKELETON_PEDREC_JOINT] = []):
    if resultsx.pose2d_pck is not None:
        print(get_heading("2D Pose Estimation", 2))
        print_results_pose_2d_pck(resultsx.pose2d_pck.pck_05_mean, resultsx.pose2d_pck.pck_2_mean)
    if resultsx.pose2d_conf is not None:
        print(get_heading("Pose Joint Confidence", 2))
        print_results_pose_2d_joint_accuracy(resultsx.pose2d_conf.conf_acc)
    if resultsx.pose3d is not None:
        print(get_heading("3D Pose Estimation", 2))
        print_results_pose_3d(resultsx.pose3d.mpjpe, joints_not_available)
        print_results_depth_distance(resultsx.pose3d.mean_joint_depth_distances, joints_not_available)
        print_results_relative_correct_depth(resultsx.pose3d.pct_correct_depth_per_pair, resultsx.pose3d.num_examples,
                                             joints_not_available)
    if resultsx.orientation is not None:
        print(get_heading("Orientation", 2))
        print(
            f"Body: E-Phi (φ): {resultsx.orientation.body.angle_error_phi}, E-Theta (θ): {resultsx.orientation.body.angle_error_theta} | Spherical distance: {resultsx.orientation.body.spherical_distance}")
        print(
            f"Head: E-Phi (φ): {resultsx.orientation.head.angle_error_phi}, E-Theta (θ): {resultsx.orientation.head.angle_error_theta} | Spherical distance: {resultsx.orientation.head.spherical_distance}")
