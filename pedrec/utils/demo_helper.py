from logging import Logger

import torch

from pedrec.configs.yolo_v4_config import YoloV4Config
from pedrec.networks.net_yolo_v4.yolov4 import YoloV4
from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file


def _num_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_detector(cfg: YoloV4Config, weights_path: str, logger: Logger, device: torch.device):
    detector = YoloV4(cfg, inference=True)
    detector.load_state_dict(load_state_dict_file(weights_path))
    logger.info(f"Loaded YoloV4 (weights: {weights_path}). Num of trainable params: {_num_trainable_params(detector)}")
    detector = detector.to(device)
    detector.eval()
    return detector


def init_pose_model(pose_model: torch.nn.Module,
                    weights_path: str,
                    logger: Logger,
                    device: torch.device):
    pose_model.load_state_dict(load_state_dict_file(weights_path))
    logger.info(f"Loaded {pose_model.__class__.__name__} (weights: {weights_path}). "
                f"Num of trainable params: {_num_trainable_params(pose_model)}")
    pose_model = pose_model.to(device)
    pose_model.eval()
    return pose_model
