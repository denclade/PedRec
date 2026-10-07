"""
Construction and loading of PedRecNets of all architecture variants.

Every checkpoint written by the training of a non default architecture gets a sidecar file ``<checkpoint>.arch.json``
with its ``PedRecArchConfig``. Checkpoints without a sidecar (all published weights) are ``v1`` networks.
"""
import dataclasses
import json
import logging
import os
from typing import Optional

import torch

from pedrec.configs.pedrec_net_config import PedRecArchConfig, PedRecNet50Config, PedRecNetConfig
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file

logger = logging.getLogger(__name__)


def arch_sidecar_path(checkpoint_path: str) -> str:
    return checkpoint_path + ".arch.json"


def save_arch(checkpoint_path: str, arch: PedRecArchConfig):
    if arch.name == "v1" and arch == PedRecArchConfig():
        return  # default architecture, no sidecar needed
    with open(arch_sidecar_path(checkpoint_path), "w") as f:
        json.dump(arch.to_dict(), f, indent=2)


def load_arch(checkpoint_path: str) -> PedRecArchConfig:
    path = arch_sidecar_path(checkpoint_path)
    if not os.path.isfile(path):
        return PedRecArchConfig()
    with open(path) as f:
        return PedRecArchConfig.from_dict(json.load(f))


def copy_arch(source_checkpoint: str, target_checkpoint: str):
    save_arch(target_checkpoint, load_arch(source_checkpoint))


def pedrec_config(arch: Optional[PedRecArchConfig] = None) -> PedRecNetConfig:
    cfg = PedRecNet50Config()
    if arch is not None:
        cfg = dataclasses.replace(cfg, arch=arch)
    return cfg


def load_pedrec_net(weights_path: str, device: torch.device = torch.device("cpu"),
                    arch: Optional[PedRecArchConfig] = None) -> PedRecNet:
    """
    Loads a PedRecNet for inference from plain network weights (``*_net.pth``) or a training checkpoint (MTL wrapper,
    keys prefixed with ``model.``). The architecture comes from ``arch`` or the sidecar of the checkpoint.
    """
    arch = arch or load_arch(weights_path)
    cfg = pedrec_config(dataclasses.replace(arch, backbone_pretrained=False))
    net = PedRecNet(cfg)
    state = load_state_dict_file(weights_path)
    if any(key.startswith("model.") for key in state):
        state = {key[len("model."):]: value for key, value in state.items() if key.startswith("model.")}
    net.load_state_dict(state)
    logger.info(f"Loaded PedRecNet ({arch.name}, weights: {weights_path})")
    return net.to(device).eval()
