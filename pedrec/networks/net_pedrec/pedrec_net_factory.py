import logging

import torch

from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file

logger = logging.getLogger(__name__)


def load_pedrec_net(weights_path: str, device: torch.device = torch.device("cpu")) -> PedRecNet:
    """
    Loads a PedRecNet for inference from plain network weights (``*_net.pth``) or a training checkpoint (MTL wrapper,
    keys prefixed with ``model.``).
    """
    net = PedRecNet(PedRecNet50Config())
    state = load_state_dict_file(weights_path)
    if any(key.startswith("model.") for key in state):
        state = {key[len("model."):]: value for key, value in state.items() if key.startswith("model.")}
    net.load_state_dict(state)
    logger.info(f"Loaded PedRecNet (weights: {weights_path})")
    return net.to(device).eval()
