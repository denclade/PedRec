"""
Checkpoint loading compatible with current PyTorch versions.

Since PyTorch 2.6 ``torch.load`` defaults to ``weights_only=True``. All PedRec checkpoints (PedRecNet v1 / v2, ST-GCN,
the MTL training checkpoints and the pose-resnet ``.pth.tar``) are plain state dicts and load fine in this mode.
Older ``.pth.tar`` files which wrap the state dict in ``{"state_dict": ...}`` are unwrapped automatically.
"""
from collections import OrderedDict
from typing import Dict, Union

import torch


def load_state_dict_file(path: str, map_location: Union[str, torch.device] = "cpu",
                         trusted: bool = False) -> Dict[str, torch.Tensor]:
    """
    Loads a state dict from ``path``.

    :param path: checkpoint file
    :param map_location: target device of the tensors (default cpu, move the model afterwards)
    :param trusted: set to True to allow arbitrary pickled objects (only for files from trusted sources which can not
        be loaded with ``weights_only=True``)
    """
    state = torch.load(path, map_location=map_location, weights_only=not trusted)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and len(state) > 0 and all(key.startswith("module.") for key in state.keys()):
        # saved from a DataParallel / DDP wrapped model
        state = OrderedDict((key[len("module."):], value) for key, value in state.items())
    return state
