import logging
import random
from typing import List, Iterator, Collection, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pedrec.models.constants.nn_constants import batch_norm_types, no_wd_types, bias_types
from pedrec.utils.list_helper import get_without_duplicates


def get_device(use_gpu: bool = True):
    logger = logging.getLogger(__name__)
    if not use_gpu:
        logger.info('Working on CPU!')
        return torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info('Working on GPU: {}!'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        logger.info('No GPU with configured CUDA found. Working on CPU!')

    return device


def set_fixed_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def affine_transform_coords_2d(pose_coords_2d, trans_inv, device: torch.device):
    new_pts = torch.ones((pose_coords_2d.shape[0], 3), dtype=torch.float, device=device)
    new_pts[:, :2] = pose_coords_2d[:, :2]
    new_pts = torch.matmul(trans_inv, new_pts.T).T
    return new_pts


class ParameterModule(nn.Module):
    "Register a lone parameter `p` in a module."

    def __init__(self, p: nn.Parameter):
        super(ParameterModule, self).__init__()
        self.val = p

    def forward(self, x):
        return x


def get_children_with_param_modules(children: List[nn.Module], parameters: Iterator[nn.Parameter]):
    child_params = []
    for child in children:
        for param in child.parameters():
            child_params.append(id(param))

    for p in parameters:
        if id(p) not in child_params:
            children.append(ParameterModule(p))
    return children


def get_flattened_layers(m):
    """
    Flattens the model and returns a (ordered) list of the containing layers and parameters. Based on the
    fastai implementation
    """
    layers = []
    children = list(m.children())
    if len(children) > 0:
        children = get_children_with_param_modules(children, m.parameters())
        for child in children:
            layers += get_flattened_layers(child)
    else:
        layers.append(m)
    return layers


def split_model_idx(model: nn.Module, idxs: Collection[int]):
    "Split `model` according to the indexes in `idxs`."
    layers = get_flattened_layers(model)

    # Add 0 and max idx if not exist
    if idxs[0] != 0: idxs = [0] + idxs
    if idxs[-1] != len(layers): idxs.append(len(layers))

    # split layer based on this idxs
    return [nn.Sequential(*layers[i:j]) for i, j in zip(idxs[:-1], idxs[1:])]


def freeze_layer_groups(layer_groups: List[nn.Sequential]):
    for layer_group in layer_groups:
        for layer in layer_group:
            freeze_layer(layer)


def unfreeze_layer_groups(layer_groups: List[nn.Sequential]):
    for layer_group in layer_groups:
        for layer in layer_group:
            unfreeze_layer(layer)


def freeze_layers(layers: Iterator[nn.Module], recursive: bool = True, hard: bool = False):
    for layer in layers:
        freeze_layer(layer, hard)

        if not recursive:
            continue
        children = layer.children()
        if children is not None:
            freeze_layers(children, hard)


def unfreeze_layers(layers: Iterator[nn.Module], recursive: bool = True):
    for layer in layers:
        unfreeze_layer(layer)

        if not recursive:
            continue
        children = layer.children()
        if children is not None:
            unfreeze_layers(children)


def freeze_layer(layer: nn.Module, hard: bool = False):
    """
    Hard = freeze everything, false = freeze everything aside of batch norm
    """
    if hard or not isinstance(layer, batch_norm_types):
        for param in layer.parameters():
            param.requires_grad = False


def unfreeze_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = True


def init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_trainable_params(m: nn.Module, recurse: bool = True) -> Iterator[nn.Parameter]:
    "Return list of trainable params in `m`."
    return filter(lambda p: p.requires_grad, m.parameters(recurse=recurse))

def __add_param(l1, l2, c, trainable_params):
    if isinstance(c, no_wd_types):
        l2 += list(trainable_params)
    elif isinstance(c, bias_types):
        bias = c.bias if hasattr(c, 'bias') else None
        l1 += [p for p in trainable_params if not (p is bias)]
        if bias is not None:
            l2.append(bias)
    else:
        l1 += list(trainable_params)
def split_no_wd_params(layer_groups: List[nn.Sequential]) -> List[Tuple[bool, List[nn.Parameter]]]:
    "Separate the parameters in `layer_groups` between `no_wd_types` and  bias (`bias_types`) from the rest."
    split_params = []
    for l in layer_groups:
        l1, l2 = [], []
        trainable_params = get_trainable_params(l, recurse=False)
        __add_param(l1, l2, l, trainable_params)
        for c in l.children():
            trainable_params = get_trainable_params(c)
            __add_param(l1, l2, c, trainable_params)

        # Since we scan the children separately, we might get duplicates (tied weights). We need to preserve the order
        # for the optimizer load of state_dict
        l1, l2 = get_without_duplicates(l1), get_without_duplicates(l2)
        split_params += [(True, l1), (False, l2)]
    return split_params


def get_mean_std(loader: DataLoader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for sample in loader:
        data, _, _ = sample
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    # fix 0 division
    mean[mean == 0] = 0.00001
    std[std == 0] = 0.00001
    return mean, std


def move_to_device(obj, device, non_blocking=True):
    if hasattr(obj, "to"):
        return obj.to(device, non_blocking=non_blocking)
    elif isinstance(obj, tuple):
        return tuple(move_to_device(o, device, non_blocking) for o in obj)
    elif isinstance(obj, list):
        return [move_to_device(o, device, non_blocking) for o in obj]
    elif isinstance(obj, dict):
        return {k: move_to_device(o, device, non_blocking) for k, o in obj.items()}
    else:
        return obj


def create_meshgrid(
        x: torch.Tensor,
        normalized_coordinates: Optional[bool]) -> torch.Tensor:
    assert len(x.shape) == 4, x.shape
    _, _, height, width = x.shape
    _device, _dtype = x.device, x.dtype
    if normalized_coordinates:
        xs = torch.linspace(0.0, 1.0, width, device=_device, dtype=_dtype)
        ys = torch.linspace(0.0, 1.0, height, device=_device, dtype=_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
        ys = torch.linspace(0, height - 1, height, device=_device, dtype=_dtype)
    return torch.meshgrid(ys, xs, indexing='ij')


def create_linspace(
        x: torch.Tensor,
        normalized_coordinates: Optional[bool]) -> torch.Tensor:
    if len(x.shape) == 2:
        _, width = x.shape
    elif len(x.shape) == 3:
        _, _, width = x.shape
    else:
        raise(x.shape)
    _device, _dtype = x.device, x.dtype
    if normalized_coordinates:
        xs = torch.linspace(0.0, 1.0, width, device=_device, dtype=_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
    return xs