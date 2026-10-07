import logging

from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file

logger = logging.getLogger(__name__)


def initialize_weights_with_same_name_and_shape(net, weights_path: str, prefix: str = None):
    """
    Copies all weights from ``weights_path`` whose name (optionally after removing ``prefix``) and shape match a
    parameter / buffer of ``net``. Everything else keeps its current initialization, mismatches are logged.
    This is used to initialize a training stage from the checkpoint of its predecessor stage.
    """
    weights = load_state_dict_file(weights_path)
    net_weights = net.state_dict()
    for name, param in weights.items():
        fixed_name = name.replace("conv_transpose_2d", "conv_transpose_shared")
        if prefix is not None:
            fixed_name = fixed_name.replace(prefix, "")
        target = name if name in net_weights else fixed_name if fixed_name in net_weights else None
        if target is None:
            logger.info(f"No weight found for '{name}'")
        elif net_weights[target].shape != param.shape:
            logger.warning(f"Shape mismatch for '{name}': checkpoint {tuple(param.shape)} <-> "
                           f"net {tuple(net_weights[target].shape)}, keeping the initialization")
        else:
            net_weights[target] = param
    net.load_state_dict(net_weights)


def initialize_pose_resnet(net, pose_resnet_weights_path: str):
    """
    Initializes a PedRecNet (wrapped in the MTL wrapper, thus the ``model.`` prefix) from the Microsoft
    "Simple Baselines" pose-resnet weights: backbone + shared deconvs + 2D head.
    """
    pose_resnet_state_dict = load_state_dict_file(pose_resnet_weights_path)
    net_weights = net.state_dict()
    for name, param in pose_resnet_state_dict.items():
        if name.startswith("final"):
            net_name = name.replace("final_layer.", "model.head_pose_2d.pose_heatmap_layer.")
        elif name.startswith("deconv_layers.6") or name.startswith("deconv_layers.7"):
            net_name = f"model.head_pose_2d.{name.replace('deconv_layers', 'deconv_head').replace('6', '0').replace('7', '1')}"
        elif name.startswith("deconv"):
            net_name = f"model.conv_transpose_shared.{name}"
        else:
            net_name = f"model.feature_extractor.{name}"
        if net_name in net_weights:
            if net_weights[net_name].shape != param.shape and len(param.shape) == 1:
                net_weights[net_name][:17] = param
            elif net_weights[net_name].shape != param.shape and len(param.shape) == 4:
                net_weights[net_name][:17, :, :, :] = param
            elif net_weights[net_name].shape != param.shape:
                logger.warning(f"Shape mismatch in {net_name}: {net_weights[net_name].shape} <-> {param.shape}")
            else:
                net_weights[net_name] = param
        else:
            logger.info(f"Skipped: {name}, tried: {net_name}")
    net.load_state_dict(net_weights)
