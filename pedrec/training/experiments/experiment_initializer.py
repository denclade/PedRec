import logging

from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file

logger = logging.getLogger(__name__)


def initialize_weights_with_same_name_and_shape(net, weights_path: str):
    """
    Copies all weights from ``weights_path`` whose name and shape match a parameter / buffer of ``net``. Everything
    else keeps its current initialization, mismatches are logged. This is used to initialize a training stage from the
    checkpoint of its predecessor stage.
    """
    weights = load_state_dict_file(weights_path)
    net_weights = net.state_dict()
    for name, param in weights.items():
        target = name if name in net_weights else None
        if target is None:
            logger.info(f"No weight found for '{name}'")
        elif net_weights[target].shape != param.shape:
            logger.warning(f"Shape mismatch for '{name}': checkpoint {tuple(param.shape)} <-> "
                           f"net {tuple(net_weights[target].shape)}, keeping the initialization")
        else:
            net_weights[target] = param
    net.load_state_dict(net_weights)

