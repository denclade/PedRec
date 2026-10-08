"""
Central location for the default data / model layout used by the command line entry points.

Every path is relative to the data root which can be configured via the ``PEDREC_DATA_DIR``
environment variable (default: ``data`` relative to the current working directory).
"""
import os


DATA_DIR_ENV = "PEDREC_DATA_DIR"


def get_data_root(data_root: str = None) -> str:
    """
    Returns the data root directory. Precedence: explicit argument > PEDREC_DATA_DIR env > "data".
    """
    if data_root:
        return data_root
    return os.environ.get(DATA_DIR_ENV, "data")


def data_path(*parts: str, data_root: str = None) -> str:
    return os.path.join(get_data_root(data_root), *parts)


# Weights (see README): PedRecNet v2, the temporal 3D lifter and the ST-GCN action recognition are trained with this
# code base, the detector (RT-DETRv2) is loaded from the Hugging Face hub / cache.
PEDREC_NET_WEIGHTS = os.path.join("models", "pedrec", "experiment_pedrec_v2_p2d3d_c_o_0_net.pth")
LIFTER_WEIGHTS = os.path.join("models", "pedrec", "pedrec_v2_lifter.pth")
EHPI3D_WEIGHTS = os.path.join("models", "ehpi3d", "ehpi_stgcn_sim_c01_actionrec_gt_pred_64frames.pth")
RTDETR_MODEL = "PekingU/rtdetr_v2_r18vd"

# Demo data
DEMO_VIDEO = os.path.join("demo", "multi_person_crossing_street.mp4")

# Experiment output (checkpoints + protocols)
PEDREC_CHECKPOINT_DIR = os.path.join("models", "pedrec", "single_results")
EHPI3D_CHECKPOINT_DIR = os.path.join("models", "ehpi3d")


def pedrec_net_weights(data_root: str = None) -> str:
    return data_path(PEDREC_NET_WEIGHTS, data_root=data_root)


def lifter_weights(data_root: str = None) -> str:
    return data_path(LIFTER_WEIGHTS, data_root=data_root)


def ehpi3d_weights(data_root: str = None) -> str:
    return data_path(EHPI3D_WEIGHTS, data_root=data_root)


def demo_video(data_root: str = None) -> str:
    return data_path(DEMO_VIDEO, data_root=data_root)
