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


# Pretrained weights (see README, "Pretrained models")
YOLO_V4_WEIGHTS = os.path.join("models", "yolo_v4", "yolov4.pth")
PEDREC_NET_WEIGHTS = os.path.join("models", "pedrec", "experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_net.pth")
EHPI3D_WEIGHTS = os.path.join("models", "ehpi3d", "ehpi_3d_sim_c01_actionrec_gt_pred_64frames.pth")
POSE_RESNET_WEIGHTS = os.path.join("models", "human_pose_baseline", "pose_resnet_50_256x192.pth.tar")

# Demo data
DEMO_VIDEO = os.path.join("demo", "multi_person_crossing_street.mp4")

# Experiment output (checkpoints + protocols)
PEDREC_CHECKPOINT_DIR = os.path.join("models", "pedrec", "single_results")
EHPI3D_CHECKPOINT_DIR = os.path.join("models", "ehpi3d")


def yolo_v4_weights(data_root: str = None) -> str:
    return data_path(YOLO_V4_WEIGHTS, data_root=data_root)


def pedrec_net_weights(data_root: str = None) -> str:
    return data_path(PEDREC_NET_WEIGHTS, data_root=data_root)


def ehpi3d_weights(data_root: str = None) -> str:
    return data_path(EHPI3D_WEIGHTS, data_root=data_root)


def demo_video(data_root: str = None) -> str:
    return data_path(DEMO_VIDEO, data_root=data_root)
