from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass()
class Pose3DValidationResults(object):
    mpjpe: np.ndarray
    mpjpe_mean: float
    mean_joint_depth_distances: np.ndarray
    mean_joint_x_distances: np.ndarray
    mean_joint_y_distances: np.ndarray
    pct_correct_depth_per_pair: np.ndarray
    pct_correct_depth_mean: float
    pct_correct_joint_position_per_pair: np.ndarray
    pct_correct_joint_position_mean: np.ndarray
    num_examples: int
    # J x 3 median signed error (pred - GT, mm) in body coordinates (out / up / forward), see get_joint_bias_body
    joint_bias_body: Optional[np.ndarray] = None
