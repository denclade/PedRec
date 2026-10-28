from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass()
class Pose2DValidationPCKResults(object):
    pck_05: np.ndarray
    pck_2: np.ndarray
    pck_05_mean: float
    pck_2_mean: float
    # J x 2 mean signed error (pred - GT) along the body axes in the image (out / up), in % of the torso length
    joint_bias_body: Optional[np.ndarray] = None
