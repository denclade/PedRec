from dataclasses import dataclass
from typing import List

import numpy as np

from pedrec.models.constants.action_mappings import ACTION
from pedrec.utils.skeleton_helper import get_skeleton_mean_score


@dataclass(unsafe_hash=True)
class Human:
    bb: List[float]
    skeleton_2d: np.ndarray
    skeleton_3d: np.ndarray
    orientation: np.ndarray
    uid: int = -1
    env_position: np.ndarray = None
    ehpi: np.ndarray = None
    actions: List[ACTION] = None
    action_probabilities: np.ndarray = None
    lifter_features: np.ndarray = None  # per frame input of the temporal 3D lifter (J x 6)
    __score: float = None

    @property
    def score(self):
        if self.__score is None:
            self.__score = get_skeleton_mean_score(self.skeleton_2d)
        return self.__score
