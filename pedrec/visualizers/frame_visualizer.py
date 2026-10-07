"""
Headless (OpenCV only) rendering of pipeline results into a frame.
"""
import math
from typing import List

import cv2
import numpy as np

from pedrec.models.human import Human
from pedrec.visualizers.bb_visualizer import draw_bb, draw_bbs
from pedrec.visualizers.skeleton_visualizer import draw_skeleton


def draw_frame_result(img: np.ndarray, humans: List[Human], objects: List[np.ndarray], fps: int = None,
                      draw_objects: bool = True, min_joint_score: float = 0.5) -> np.ndarray:
    """
    Draws object bbs, human bbs + ids, skeletons, body orientation and actions into ``img`` (RGB, in place).
    """
    if draw_objects and len(objects) > 0:
        draw_bbs(img, np.array(objects))
    for human in humans:
        title = f"id {human.uid}" if human.uid != -1 else "human"
        if human.orientation is not None:
            body_phi_deg = math.degrees(human.orientation[0, 1])
            title += f" | body {body_phi_deg:.0f} deg"
        if human.actions:
            title += " | " + ", ".join(action.name for action in human.actions)
        draw_bb(img, human.bb, title=title)
        if human.skeleton_2d is not None and np.any(human.skeleton_2d[:, 2] > 0):
            draw_skeleton(img, human.skeleton_2d, min_joint_score=min_joint_score)
    if fps is not None:
        cv2.putText(img, f"{fps} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img
