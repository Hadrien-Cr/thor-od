from dataclasses import dataclass
from typing import Optional
from enum import Enum
import numpy as np

from habitat_sim.agent.agent import AgentState

class DiscreteNavigationAction(Enum):
    """Discrete navigation controls."""

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


@dataclass
class Observations:
    # Pose
    gps: np.ndarray  # (x, y) where positive x is forward, positive y is translation to left in meters
    compass: np.ndarray  # positive theta is rotation to left in radians - consistent with robot

    # Camera
    rgb: np.ndarray  # (camera_height, camera_width, 3) in [0, 255]
    depth: np.ndarray  # (camera_height, camera_width) in meters
    semantic: np.ndarray # (camera_height, camera_width) in [0, num_sem_categories - 1]
    camera_pose: np.ndarray

    task_observations: dict

@dataclass
class Labels:
    instances: list[dict]

@dataclass
class Pose:
    position: np.ndarray
    orientation: np.ndarray