# Copied from https://github.com/isaac-sim/IsaacLabEvalTasks/blob/main/scripts/robot_joints.py

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict


@dataclass
class JointsAbsPosition:
    joints_pos: torch.Tensor
    """Joint positions in radians"""

    joints_order_config: Dict[str, int]
    """Joints order configuration"""

    device: torch.device
    """Device to store the tensor on"""

    @staticmethod
    def zero(joint_order_config: Dict[str, int], device: torch.device):
        return JointsAbsPosition(
            joints_pos=torch.zeros((len(joint_order_config)), device=device),
            joints_order_config=joint_order_config,
            device=device,
        )

    def to_array(self) -> torch.Tensor:
        return self.joints_pos.cpu().numpy()

    @staticmethod
    def from_array(array: np.ndarray, joint_order_config: Dict[str, int], device: torch.device) -> "JointsAbsPosition":
        return JointsAbsPosition(
            joints_pos=torch.from_numpy(array).to(device), joints_order_config=joint_order_config, device=device
        )

    def set_joints_pos(self, joints_pos: torch.Tensor):
        self.joints_pos = joints_pos.to(self.device)

    def get_joints_pos(self, device: torch.device = None) -> torch.Tensor:
        if device is None:
            return self.joints_pos
        else:
            return self.joints_pos.to(device)