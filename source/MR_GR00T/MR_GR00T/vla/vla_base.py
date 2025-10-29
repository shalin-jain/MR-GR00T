# Copied from https://github.com/isaac-sim/IsaacLabEvalTasks/blob/main/scripts/policies/policy_base.py

from abc import ABC, abstractmethod

from MR_GR00T.utils.robot_joints import JointsAbsPosition

from isaaclab.sensors import Camera


class VLABase(ABC):
    """A base class for all policies."""

    @abstractmethod
    def step(self, current_state: JointsAbsPosition, camera: Camera) -> JointsAbsPosition:
        """Called every simulation step to update policy's internal state."""
        pass

    @abstractmethod
    def get_new_goal(self, current_state: JointsAbsPosition, camera: Camera) -> JointsAbsPosition:
        """Generates a goal given the current state and camera observations."""
        pass

    @abstractmethod
    def reset(self):
        """Resets the policy's internal state."""
        pass