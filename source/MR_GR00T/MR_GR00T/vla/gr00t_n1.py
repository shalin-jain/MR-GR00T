# Copied and modified from https://github.com/isaac-sim/IsaacLabEvalTasks/blob/main/scripts/policies/gr00t_n1_policy.py

import os
from typing import Any, Dict, Tuple

from isaaclab.sensors import Camera
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from MR_GR00T.utils.io_utils import load_gr1_joints_config
from MR_GR00T.utils.image_conversion import resize_frames_with_padding
from MR_GR00T.utils.joints_conversion import remap_policy_joints_to_sim_joints, remap_sim_joints_to_policy_joints
from MR_GR00T.utils.robot_joints import JointsAbsPosition
from .config.args import Gr00tN1ClosedLoopArguments
from .vla_base import VLABase

class Gr00tN1(VLABase):
    def __init__(self, args: Gr00tN1ClosedLoopArguments):
        self.args = args
        self.model = self._load_model()
        self._load_model_joints_config()
        self._load_sim_joints_config()

    def _load_model_joints_config(self):
        """
        Load the joint configuration of the Fourier GR1 as GR00T N1 expects it.
        """
        self.gr00t_joints_config = load_gr1_joints_config(self.args.gr00t_joints_config_path)

    def _load_sim_joints_config(self):
        """
        Load the joint configuration of the Fourier GR1 as simulation expects it.
        """
        self.sim_gr1_state_joint_config = load_gr1_joints_config(self.args.state_joints_config_path)
        self.sim_gr1_action_joint_config = load_gr1_joints_config(self.args.action_joints_config_path)

    def _load_model(self):
        """
        Load the model from the model path.
        """
        assert os.path.exists(self.args.model_path), f"Model path {self.args.model_path} does not exist"

        # Lookup the configured data preprocessors
        self.data_config = DATA_CONFIG_MAP[self.args.data_config]
        modality_config = self.data_config.modality_config()
        modality_transform = self.data_config.transform()

        # Load the model
        return Gr00tPolicy(
            model_path=self.args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=self.args.embodiment_tag,
            denoising_steps=self.args.denoising_steps,
            device=self.args.policy_device,
        )

    def step(self, current_state: JointsAbsPosition, camera: Camera) -> JointsAbsPosition:
        """Call every simulation step to update policy's internal state."""
        pass

    def get_new_goal(
        self, current_state: JointsAbsPosition, ego_camera: Camera, language_instruction: str
    ) -> Tuple[JointsAbsPosition, Dict[str, Any]]:
        """
        Run policy prediction on the given observations. Produce a new action goal for the robot.

        Args:
            current_state: robot proprioceptive state observation
            ego_camera: camera sensor observation
            language_instruction: language instruction for the task

        Returns:
            A tuple containing the inferred action for robot joints and the backbone embedding.
        """
        rgb = ego_camera.data.output["rgb"]
        # Apply preprocessing to rgb
        rgb = resize_frames_with_padding(
            rgb, target_image_size=self.args.target_image_size, bgr_conversion=False, pad_img=True
        )
        # Retrieve joint positions as proprioceptive states and remap to policy joint orders
        robot_state_policy = remap_sim_joints_to_policy_joints(current_state, self.gr00t_joints_config)

        # Pack inputs to dictionary and run the inference
        observations = {
            "annotation.human.action.task_description": [language_instruction],  # list of strings
            "video.ego_view": rgb.reshape(-1, 1, 256, 256, 3),  # numpy array of shape (N, 1, 256, 256, 3)
            "state.left_arm": robot_state_policy["left_arm"].reshape(-1, 1, 7),  # numpy array of shape (N, 1, 7)
            "state.right_arm": robot_state_policy["right_arm"].reshape(-1, 1, 7),  # numpy array of shape (N, 1, 7)
            "state.left_hand": robot_state_policy["left_hand"].reshape(-1, 1, 6),  # numpy array of shape (N, 1, 6)
            "state.right_hand": robot_state_policy["right_hand"].reshape(-1, 1, 6),  # numpy array of shape (N, 1, 6)
        }
        robot_action_policy, backbone_embedding = self.model.get_action(observations)

        robot_action_sim = remap_policy_joints_to_sim_joints(
            robot_action_policy, self.gr00t_joints_config, self.sim_gr1_action_joint_config, self.args.simulation_device
        )

        return robot_action_sim, backbone_embedding

    def reset(self):
        """Resets the policy's internal state."""
        # As GN1 is a single-shot policy, we don't need to reset its internal state
        pass
