# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab_assets.robots.fourier import GR1T2_CFG
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from MR_GR00T.vla.config.args import Gr00tN1ClosedLoopArguments

joint_names_dict = {
    # arm joint
    "left_shoulder_pitch_joint": 0,
    "right_shoulder_pitch_joint": 1,
    "left_shoulder_roll_joint": 2,
    "right_shoulder_roll_joint": 3,
    "left_shoulder_yaw_joint": 4,
    "right_shoulder_yaw_joint": 5,
    "left_elbow_pitch_joint": 6,
    "right_elbow_pitch_joint": 7,
    "left_wrist_yaw_joint": 8,
    "right_wrist_yaw_joint": 9,
    "left_wrist_roll_joint": 10,
    "right_wrist_roll_joint": 11,
    "left_wrist_pitch_joint": 12,
    "right_wrist_pitch_joint": 13,
    # hand joints
    "L_index_proximal_joint": 14,
    "L_middle_proximal_joint": 15,
    "L_pinky_proximal_joint": 16,
    "L_ring_proximal_joint": 17,
    "L_thumb_proximal_yaw_joint": 18,
    "R_index_proximal_joint": 19,
    "R_middle_proximal_joint": 20,
    "R_pinky_proximal_joint": 21,
    "R_ring_proximal_joint": 22,
    "R_thumb_proximal_yaw_joint": 23,
    "L_index_intermediate_joint": 24,
    "L_middle_intermediate_joint": 25,
    "L_pinky_intermediate_joint": 26,
    "L_ring_intermediate_joint": 27,
    "L_thumb_proximal_pitch_joint": 28,
    "R_index_intermediate_joint": 29,
    "R_middle_intermediate_joint": 30,
    "R_pinky_intermediate_joint": 31,
    "R_ring_intermediate_joint": 32,
    "R_thumb_proximal_pitch_joint": 33,
    "L_thumb_distal_joint": 34,
    "R_thumb_distal_joint": 35,
}
joint_names = list(joint_names_dict.keys())
tuned_joint_names = ["left-arm", "right-arm"]

# Define arm-only joint names (indices 0-13: shoulders, elbows, wrists)
arm_joint_names = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
]

# Joint initialization values
initial_joint_pos ={
    # right-arm
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": -1.5708,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    # left-arm
    "left_shoulder_pitch_joint": -0.10933163,
    "left_shoulder_roll_joint": 0.43292055,
    "left_shoulder_yaw_joint": -0.15983289,
    "left_elbow_pitch_joint": -1.48233023,
    "left_wrist_yaw_joint": 0.2359135,
    "left_wrist_roll_joint": 0.26184522,
    "left_wrist_pitch_joint": 0.00830735,
    # right hand
    "R_index_intermediate_joint": 0.0,
    "R_index_proximal_joint": 0.0,
    "R_middle_intermediate_joint": 0.0,
    "R_middle_proximal_joint": 0.0,
    "R_pinky_intermediate_joint": 0.0,
    "R_pinky_proximal_joint": 0.0,
    "R_ring_intermediate_joint": 0.0,
    "R_ring_proximal_joint": 0.0,
    "R_thumb_distal_joint": 0.0,
    "R_thumb_proximal_pitch_joint": 0.0,
    "R_thumb_proximal_yaw_joint": -1.57,
    # left hand
    "L_index_intermediate_joint": 0.0,
    "L_index_proximal_joint": 0.0,
    "L_middle_intermediate_joint": 0.0,
    "L_middle_proximal_joint": 0.0,
    "L_pinky_intermediate_joint": 0.0,
    "L_pinky_proximal_joint": 0.0,
    "L_ring_intermediate_joint": 0.0,
    "L_ring_proximal_joint": 0.0,
    "L_thumb_distal_joint": 0.0,
    "L_thumb_proximal_pitch_joint": 0.0,
    "L_thumb_proximal_yaw_joint": -1.57,
    # --
    "head_.*": 0.0,
    "waist_.*": 0.0,
    ".*_hip_.*": 0.0,
    ".*_knee_.*": 0.0,
    ".*_ankle_.*": 0.0,
}
initial_joint_vel = {".*": 0.0}

##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Define robot
    robot_cfg: ArticulationCfg = GR1T2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/robot_1",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos=initial_joint_pos,
            joint_vel=initial_joint_vel,
        ),
    )

    # define POV cameras
    robot_pov_camera = TiledCameraCfg(
        height=160,
        width=256,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0, 0.12, 1.85418), rot=(-0.17246, 0.98502, 0.0, 0.0), convention="ros"
        ),
        prim_path="{ENV_REGEX_NS}/robot_1_camera",
        update_period=0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2)),
    )

    # Table
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/table.usd",
            scale=(1.0, 1.0, 1.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Fixed in place, cannot be moved by physics
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueExhaustPipe",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.04904, 0.31, 1.2590], rot=[0, 0, 1.0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_exhaust_pipe.usd",
            scale=(0.5, 0.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueSortingBin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.16605, 0.39, 0.98634], rot=[1.0, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd",
            scale=(1.0, 1.7, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# VLA config
##
@configclass
class VLACfg:
    args: Gr00tN1ClosedLoopArguments = Gr00tN1ClosedLoopArguments(model_path="/home/sjain441/MR-GR00T/MR_GR00T/nvidia/finetuned")
    command: str = "Pick up the blue pipe and place it in the bin on the right"
    backbone_embedding_dim: int = 1536
    state_embedding_dim : int = 64

##
# Curriculum config
##
@configclass
class CurriculumCfg:
    """Configuration for curriculum learning."""

    # Success rate thresholds
    success_threshold: float = 0.85
    """Success rate threshold (0-1) to trigger curriculum expansion. High because VLA starts strong."""

    # XY range parameters for object offset
    object_x_min_start: float = -0.02
    """Initial minimum X offset for object spawn (m)."""
    object_x_max_start: float = 0.02
    """Initial maximum X offset for object spawn (m)."""
    object_y_min_start: float = -0.02
    """Initial minimum Y offset for object spawn (m)."""
    object_y_max_start: float = 0.02
    """Initial maximum Y offset for object spawn (m)."""

    object_x_min_final: float = -0.1
    """Final minimum X offset for object spawn (m)."""
    object_x_max_final: float = 0.1
    """Final maximum X offset for object spawn (m)."""
    object_y_min_final: float = -0.1
    """Final minimum Y offset for object spawn (m)."""
    object_y_max_final: float = 0.1
    """Final maximum Y offset for object spawn (m)."""

    # XY range parameters for bin offset
    bin_x_min_start: float = -0.01
    """Initial minimum X offset for bin spawn (m)."""
    bin_x_max_start: float = 0.01
    """Initial maximum X offset for bin spawn (m)."""
    bin_y_min_start: float = -0.01
    """Initial minimum Y offset for bin spawn (m)."""
    bin_y_max_start: float = 0.01
    """Initial maximum Y offset for bin spawn (m)."""

    bin_x_min_final: float = -0.1
    """Final minimum X offset for bin spawn (m)."""
    bin_x_max_final: float = 0.1
    """Final maximum X offset for bin spawn (m)."""
    bin_y_min_final: float = -0.1
    """Final minimum Y offset for bin spawn (m)."""
    bin_y_max_final: float = 0.1
    """Final maximum Y offset for bin spawn (m)."""

    # Expansion parameters
    expansion_rate: float = 0.05
    """Amount to expand ranges by when threshold is met (fraction of final range)."""

    update_frequency: int = 200
    """Number of EPISODES (trials) between curriculum checks. With 50 envs, this is ~4 resets."""

    warmup_resets: int = 50
    """Number of episodes before curriculum starts expanding. With 50 envs, this is ~1 reset."""

@configclass
class SgGr00tRlEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5
    episode_length_s = 15.0

    # agent specification and spaces definition
    # Note: These will be updated in __post_init__ based on residual_arms_only setting
    action_space = len(joint_names)  # Will be overridden if residual_arms_only=True
    observation_space = 1536 + len(joint_names) + len(joint_names) # vla, joint positions, actions

    # bin dimensions are estimated as 0.25m x 0.5m x 0.1m
    bin_half_width_x: float = 0.25 / 2   # 0.25m / 2
    bin_half_width_y: float = 0.5 / 2  # 0.5m / 2
    bin_half_height_z: float = 0.1 / 2   # 0.1m / 2

    # synchronized reset for efficient VLA inference
    sync_reset: bool = True

    state_space = 0
    joint_names = joint_names
    residual_joint_names = joint_names  # Will be updated to arm_joint_names if residual_arms_only=True

    # vla
    vla = VLACfg()

    # curriculum (optional - set to None to disable)
    curriculum = CurriculumCfg()

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)
    num_rerenders_on_reset: int = 1

    # scene
    scene: InteractiveSceneCfg = ObjectTableSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)

    # actions
    action_scale: float = 2.0  # scale for actions - INCREASED to allow larger residual corrections
    residual_arms_only: bool = True  # if True, residual policy only controls arm joints (0-13), hands use VLA only

    # observations
    use_vla_obs: bool = False  # if True, use VLA backbone embedding in obs; if False, use explicit state (EEF, object, bin positions)

    def __post_init__(self):
        """Post initialization."""
        # Update action/observation space based on residual_arms_only setting
        if self.residual_arms_only:
            self.residual_joint_names = arm_joint_names
            self.action_space = len(arm_joint_names)  # Only 14 arm joints

            # Calculate observation space based on observation mode
            if self.use_vla_obs:
                # VLA mode: backbone embedding (1536) + all joint pos (36) + arm actions (14)
                self.observation_space = 1536 + len(self.joint_names) + len(arm_joint_names)
            else:
                # Explicit state mode: left_eef_pos (3) + right_eef_pos (3) + object_pos (3) + bin_pos (3) + all joint pos (36) + arm actions (14)
                self.observation_space = 3 + 3 + 3 + 3 + len(self.joint_names) + len(arm_joint_names)
        else:
            self.residual_joint_names = self.joint_names
            self.action_space = len(self.joint_names)

            # Calculate observation space based on observation mode
            if self.use_vla_obs:
                # VLA mode: backbone embedding (1536) + all joint pos (36) + all actions (36)
                self.observation_space = 1536 + len(self.joint_names) + len(self.joint_names)
            else:
                # Explicit state mode: left_eef_pos (3) + right_eef_pos (3) + object_pos (3) + bin_pos (3) + all joint pos (36) + all actions (36)
                self.observation_space = 3 + 3 + 3 + 3 + len(self.joint_names) + len(self.joint_names)

        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss
        self.wait_for_textures = False

        # replace the stiffness and dynamics in arm joints in the robot
        for joint_name in tuned_joint_names:
            self.scene.robot_cfg.actuators[joint_name].stiffness = 3000
            self.scene.robot_cfg.actuators[joint_name].damping = 100
