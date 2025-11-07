# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab_assets.robots.fourier import GR1T2_CFG
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
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

    # Define two robots
    robot_1_cfg: ArticulationCfg = GR1T2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/robot_1",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.5, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos=initial_joint_pos,
            joint_vel=initial_joint_vel,
        ),
    )
    robot_2_cfg: ArticulationCfg = GR1T2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/robot_2",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.5, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos=initial_joint_pos,
            joint_vel=initial_joint_vel,
        ),
    )

    # define POV cameras
    robot_1_pov_cam = TiledCameraCfg(
        height=160,
        width=256,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.5, 0.12, 1.85418), rot=(-0.17246, 0.98502, 0.0, 0.0), convention="ros"
        ),
        prim_path="{ENV_REGEX_NS}/robot_1_camera",
        update_period=0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2)),
    )
    robot_2_pov_cam = TiledCameraCfg(
        height=160,
        width=256,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5, 0.12, 1.85418), rot=(-0.17246, 0.98502, 0.0, 0.0), convention="ros"
        ),
        prim_path="{ENV_REGEX_NS}/robot_2_camera",
        update_period=0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2)),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/table.usd",
            scale=(1.0, 1.0, 1.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueExhaustPipe",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.54904, 0.31, 1.2590], rot=[0, 0, 1.0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_exhaust_pipe.usd",
            scale=(0.5, 0.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueSortingBin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.66605, 0.39, 0.98634], rot=[1.0, 0, 0, 0]),
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
    commands: dict = {
        "robot_1": "Pick up the blue pipe and pass it to your left",
        "robot_2": "Wait to receive the blue pipe from the left and place it in the bin on the right"
    }
    backbone_embedding_dim: int = 1536

@configclass
class MrGr00tMarlEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 5
    episode_length_s = 20.0

    # multi-agent specification and spaces definition
    possible_agents = ["robot_1", "robot_2"]
    action_spaces = {"robot_1": len(joint_names), "robot_2": len(joint_names)}
    observation_spaces = {"robot_1": 1, "robot_2": 1} # TODO define these properly
    state_space = -1
    joint_names = joint_names

    # vla
    vla = VLACfg()

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    def __post_init__(self):
        """Post initialization."""
        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss
        self.wait_for_textures = False

        # replace the stiffness and dynamics in arm joints in the robot
        for joint_name in tuned_joint_names:
            self.scene.robot_1_cfg.actuators[joint_name].stiffness = 3000
            self.scene.robot_1_cfg.actuators[joint_name].damping = 100
            self.scene.robot_2_cfg.actuators[joint_name].stiffness = 3000
            self.scene.robot_2_cfg.actuators[joint_name].damping = 100
