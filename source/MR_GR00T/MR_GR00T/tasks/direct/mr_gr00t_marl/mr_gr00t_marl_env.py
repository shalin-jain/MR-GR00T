# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, saturate

from MR_GR00T.vla.gr00t_n1 import Gr00tN1
from MR_GR00T.utils.robot_joints import JointsAbsPosition
from .mr_gr00t_marl_env_cfg import MrGr00tMarlEnvCfg
from .mdp.observations import (
    object_obs,
    get_left_eef_pos,
    get_left_eef_quat,
    get_right_eef_pos,
    get_right_eef_quat,
    get_all_robot_link_state
)


class MrGr00tMarlEnv(DirectMARLEnv):
    cfg: MrGr00tMarlEnvCfg

    def __init__(self, cfg: MrGr00tMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # get joint names and ids
        self._joint_ids, self._joint_names = self.robot_1.find_joints(self.cfg.joint_names, preserve_order=True)

        # action processing
        self.processed_actions = {}
        robot_joint_limits = self.robot_1.root_physx_view.get_dof_limits().to(self.device)
        self.low_robot_joint_limits = robot_joint_limits[..., 0][:, self._joint_ids]
        self.high_robot_joint_limits = robot_joint_limits[..., 1][:, self._joint_ids]

        # load gr00t model, initialize relevant variables
        self.vla = Gr00tN1(self.cfg.vla.args)
        self.vla_chunk = self.cfg.vla.args.num_feedback_actions
        for robot_id, robot in self.robots.items():
            # initialize vla state
            robot["vla_state"] = JointsAbsPosition(
                robot["articulation"].data.joint_pos, self.vla.sim_gr1_state_joint_config, self.device
            )

            # initialize vla counter for tracking chunk execution
            robot["vla_counter"] = torch.zeros((self.scene.num_envs,), device=self.device) + (self.vla_chunk-1)
            robot["vla_counter"] = robot["vla_counter"].int()

            # initialize vla actions
            robot["vla_actions"] = robot["articulation"].data.default_joint_pos[:, self._joint_ids].clone()
            robot["vla_actions"] = robot["vla_actions"].unsqueeze(1).repeat(1, self.vla_chunk, 1)  # expand out chunk dim
            robot["vla_backbone_embedding"] = torch.zeros((self.scene.num_envs, self.cfg.vla.backbone_embedding_dim), device=self.device)

            # get configured lanugage commands
            robot["vla_command"] = self.cfg.vla.commands[robot_id]

    def _setup_scene(self):
        """
        Initialize robots and associated cameras
        """
        self.robots = {}
        self.robot_1 = Articulation(self.cfg.scene.robot_1_cfg)
        self.robot_2 = Articulation(self.cfg.scene.robot_2_cfg)
        self.camera_1 = self.scene.sensors["robot_1_pov_cam"]
        self.camera_2 = self.scene.sensors["robot_2_pov_cam"]

        # add articulations to scene
        self.scene.articulations["robot_1"] = self.robot_1
        self.scene.articulations["robot_2"] = self.robot_2


        # add to robot dictionary
        self.robots["robot_1"] = {
            "articulation": self.robot_1,
            "camera": self.camera_1
        }
        self.robots["robot_2"] = {
            "articulation": self.robot_2,
            "camera": self.camera_2
        }

    def _vla_inference(self, robot_id: str):
        """
        Run VLA Inference to set the VLA actions.

        Args:
            robot_id (str): robot id.
        """

        robot = self.robots[robot_id]

        # increment counters
        robot["vla_counter"] += 1

        # only update envs where chunk has been completed
        env_ids = robot["vla_counter"] >= self.vla_chunk

        # if no environments need inference, just increment all counters and return
        if not torch.any(env_ids):
            return

        # set the joint state
        joint_pos = robot["articulation"].data.joint_pos
        robot["vla_state"].set_joints_pos(joint_pos)

        # run VLA inference
        goal, backbone_embedding = self.vla.get_new_goal(
            robot["vla_state"],
            robot["camera"],
            robot["vla_command"]
        )
        goal_joint_pos = goal.get_joints_pos(self.device)

        # truncate to configured chunk length
        goal_joint_pos = goal_joint_pos[:, :self.vla_chunk, :]

        # update actions and embeddings for relevant envs
        robot["vla_actions"][env_ids, :, :] = goal_joint_pos[env_ids, :, :]
        vla_backbone_embedding = backbone_embedding["backbone_features"][env_ids].mean(dim=1)
        robot["vla_backbone_embedding"][env_ids] = vla_backbone_embedding.float()

        # update counters for relevant envs
        robot["vla_counter"][env_ids] = 0

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """
        Process the given action as a residual action to GR00T N1 Inference.

        Args:
            actions (dict[str, torch.Tensor]): dictionary of actions per robot id.
        """
        for robot_id, robot in self.robots.items():
            env_indices = torch.arange(self.scene.num_envs, device=self.device)
            groot_action = robot["vla_actions"][env_indices, robot["vla_counter"], :]
            self.processed_actions[robot_id] = actions[robot_id] + groot_action.squeeze()

    def _apply_action(self) -> None:
        """
        Apply processed actions to configured joints.
        """
        for robot_id, robot in self.robots.items():
            articulation = robot["articulation"]
            actions = saturate(
                self.processed_actions[robot_id],
                self.low_robot_joint_limits,
                self.high_robot_joint_limits,
            )
            articulation.set_joint_position_target(actions, self._joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """
        Get observations for the residual policy.

        Returns:
            dict[str, torch.Tensor]: dictionary of observations per robot id.
        """
        obs = {}
        for robot_id, _ in self.robots.items():
            self._vla_inference(robot_id)
            # pass in vla backbone embedding and action as observation (TODO: make this its own function?)
            env_indices = torch.arange(self.scene.num_envs, device=self.device)
            vla_action = self.robots[robot_id]["vla_actions"][env_indices, self.robots[robot_id]["vla_counter"], :].squeeze(1)
            obs[robot_id] = torch.concatenate([self.robots[robot_id]["vla_backbone_embedding"], vla_action], dim=-1)
        return obs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """
        Get rewards.

        Returns:
            dict[str, torch.Tensor]: dictionary of rewards per robot id.
        """
        rew = {}
        for robot_id, robot in self.robots.items():
            dist_to_bin = torch.norm(self.scene["object"].data.root_pos_w - self.scene["bin"].data.root_pos_w, dim=-1)
            progress_rew = -0.1 * dist_to_bin
            success_bonus = torch.where(dist_to_bin < 0.1, 10.0, 0.0)
            rew[robot_id] = progress_rew + success_bonus
        return rew

    def _get_terminations(self) -> torch.Tensor:
        """
        Get environment terminations.

        Returns:
            A tensor indicating which environments have terminated. Shape is (num_envs,)
        """

        # terminate if rod falls
        return torch.where(self.scene["object"].data.root_pos_w[:, 2] < 0.75, 1, 0)

    def _get_dones(self) -> tuple[dict, dict]:
        """
        Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out (keyed by the agent ID).
            Shape of individual tensors is (num_envs,).
        """
        time_outs = {}
        dones = {}
        time_out = self.episode_length_buf > self.max_episode_length
        for robot_id in self.robots.keys():
            time_outs[robot_id] = time_out
            dones[robot_id] = torch.logical_or(
                time_out,
                self._get_terminations()
            )

        return (time_outs, dones)

    def _reset_idx(self, env_ids: Sequence[int]):
        """
        Copied and minimally modified from default implementation.
        Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        self.scene.reset(env_ids)

        # set robot spawn configuration
        # by default reset tries to move them to this position which errors
        for _, robot in self.robots.items():
            articulation = robot["articulation"]
            articulation.write_joint_state_to_sim(articulation.data.default_joint_pos, articulation.data.default_joint_vel)

            # reset vla counters
            robot["vla_counter"] = torch.zeros((self.scene.num_envs,), device=self.device) + (self.vla_chunk-1)
            robot["vla_counter"] = robot["vla_counter"].int()

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # reset noise models
        if self.cfg.action_noise_model:
            for noise_model in self._action_noise_model.values():
                noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            for noise_model in self._observation_noise_model.values():
                noise_model.reset(env_ids)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0
