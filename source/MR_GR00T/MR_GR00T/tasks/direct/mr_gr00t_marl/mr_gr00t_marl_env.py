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
from isaaclab.utils.math import sample_uniform

from .mr_gr00t_marl_env_cfg import MrGr00tMarlEnvCfg


class MrGr00tMarlEnv(DirectMARLEnv):
    cfg: MrGr00tMarlEnvCfg

    def __init__(self, cfg: MrGr00tMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # get joint names and ids
        self._joint_ids, self._joint_names = self.robot_1.find_joints(self.cfg.joint_names, preserve_order=True)

        # store processed_actions
        self.processed_actions = {}

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

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """
        Process the given action as a residual action to GR00T N1 Inference.

        Args:
            actions (dict[str, torch.Tensor]): dictionary of actions per robot id.
        """
        # TODO: this needs to be set to actual gr00t inference
        for robot_id, robot in self.robots.items():
            articulation = robot["articulation"]
            camera = robot["camera"]
            groot_action = articulation.data.default_joint_pos[:, self._joint_ids].clone() # TODO: groot inference goes here!
            self.processed_actions[robot_id] = actions[robot_id] + groot_action

    def _apply_action(self) -> None:
        """
        Apply processed actions to configured joints.
        """
        for robot_id, robot in self.robots.items():
            articulation = robot["articulation"]
            articulation.set_joint_position_target(self.processed_actions[robot_id], self._joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """
        Get observations for the residual policy.

        Returns:
            dict[str, torch.Tensor]: dictionary of observations per robot id.
        """
        obs = {}
        for robot_id, robot in self.robots.items():
            # TODO: fix this
            obs[robot_id] = torch.zeros((self.scene.num_envs, 1), device=self.device)
        return obs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """
        Get rewards.

        Returns:
            dict[str, torch.Tensor]: dictionary of rewards per robot id.
        """
        rew = {}
        for robot_id, robot in self.robots.items():
            # TODO: fix this
            rew[robot_id] = torch.zeros((self.scene.num_envs,), device=self.device)
        return rew

    def _get_terminations(self) -> torch.Tensor:
        """
        Get environment terminations.
        Returns:
            A tensor indicating which environments have terminated. Shape is (num_envs,)
        """

        # TODO: see if we want earlier terminations
        return torch.zeros((self.scene.num_envs,), device=self.device)

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

