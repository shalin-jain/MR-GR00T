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
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .mr_gr00t_marl_env_cfg import MrGr00tMarlEnvCfg


class MrGr00tMarlEnv(DirectMARLEnv):
    cfg: MrGr00tMarlEnvCfg

    def __init__(self, cfg: MrGr00tMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robots = {}
        self.processed_actions = {}

    def _setup_scene(self):
        self.robot_1 = Articulation(self.cfg.robot_1_cfg)
        self.robot_2 = Articulation(self.cfg.robot_2_cfg)

        # add articulations to scene
        self.scene.articulations["robot_1"] = self.robot_1
        self.scene.articulations["robot_2"] = self.robot_2

        # add to robot dictionary
        self.robots["robot_1"] = {
            "articulation": self.robot_1
            "camera": self.cfg.robot_1_pov_cam
        }
        self.robots["robot_2"] = {
            "articulation": self.robot_2
            "camera": self.cfg.robot_2_pov_cam
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
            groot_action = 0 # groot inference here!
            self.processed_actions[robot_id] = actions[robot_id] + groot_action

    def _apply_action(self) -> None:
        """
        Apply processed actions to configured joints.
        """
        for robot_id, robot in self.robots.items():
            articulation = robot["articulation"]
            articulation.set_joint_position_target(self.processed_actions[robot_id], self.cfg.joint_names)

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

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Get dones.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: tuple (dones, terminations)
        """
        dones = {}
        terminations = {}
        for robot_id, robot in self.robots.items:
            # TODO: fix this
            dones[robot_id] = torch.zeros((self.scene.num_envs,), device=self.device)
            terminations[robot_id] = torch.zeros((self.scene.num_envs,), device=self.device)
        return dones, terminations

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Batched reset of environments by id.

        Args:
            env_ids (Sequence[int]): environment id's to reset.
        """
        self.scene.reset(env_ids)

        ######################################################################
        # Copied from Original DirectMARLEnv Implementation
        ######################################################################
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
        ######################################################################
