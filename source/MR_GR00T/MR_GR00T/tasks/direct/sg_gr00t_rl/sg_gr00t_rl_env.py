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
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import saturate

from MR_GR00T.vla.gr00t_n1 import Gr00tN1
from MR_GR00T.utils.robot_joints import JointsAbsPosition
from .sg_gr00t_rl_env_cfg import SgGr00tRlEnvCfg
from .mdp.curriculum import SpawnCurriculumManager
from .mdp.observations import (
    object_obs,
    get_left_eef_pos,
    get_left_eef_quat,
    get_right_eef_pos,
    get_right_eef_quat,
    get_all_robot_link_state
)


class SgGr00tRlEnv(DirectRLEnv):
    cfg: SgGr00tRlEnvCfg

    def __init__(self, cfg: SgGr00tRlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # get joint names and ids for ALL joints
        self._joint_ids, self._joint_names = self.robot["articulation"].find_joints(self.cfg.joint_names, preserve_order=True)

        # get residual joint ids (subset for arm-only mode)
        self._residual_joint_ids, self._residual_joint_names = self.robot["articulation"].find_joints(
            self.cfg.residual_joint_names, preserve_order=True
        )

        # Verify that residual joints are a prefix subset of all joints (arm joints should be first 14)
        if self.cfg.residual_arms_only:
            # Since joint_names has arms first and arm_joint_names is the first 14 elements,
            # residual_joint_ids should match the first 14 elements of joint_ids
            assert len(self._residual_joint_ids) == 14, f"Expected 14 arm joints, got {len(self._residual_joint_ids)}"
            # Convert to tensors for comparison if needed
            residual_ids_tensor = torch.as_tensor(self._residual_joint_ids, device=self.device)
            joint_ids_tensor = torch.as_tensor(self._joint_ids[:14], device=self.device)
            assert torch.all(residual_ids_tensor == joint_ids_tensor), \
                "Arm joints are not the first 14 joints in the ordered list! Check joint_names ordering."
            # Since they match, residual actions at positions 0-13 map directly to full actions at 0-13
            self.num_residual_joints = 14
        else:
            self.num_residual_joints = len(self._joint_ids)

        # action processing
        robot_joint_limits = self.robot["articulation"].root_physx_view.get_dof_limits().to(self.device)
        self.low_robot_joint_limits = robot_joint_limits[..., 0][:, self._joint_ids]
        self.high_robot_joint_limits = robot_joint_limits[..., 1][:, self._joint_ids]

        # store object, bin. and table for easy access
        self.object = self.scene["object"]
        self.bin = self.scene["bin"]
        self.table = self.scene["table"]

        # load gr00t model, initialize relevant variables
        self.vla = Gr00tN1(self.cfg.vla.args)
        self.vla_chunk = self.cfg.vla.args.num_feedback_actions

        # initialize vla state
        self.robot["vla_state"] = JointsAbsPosition(
            self.robot["articulation"].data.joint_pos, self.vla.sim_gr1_state_joint_config, self.device
        )

        # initialize vla counter for tracking chunk execution
        self.robot["vla_counter"] = torch.zeros((self.scene.num_envs,), device=self.device) + (self.vla_chunk-1)
        self.robot["vla_counter"] = self.robot["vla_counter"].int()

        # initialize vla actions
        self.robot["vla_actions"] = self.robot["articulation"].data.default_joint_pos[:, self._joint_ids].clone()
        self.robot["vla_actions"] = self.robot["vla_actions"].unsqueeze(1).repeat(1, self.vla_chunk, 1)  # expand out chunk dim

        # initialize vla embedding
        self.robot["vla_backbone_embedding"] = torch.zeros((self.scene.num_envs, self.cfg.vla.backbone_embedding_dim), device=self.device)
        self.robot["vla_state_embedding"] = torch.zeros((self.scene.num_envs, self.cfg.vla.state_embedding_dim), device=self.device)

        # get configured language command
        self.robot["vla_command"] = self.cfg.vla.command

        # initialize processed actions with default joint positions
        self.processed_actions = self.robot["articulation"].data.default_joint_pos[:, self._joint_ids].clone()
        self.residual_actions = torch.zeros_like(self.processed_actions)

        # track previous actions for action rate penalty
        self.previous_actions = self.processed_actions.clone()

        # initialize curriculum manager if enabled
        if hasattr(self.cfg, 'curriculum') and self.cfg.curriculum is not None:
            self.curriculum_manager = SpawnCurriculumManager(self.cfg.curriculum, self.device)
        else:
            self.curriculum_manager = None

        # initialize buffer for pending resets (for batched reset)
        self.pending_reset_buf = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.device)
        self.pending_reset_terminated = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.device)
        self.pending_reset_time_outs = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.device)

        # initialze logging dictionary
        self.extras["log"] = {}

    def _setup_scene(self):
        """
        Initialize robot and associated camera
        """
        self.articulation = Articulation(self.cfg.scene.robot_cfg)
        self.camera = self.scene.sensors["robot_pov_camera"]

        # add articulations to scene
        self.scene.articulations["robot"] = self.articulation
        self.scene.sensors["camera"] = self.camera

        # add to robot dictionary
        self.robot = {
            "articulation": self.articulation,
            "camera": self.camera
        }

    def _vla_inference(self):
        """
        Run VLA Inference to set the VLA actions.
        """

        robot = self.robot

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
        goal, backbone_embedding, action_inputs = self.vla.get_new_goal(
            robot["vla_state"],
            robot["camera"],
            robot["vla_command"]
        )
        goal_joint_pos = goal.get_joints_pos(self.device)

        # truncate to configured chunk length
        goal_joint_pos = goal_joint_pos[:, :self.vla_chunk, :]

        # update actions and embeddings for relevant envs
        robot["vla_actions"][env_ids, :, :] = goal_joint_pos[env_ids, :, :]

        # masked mean pooling of backbone embedding
        features = backbone_embedding["backbone_features"][env_ids]
        mask = backbone_embedding["backbone_attention_mask"][env_ids].unsqueeze(-1).float()
        vla_backbone_embedding = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        robot["vla_backbone_embedding"][env_ids] = vla_backbone_embedding.float()

        # vla state embedding
        state_embedding = action_inputs["state"][env_ids].squeeze(1)
        robot["vla_state_embedding"][env_ids] = state_embedding.float()

        # update counters for relevant envs
        robot["vla_counter"][env_ids] = 0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Process the given action as a residual action to GR00T N1 Inference.

        Args:
            actions (torch.Tensor): robot actions. Shape depends on cfg.residual_arms_only:
                - If True: (num_envs, 14) for arm joints only
                - If False: (num_envs, 36) for all joints
        """
        robot = self.robot
        env_indices = torch.arange(self.scene.num_envs, device=self.device)
        groot_action = robot["vla_actions"][env_indices, robot["vla_counter"], :]

        if self.cfg.residual_arms_only:
            # Residual policy only controls arm joints (first 14 in ordered list)
            # Create full residual action tensor, with zeros for hand joints
            full_residual_actions = torch.zeros_like(groot_action)
            arm_residual = actions * self.cfg.action_scale
            # Arm joints are verified to be at positions 0-13 (checked in __init__)
            full_residual_actions[:, :self.num_residual_joints] = arm_residual
            self.residual_actions = full_residual_actions

            # Log only arm residual statistics
            self.extras["log"]["residual_action_mean"] = torch.abs(arm_residual).mean()
            self.extras["log"]["residual_action_max"] = torch.abs(arm_residual).max()
            self.extras["log"]["residual_action_std"] = arm_residual.std()
            self.extras["log"]["policy_output_mean"] = torch.abs(actions).mean()
            self.extras["log"]["policy_output_max"] = torch.abs(actions).max()
        else:
            # Residual policy controls all joints
            self.residual_actions = actions * self.cfg.action_scale

            # Log all residual statistics
            self.extras["log"]["residual_action_mean"] = torch.abs(self.residual_actions).mean()
            self.extras["log"]["residual_action_max"] = torch.abs(self.residual_actions).max()
            self.extras["log"]["residual_action_std"] = self.residual_actions.std()
            self.extras["log"]["policy_output_mean"] = torch.abs(actions).mean()
            self.extras["log"]["policy_output_max"] = torch.abs(actions).max()

        self.processed_actions = self.residual_actions + groot_action.squeeze()
        # self.processed_actions = groot_action.squeeze()  # FOR DEBUG
        # self.processed_actions = robot['articulation'].data.default_joint_pos[:, self._joint_ids]  # FOR DEBUG

    def _apply_action(self) -> None:
        """
        Apply processed actions to configured joints.
        """
        robot = self.robot
        articulation = robot["articulation"]
        actions = saturate(
            self.processed_actions,
            self.low_robot_joint_limits,
            self.high_robot_joint_limits,
        )
        articulation.set_joint_position_target(actions, self._joint_ids)

    def step(self, action: torch.Tensor):
        """
        Override step to batch resets for more efficient VLA inference.

        Instead of resetting immediately when an environment is done, we accumulate
        pending resets and perform them in batches to enable parallel VLA inference.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step: update env counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # compute dones and rewards
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # update previous actions AFTER computing rewards (for next step's action rate penalty)
        self.previous_actions = self.processed_actions.clone()

        # handle resets: either synchronized (all at once) or immediate (as they finish)
        if self.cfg.sync_reset:
            # accumulate pending resets
            newly_done = self.reset_buf & ~self.pending_reset_buf
            if torch.any(newly_done):
                self.pending_reset_buf |= newly_done
                self.pending_reset_terminated |= self.reset_terminated & newly_done
                self.pending_reset_time_outs |= self.reset_time_outs & newly_done

            # only reset when ALL environments are done
            num_pending = torch.sum(self.pending_reset_buf).item()
            if num_pending == self.scene.num_envs:
                reset_env_ids = self.pending_reset_buf.nonzero(as_tuple=False).squeeze(-1)
                self._reset_idx(reset_env_ids)

                # clear pending reset buffers
                self.pending_reset_buf[:] = False
                self.pending_reset_terminated[:] = False
                self.pending_reset_time_outs[:] = False

                # if sensors are added to the scene, make sure we render to reflect changes in reset
                if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                    for _ in range(self.cfg.num_rerenders_on_reset):
                        self.sim.render()
        else:
            # immediate reset (default Isaac Lab behavior)
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self._reset_idx(reset_env_ids)
                # if sensors are added to the scene, make sure we render to reflect changes in reset
                if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                    for _ in range(self.cfg.num_rerenders_on_reset):
                        self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _get_observations(self) -> torch.Tensor:
        """
        Get observations for the residual policy.

        Returns:
            torch.Tensor: robot observations.

            If use_vla_obs=True:
                - VLA backbone embedding (1536 dim)
                - All joint positions (36 dim)
                - VLA actions for residual joints only (14 or 36 dim depending on cfg.residual_arms_only)

            If use_vla_obs=False:
                - Left EEF position (3 dim)
                - Right EEF position (3 dim)
                - Object position (3 dim)
                - Bin position (3 dim)
                - All joint positions (36 dim)
                - VLA actions for residual joints only (14 or 36 dim depending on cfg.residual_arms_only)
        """
        self._vla_inference()
        env_indices = torch.arange(self.scene.num_envs, device=self.device)
        vla_action_full = self.robot["vla_actions"][env_indices, self.robot["vla_counter"], :].squeeze(1)

        # Only include VLA actions for joints that residual policy controls
        if self.cfg.residual_arms_only:
            # Arm joints are at positions 0-13 (verified in __init__)
            vla_action_subset = vla_action_full[:, :self.num_residual_joints]
        else:
            vla_action_subset = vla_action_full

        # Choose observation mode
        if self.cfg.use_vla_obs:
            # VLA embedding mode
            obs = torch.concatenate(
                [
                    self.robot["vla_backbone_embedding"],
                    self.robot["articulation"].data.joint_pos[:, self._joint_ids],  # All joint positions
                    vla_action_subset  # VLA actions for controlled joints only
                ],
                dim=-1
            )
        else:
            # Explicit state mode
            # Get end effector positions
            body_pos_w = self.robot["articulation"].data.body_pos_w
            left_eef_idx = self.robot["articulation"].data.body_names.index("left_hand_roll_link")
            right_eef_idx = self.robot["articulation"].data.body_names.index("right_hand_roll_link")
            left_eef_pos = body_pos_w[:, left_eef_idx] - self.scene.env_origins
            right_eef_pos = body_pos_w[:, right_eef_idx] - self.scene.env_origins

            # Get object and bin positions (relative to env origins)
            object_pos = self.object.data.root_pos_w - self.scene.env_origins
            bin_pos = self.bin.data.root_pos_w - self.scene.env_origins

            obs = torch.concatenate(
                [
                    left_eef_pos,  # 3 dim
                    right_eef_pos,  # 3 dim
                    object_pos,  # 3 dim
                    bin_pos,  # 3 dim
                    self.robot["articulation"].data.joint_pos[:, self._joint_ids],  # All joint positions (36 dim)
                    vla_action_subset  # VLA actions for controlled joints only (14 or 36 dim)
                ],
                dim=-1
            )

        return {"policy": obs}

    def _check_object_in_bin(self, object_pos: torch.Tensor, bin_pos: torch.Tensor) -> torch.Tensor:
        """
        Check if objects are within the bin's 3D bounding box.

        Args:
            object_pos: Object positions in world frame. Shape: (num_envs, 3)
            bin_pos: Bin positions in world frame. Shape: (num_envs, 3)

        Returns:
            Boolean tensor indicating success for each environment. Shape: (num_envs,)
        """
        # calculate relative position
        rel_pos = object_pos - bin_pos

        # get bin dimensions from config
        bin_half_width_x = self.cfg.bin_half_width_x
        bin_half_width_y = self.cfg.bin_half_width_y
        bin_half_height_z = self.cfg.bin_half_height_z

        # check if object is within bin bounds in each dimension
        within_x = torch.abs(rel_pos[:, 0]) < bin_half_width_x
        within_y = torch.abs(rel_pos[:, 1]) < bin_half_width_y
        within_z = torch.abs(rel_pos[:, 2]) < bin_half_height_z

        # success only if within bounds in all three dimensions
        return within_x & within_y & within_z

    def _get_rewards(self) -> torch.Tensor:
        """
        Get rewards.

        Returns:
            torch.Tensor: robot rewards.
        """
        # For fine-tuning from VLA: Use sparse rewards to preserve good initialization
        # Only provide signal on success, let RL refine the VLA policy

        # check if object is in bin for success bonus
        success = self._check_object_in_bin(
            self.scene["object"].data.root_pos_w,
            self.scene["bin"].data.root_pos_w
        )
        success_rew = torch.where(success, 10.0, 0.0)

        # action rate penalty: penalize large changes in actions to encourage smoothness
        # compute L2 norm of action differences
        action_rate = torch.norm(self.processed_actions - self.previous_actions, dim=-1)
        action_rate_penalty = -1.0e-1 * action_rate

        # action norm penalty: penalize large residual actions to encourage minimal intervention
        # only penalize the residual (policy output), not the VLA baseline
        residual_norm = torch.norm(self.residual_actions, dim=-1)
        action_norm_penalty = -1.0e-1 * residual_norm

        # log action rate penalty
        self.extras["log"]["action_rate_penalty"] = action_rate_penalty.mean()
        self.extras["log"]["action_norm_penalty"] = action_norm_penalty.mean()

        return success_rew + action_rate_penalty + action_norm_penalty

    def _get_terminations(self) -> torch.Tensor:
        """
        Get environment terminations.

        Returns:
            A tensor indicating which environments have terminated. Shape is (num_envs,)
        """

        # terminate if rod falls
        fell = self.object.data.root_pos_w[:, 2] < 0.75

        # terminate if rod is successfully placed in bin
        success = self._check_object_in_bin(
            self.object.data.root_pos_w,
            self.bin.data.root_pos_w
        )

        return torch.where(fell | success, 1, 0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).
        """
        time_outs = self.episode_length_buf > self.max_episode_length
        dones = torch.logical_or(
            time_outs,
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
        # check if objects are in bin
        success = self._check_object_in_bin(
            self.object.data.root_pos_w[env_ids],
            self.bin.data.root_pos_w[env_ids]
        )
        success_rate = success.float().mean()

        # also log 2D distance for comparison
        rel_pos = self.object.data.root_pos_w[env_ids] - self.bin.data.root_pos_w[env_ids]
        dist_to_bin = torch.norm(rel_pos[:, :3], dim=-1)
        self.extras["log"]["final_dist_to_bin"] = dist_to_bin.mean()
        self.extras["log"]["success_rate"] = success_rate * 100.0  # Convert to percentage for logging
        print(f"[INFO] Resetting {len(env_ids)} environments. Success Rate: {success_rate*100.0:.2f}%, Final Dist to Bin: {self.extras['log']['final_dist_to_bin']:.3f} m")

        # update curriculum if enabled
        if self.curriculum_manager is not None:
            curriculum_expanded = self.curriculum_manager.update(success_rate, len(env_ids))
            if curriculum_expanded:
                print(f"[INFO] Curriculum expanded to level {self.curriculum_manager.curriculum_level}")
                print(f"       Object XY ranges: [{self.curriculum_manager.object_x_min:.3f}, {self.curriculum_manager.object_x_max:.3f}] x [{self.curriculum_manager.object_y_min:.3f}, {self.curriculum_manager.object_y_max:.3f}]")
                print(f"       Bin XY ranges: [{self.curriculum_manager.bin_x_min:.3f}, {self.curriculum_manager.bin_x_max:.3f}] x [{self.curriculum_manager.bin_y_min:.3f}, {self.curriculum_manager.bin_y_max:.3f}]")
            # add curriculum info to extras for logging
            self.extras["log"].update(self.curriculum_manager.get_info())

        # set robot spawn configuration
        robot = self.robot
        articulation = robot["articulation"]

        # reset joint states
        articulation.write_joint_state_to_sim(
            articulation.data.default_joint_pos[env_ids, :],
            articulation.data.default_joint_vel[env_ids, :],
            env_ids=env_ids
        )

        # set joint position targets to default positions
        default_joint_targets = articulation.data.default_joint_pos[env_ids][:, self._joint_ids]
        articulation.set_joint_position_target(
            default_joint_targets,
            joint_ids=self._joint_ids,
            env_ids=env_ids
        )

        # reset vla counters to 0
        robot["vla_counter"][env_ids] = 0
        robot["vla_actions"][env_ids] = default_joint_targets.unsqueeze(1).repeat(1, self.vla_chunk, 1)
        robot["vla_backbone_embedding"][env_ids] = torch.zeros((env_ids.shape[0], self.cfg.vla.backbone_embedding_dim), device=self.device)

        # reset previous actions for action rate penalty
        self.previous_actions[env_ids] = default_joint_targets

        # reset object states
        if self.curriculum_manager is not None:
            # sample object position from curriculum
            base_object_pos = self.object.data.default_root_state[0, :3].clone()
            object_positions = self.curriculum_manager.sample_object_position(base_object_pos, len(env_ids))
            default_object_state = self.object.data.default_root_state[env_ids].clone()
            default_object_state[:, :3] = object_positions + self.scene.env_origins[env_ids, :3]
        else:
            # use default position
            default_object_state = self.object.data.default_root_state[env_ids].clone()
            default_object_state[:, :3] += self.scene.env_origins[env_ids, :3]

        self.object.write_root_pose_to_sim(
            default_object_state[:, :7], env_ids
        )
        self.object.write_root_velocity_to_sim(
            torch.zeros_like(self.object.data.default_root_state[env_ids][:, 7:]), env_ids
        )

        # reset bin states (if curriculum enabled)
        if self.curriculum_manager is not None:
            # sample bin position from curriculum
            base_bin_pos = self.bin.data.default_root_state[0, :3].clone()
            bin_positions = self.curriculum_manager.sample_bin_position(base_bin_pos, len(env_ids))
            default_bin_state = self.bin.data.default_root_state[env_ids].clone()
            default_bin_state[:, :3] = bin_positions + self.scene.env_origins[env_ids, :3]

            self.bin.write_root_pose_to_sim(
                default_bin_state[:, :7], env_ids
            )
            self.bin.write_root_velocity_to_sim(
                torch.zeros_like(self.bin.data.default_root_state[env_ids][:, 7:]), env_ids
            )

        # reset table
        default_table_state = self.table.data.default_root_state[env_ids].clone()
        default_table_state[:, :3] += self.scene.env_origins[env_ids, :3]
        self.table.write_root_pose_to_sim(
            default_table_state[:, :7], env_ids
        )
        self.table.write_root_velocity_to_sim(
            torch.zeros_like(self.table.data.default_root_state[env_ids][:, 7:]), env_ids
        )

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
