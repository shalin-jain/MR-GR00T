# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with a trained PPO residual policy."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="PPO residual policy agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO policy checkpoint (.pt file)")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import MR_GR00T.tasks  # noqa: F401

import numpy as np
from scipy.stats import wasserstein_distance


class GaussianPolicy(nn.Module):
    """Gaussian policy network that outputs mean and log_std for action distribution."""

    def __init__(self, state_dim, action_dim, hidden_dims=[512, 512],
                 min_log_std=-20.0, max_log_std=2.0, initial_log_std=-4.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        # Build network layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim

        # Mean output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Bound actions to [-1, 1]

        self.net_container = nn.Sequential(*layers)

        # Log std parameter (learnable)
        self.log_std_parameter = nn.Parameter(torch.ones(action_dim) * initial_log_std)

    def forward(self, state, deterministic=False):
        """
        Forward pass of policy.

        Args:
            state: Input state tensor
            deterministic: If True, return mean action. If False, sample from distribution.

        Returns:
            action: Sampled action (or mean if deterministic)
            log_prob: Log probability of the action (None if deterministic)
        """
        # Get mean from network
        mean = self.net_container(state)

        if deterministic:
            # Return mean action directly
            return mean, None
        else:
            # Sample from Gaussian distribution
            log_std = torch.clamp(self.log_std_parameter, self.min_log_std, self.max_log_std)
            std = torch.exp(log_std)

            # Create distribution and sample
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

            # Clip action to [-1, 1] range
            action = torch.clamp(action, -1.0, 1.0)

            return action, log_prob


def load_ppo_policy(checkpoint_path, device):
    """
    Load PPO policy from skrl checkpoint.

    The skrl checkpoint format typically contains:
    - 'policy': state_dict of the policy network
    - 'value': state_dict of the value network
    - 'optimizer_policy': optimizer state
    - 'optimizer_value': optimizer state
    - (potentially) 'timestep', 'learning_rate', etc.
    """
    print(f"[INFO] Loading PPO checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Print checkpoint structure for debugging
    print(f"[INFO] Checkpoint keys: {list(checkpoint.keys())}")

    # Extract policy state dict
    if 'policy' in checkpoint:
        policy_state_dict = checkpoint['policy']
        print(f"[INFO] Found 'policy' key in checkpoint")
    elif 'models' in checkpoint and 'policy' in checkpoint['models']:
        policy_state_dict = checkpoint['models']['policy']
        print(f"[INFO] Found 'models.policy' key in checkpoint")
    else:
        # Fallback: assume checkpoint is the state dict itself
        policy_state_dict = checkpoint
        print(f"[INFO] Using entire checkpoint as state_dict")

    print(f"[INFO] Policy state_dict keys: {list(policy_state_dict.keys())[:10]}...")  # Show first 10 keys

    # Infer dimensions from state dict
    # Look for first layer weights to determine input dimension
    first_layer_key = None
    for key in policy_state_dict.keys():
        if ('net_container.0.weight' in key or
            '0.weight' in key or
            'net.0.weight' in key):
            first_layer_key = key
            break

    if first_layer_key is None:
        raise ValueError(f"Could not find first layer weights in checkpoint. Keys: {list(policy_state_dict.keys())}")

    first_layer_weight = policy_state_dict[first_layer_key]
    state_dim = first_layer_weight.shape[1]  # Input dimension
    hidden_dim = first_layer_weight.shape[0]  # Hidden dimension

    # Look for output layer to determine action dimension
    output_layer_key = None
    for key in policy_state_dict.keys():
        if 'weight' in key and 'log_std' not in key.lower():
            output_layer_key = key

    if output_layer_key is None:
        raise ValueError(f"Could not find output layer weights in checkpoint")

    output_layer_weight = policy_state_dict[output_layer_key]
    action_dim = output_layer_weight.shape[0]  # Output dimension

    print(f"[INFO] Inferred dimensions from checkpoint:")
    print(f"  - State dimension: {state_dim}")
    print(f"  - Action dimension: {action_dim}")
    print(f"  - First hidden dimension: {hidden_dim}")

    # Detect hidden layer structure
    hidden_dims = []
    for i in range(10):  # Check up to 10 layers
        layer_key = f'net_container.{i*2}.weight'  # Linear layers are at even indices (0, 2, 4, ...)
        if layer_key in policy_state_dict:
            hidden_dims.append(policy_state_dict[layer_key].shape[0])
        else:
            break

    # Remove last dimension (output layer)
    if hidden_dims:
        hidden_dims = hidden_dims[:-1]

    if not hidden_dims:
        hidden_dims = [512, 512]  # Default fallback

    print(f"  - Hidden dimensions: {hidden_dims}")

    # Create model with inferred dimensions
    model = GaussianPolicy(state_dim, action_dim, hidden_dims=hidden_dims).to(device)

    # Load state dict with strict=False to handle potential mismatches
    result = model.load_state_dict(policy_state_dict, strict=False)

    print(f"[INFO] Policy loaded successfully")
    if result.missing_keys:
        print(f"[WARNING] Missing keys in checkpoint: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[WARNING] Unexpected keys in checkpoint: {result.unexpected_keys}")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {total_params:,} total ({trainable_params:,} trainable)")

    # Extract additional checkpoint info if available
    if 'timestep' in checkpoint:
        print(f"[INFO] Checkpoint timestep: {checkpoint['timestep']}")
    if 'iteration' in checkpoint:
        print(f"[INFO] Checkpoint iteration: {checkpoint['iteration']}")

    model.eval()
    return model


def main():
    """Run PPO residual policy agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": "/home/sjain441/MR-GR00T/MR_GR00T/videos",
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # load PPO policy
    device = env.unwrapped.device
    ppo_policy = load_ppo_policy(args_cli.checkpoint, device)

    print(f"\n[INFO] Running policy in deterministic mode (using mean actions)\n")

    # reset environment
    obs_dict, _ = env.reset()

    # log actions to compute the mean and std
    action_log = {}

    # simulate environment
    steps = 0
    episodes = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get observations (includes VLA action at the end)
            state = obs_dict["policy"]

            # Get action from policy (always deterministic - use mean)
            action, _ = ppo_policy(state, deterministic=True)

            # Step environment with predicted residual
            obs_dict, _, _, _, _ = env.step(action)

            # get curriculum level
            # only log first 100 steps of each episode
            # if steps < 100:
            curriculum_level = env.unwrapped.curriculum_manager.curriculum_level
            if curriculum_level not in action_log:
                action_log[curriculum_level] = []
            action_log[curriculum_level].append(action.cpu().numpy())
            steps += 1
        if steps % 300 == 0:
            episodes += 50  # assuming num_envs=50
            steps = 0
            print(f"[INFO] Episodes simulated: {episodes}")
        if episodes > 2000:
            break

    # compute action statistics and Wasserstein distances
    action_stats = {}
    wasserstein_distances = {}

    # Get level 0 actions as reference (if available)
    reference_level = min(action_log.keys()) if action_log else None
    reference_actions = None
    reference_actions_normalized = None

    if reference_level is not None:
        reference_actions = np.concatenate(action_log[reference_level], axis=0)
        # Normalize reference actions per dimension (z-score normalization)
        reference_mean = np.mean(reference_actions, axis=0)
        reference_std = np.std(reference_actions, axis=0)
        # Avoid division by zero
        reference_std = np.where(reference_std == 0, 1.0, reference_std)
        reference_actions_normalized = (reference_actions - reference_mean) / reference_std

    for level, actions in action_log.items():
        actions_array = np.concatenate(actions, axis=0)
        mean_action = np.mean(actions_array, axis=0)
        std_action = np.std(actions_array, axis=0)

        # Normalize actions using reference statistics for fair comparison
        actions_normalized = None
        if reference_level is not None:
            actions_normalized = (actions_array - reference_mean) / reference_std

        # Compute Wasserstein distance per action dimension on normalized actions
        wasserstein_dists = []
        if reference_actions_normalized is not None and level != reference_level and actions_normalized is not None:
            for dim in range(actions_normalized.shape[1]):
                wd = wasserstein_distance(reference_actions_normalized[:, dim], actions_normalized[:, dim])
                wasserstein_dists.append(wd)

        # Average Wasserstein distance across all action dimensions
        avg_wasserstein = np.mean(wasserstein_dists) if wasserstein_dists else 0.0

        action_stats[level] = {
            "mean_action": mean_action.tolist(),
            "std_action": std_action.tolist(),
            "wasserstein_distance_from_level_{}".format(reference_level): avg_wasserstein,
            "wasserstein_per_dim": wasserstein_dists
        }
        wasserstein_distances[level] = avg_wasserstein

    # Save to json
    with open("action_stats.json", "w") as f:
        import json
        json.dump(action_stats, f, indent=4)

    print("\n[INFO] Action statistics per curriculum level:")
    for level in sorted(action_log.keys()):
        actions_array = np.concatenate(action_log[level], axis=0)
        mean_action = np.mean(actions_array, axis=0)
        std_action = np.std(actions_array, axis=0)
        wd = wasserstein_distances.get(level, 0.0)
        print(f"  - Level {level}:")
        print(f"      Mean action magnitude: {np.linalg.norm(mean_action):.4f}")
        print(f"      Mean std: {np.mean(std_action):.4f}")
        print(f"      Wasserstein distance from level {reference_level}: {wd:.4f}")

    # close the simulator
    env.close()
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
