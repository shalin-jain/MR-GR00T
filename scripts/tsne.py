# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to analyze VLA embeddings from an Isaac Lab environment.

This script launches the specified task with many parallel environments.
It manually resets the target object's position to a different random
location in each environment. It then performs a single simulation step
(with zero actions) to trigger the VLA inference.

Finally, it collects the VLA's backbone embeddings, runs t-SNE to reduce
them to 2D, and plots the result, colored by the object's initial X and Y
positions. This helps verify that the VLA embeddings capture spatial
information about the scene.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="VLA embedding analysis for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--num_envs", type=int, default=512, help="Number of environments to simulate. (Recommended: 500+ for t-SNE)"
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import DirectMARLEnv

import MR_GR00T.tasks  # noqa: F401


def main():
    """Run VLA inference over randomized object locations and plot embeddings."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info
    print(f"[INFO]: Task: {args_cli.task}")
    print(f"[INFO]: Number of environments: {env.unwrapped.num_envs}")
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment to initialize all actors and buffers
    print("[INFO]: Resetting environment to initialize...")
    env.reset()
    print("[INFO]: Environment reset complete.")

    # # check that this is the environment we expect
    # if not isinstance(env.unwrapped, DirectMARLEnv) or "object" not in env.unwrapped.scene:
    #     print(f"[ERROR]: This script requires a DirectMARLEnv with an actor named 'object' in the scene.")
    #     print("       Please run this with your MrGr00tMarlEnv task.")
    #     env.close()
    #     return

    # get actors for easy access
    object_actor = env.unwrapped.scene["object"]
    # We'll just get embeddings from robot_1's perspective
    robot = env.unwrapped.robots["robot_1"]
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    # Collect at least 1000 data points
    min_samples = 1000
    num_iterations = max(1, (min_samples + num_envs - 1) // num_envs)  # Ceiling division

    print(f"[INFO]: Collecting {num_iterations * num_envs} samples over {num_iterations} iterations...")

    all_embeddings = []
    all_positions_x = []
    all_positions_y = []

    # create zero actions once
    actions = {}
    for robot_id in env.unwrapped.robots.keys():
        actions[robot_id] = torch.zeros(
            (env.unwrapped.num_envs, env.action_space(robot_id).shape[-1]), device=env.unwrapped.device
        )

    for iteration in range(num_iterations):
        print(f"[INFO]: Iteration {iteration + 1}/{num_iterations}")

        # --- 1. Manually set object positions ---
        # define a randomization range (relative to env origin)
        # randomize X position only, keep Y fixed
        x_coords = ((torch.rand(num_envs, device=device) - 0.5) * 0.25) - 0.5  # -0.75m to +0.75m
        # x_coords = torch.zeros(num_envs, device=device) - 0.5  # -0.75m to +0.75m
        y_coords = ((torch.rand(num_envs, device=device) - 0.5) * 0.1) + 0.31  # Fixed at 0.0m
        # y_coords = torch.zeros(num_envs, device=device) + 0.31  # Fixed at 0.0m
        # Use the object's configured initial Z height (1.2590m) to keep it visible on the table
        z_coord = 1.2590
        new_positions = torch.stack([x_coords, y_coords, torch.full_like(x_coords, z_coord)], dim=1)

        # get default poses (position + quaternion)
        default_pose = object_actor.data.default_root_state[:, :7].clone()
        # add env origins to our relative positions
        default_pose[:, :3] = new_positions + env.unwrapped.scene.env_origins[:, :3]

        # write new poses to simulation (must be outside inference mode)
        object_actor.write_root_pose_to_sim(default_pose)
        # reset velocities to zero
        object_actor.write_root_velocity_to_sim(torch.zeros_like(object_actor.data.default_root_state[:, 7:]))

        # --- 2. Run one step to get VLA embeddings ---
        # this will trigger _get_observations -> _vla_inference
        # which will populate robot["vla_backbone_embedding"]
        env.step(actions)

        # --- 3. Collect data from this iteration ---
        all_embeddings.append(robot["vla_backbone_embedding"].cpu().numpy())
        all_positions_x.append(x_coords.cpu().numpy())
        all_positions_y.append(y_coords.cpu().numpy())

        # reset vla counters to force inferece next step
        robot["vla_counter"] = torch.ones(num_envs, dtype=torch.int, device=device) * 15

    # Concatenate all collected data
    print("[INFO]: Combining all collected data...")
    embeddings = np.concatenate(all_embeddings, axis=0)
    positions_x = np.concatenate(all_positions_x, axis=0)
    positions_y = np.concatenate(all_positions_y, axis=0)

    if embeddings.shape[0] == 0 or np.all(embeddings == 0):
        print("[ERROR]: Embeddings are all zero or empty. VLA inference might not have run correctly.")
        env.close()
        return

    print(f"[INFO]: Collected {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")
    print(f"[INFO]: X-position range: {positions_x.min():.3f} to {positions_x.max():.3f} m")
    print(f"[INFO]: Y-position range: {positions_y.min():.3f} to {positions_y.max():.3f} m")

    # --- 4. Run t-SNE ---

    # t-SNE perplexity is ideally < num_samples / 3
    num_samples = embeddings.shape[0]
    if num_samples < 50:
        print(f"[WARN]: Only {num_samples} samples. t-SNE may be unreliable. Try more iterations.")
        perplexity_val = max(5, num_samples - 1)
    else:
        perplexity_val = min(50, num_samples // 3)  # Adaptive perplexity

    print(f"[INFO]: Running t-SNE on {num_samples} samples (perplexity={perplexity_val})... This may take a moment.")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, max_iter=1000, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings)
    print("[INFO]: t-SNE complete.")

    # --- 5. Plot results ---
    print("[INFO]: Generating plot...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))
    fig.suptitle(f't-SNE of VLA Embeddings (n={embeddings.shape[0]}) - Task: {args_cli.task}', fontsize=16)

    # Plot 1: t-SNE colored by X-position
    sc1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=positions_x, cmap='viridis', alpha=0.7)
    fig.colorbar(sc1, ax=ax1, label='Object X-Position (m)')
    ax1.set_title('t-SNE Colored by X-Position')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: t-SNE colored by Y-position
    sc2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=positions_y, cmap='plasma', alpha=0.7)
    fig.colorbar(sc2, ax=ax2, label='Object Y-Position (m)')
    ax2.set_title('t-SNE Colored by Y-Position')
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Plot 3: 2D Scatter of actual object positions (X vs Y)
    sc3 = ax3.scatter(positions_x, positions_y, c=np.arange(len(positions_x)), cmap='coolwarm', alpha=0.7)
    fig.colorbar(sc3, ax=ax3, label='Sample Index')
    ax3.set_title('Actual Object Positions (X vs Y)')
    ax3.set_xlabel('Object X-Position (m)')
    ax3.set_ylabel('Object Y-Position (m)')
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # save the plot
    output_filename = "vla_embeddings_tsne.png"
    plt.savefig(output_filename)
    print(f"[SUCCESS]: Plot saved to {output_filename}")

    # You can optionally show the plot if you are in a GUI environment
    # plt.show()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()