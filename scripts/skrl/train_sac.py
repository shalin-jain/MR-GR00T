# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train single agent GR00T environment with SAC agent using skrl.

This script supports:
- Zero initialization of the last layer to initialize actor distribution about 0
- Orthogonal weight initialization for better training stability
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with SAC using skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Sg-Gr00t-Rl-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum training timesteps.")
parser.add_argument("--zero_init_last_layer", action="store_true", default=False, 
                    help="Initialize last layer of actor to zeros to center distribution about 0")
parser.add_argument("--orthogonal_init", action="store_true", default=False,
                    help="Use orthogonal initialization for network weights")
parser.add_argument("--orthogonal_gain", type=float, default=1.0,
                    help="Gain parameter for orthogonal initialization (default: 1.0)")
parser.add_argument("--bc_checkpoint", type=str, default=None,
                    help="Path to behavioral cloning checkpoint to initialize policy before RL training")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import torch.nn as nn
from datetime import datetime

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import Shape

import gymnasium as gym

import MR_GR00T.tasks  # noqa: F401


# Define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, initial_log_std=-4.0,
                 zero_init_last=False, orthogonal_init=False, orthogonal_gain=1.0):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, self.num_actions),
            nn.Tanh()
        )
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=initial_log_std))

        # Apply initialization
        if orthogonal_init:
            self._apply_orthogonal_init(orthogonal_gain)
        
        if zero_init_last:
            self._zero_init_last_layer()

    def _apply_orthogonal_init(self, gain):
        """Apply orthogonal initialization to all linear layers"""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _zero_init_last_layer(self):
        """Initialize last linear layer to zeros to center distribution about 0"""
        # Find the last linear layer (before Tanh)
        for module in reversed(list(self.net.modules())):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                break

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 orthogonal_init=False, orthogonal_gain=1.0, use_layer_norm=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # Build network with optional LayerNorm (RLPD recommendation)
        if use_layer_norm:
            self.net = nn.Sequential(
                nn.Linear(self.num_observations + self.num_actions, 1024),
                nn.LayerNorm(1024),
                nn.ELU(),
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ELU(),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.ELU(),
                nn.Linear(512, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(self.num_observations + self.num_actions, 1024),
                nn.ELU(),
                nn.Linear(1024, 512),
                nn.ELU(),
                nn.Linear(512, 512),
                nn.ELU(),
                nn.Linear(512, 1)
            )

        # Apply initialization
        if orthogonal_init:
            self._apply_orthogonal_init(orthogonal_gain)

    def _apply_orthogonal_init(self, gain):
        """Apply orthogonal initialization to all linear layers"""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


def main():
    """Train with SAC agent."""
    
    # Load configuration from the environment's registered config
    env_spec = gym.spec(args_cli.task)
    cfg_entry_point = env_spec.kwargs.get("skrl_sac_cfg_entry_point")
    
    # Parse the entry point to get the module and file
    if cfg_entry_point:
        import importlib
        from omegaconf import OmegaConf
        
        module_path, file_name = cfg_entry_point.split(":")
        module = importlib.import_module(module_path)
        cfg_file_path = os.path.join(os.path.dirname(module.__file__), file_name)
        
        # Load configuration from YAML
        agent_cfg = OmegaConf.to_container(OmegaConf.load(cfg_file_path), resolve=True)
    else:
        raise ValueError(f"No SAC config entry point found for task {args_cli.task}")
    
    # Set seed for reproducibility
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 42)
    set_seed(seed)
    
    # Load environment configuration
    env_spec = gym.spec(args_cli.task)
    env_cfg_entry_point = env_spec.kwargs.get("env_cfg_entry_point")
    
    if env_cfg_entry_point:
        import importlib
        module_path, class_name = env_cfg_entry_point.split(":")
        module = importlib.import_module(module_path)
        env_cfg_class = getattr(module, class_name)
        env_cfg = env_cfg_class()
        
        # Override configurations with CLI arguments
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = seed
    else:
        raise ValueError(f"No env config entry point found for task {args_cli.task}")
    
    # Setup logging directory
    log_root_path = os.path.join("logs", "skrl", agent_cfg.get("agent", {}).get("experiment", {}).get("directory", "sg_gr00t_rl"))
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_sac_torch"
    if args_cli.zero_init_last_layer:
        log_dir += "_zero_init"
    if args_cli.orthogonal_init:
        log_dir += f"_orth_{args_cli.orthogonal_gain}"
    
    # Update agent config with log directory
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)
    
    # Set log directory for environment
    env_cfg.log_dir = log_dir

    # Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print(f"  Video folder: {video_kwargs['video_folder']}")
        print(f"  Video interval: {args_cli.video_interval}")
        print(f"  Video length: {args_cli.video_length}")
        
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap around environment for skrl
    env = wrap_env(env)
    
    device = env.device

    # Instantiate a memory as rollout buffer (any memory can be used for this)
    memory_size = agent_cfg.get("memory", {}).get("memory_size", 100000)
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    # Get model configuration from YAML (can be overridden by command line args)
    model_cfg = agent_cfg.get("models", {})
    policy_cfg = model_cfg.get("policy", {})
    critic_cfg = model_cfg.get("critic", {})
    
    # Command line args override YAML config
    zero_init = args_cli.zero_init_last_layer if args_cli.zero_init_last_layer else policy_cfg.get("zero_init_last_layer", False)
    orthogonal_init = args_cli.orthogonal_init if args_cli.orthogonal_init else policy_cfg.get("orthogonal_init", False)
    orthogonal_gain = args_cli.orthogonal_gain if args_cli.orthogonal_init else policy_cfg.get("orthogonal_gain", 1.0)
    
    # Instantiate the agent's models (function approximators).
    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
    models = {}
    models["policy"] = StochasticActor(
        env.observation_space, env.action_space, device,
        clip_log_std=policy_cfg.get("clip_log_std", True),
        min_log_std=policy_cfg.get("min_log_std", -20),
        max_log_std=policy_cfg.get("max_log_std", 2),
        initial_log_std=policy_cfg.get("initial_log_std", -4.0),
        zero_init_last=zero_init,
        orthogonal_init=orthogonal_init,
        orthogonal_gain=orthogonal_gain
    )
    
    # Get critic initialization settings (can be different from policy)
    critic_orthogonal_init = args_cli.orthogonal_init if args_cli.orthogonal_init else critic_cfg.get("orthogonal_init", False)
    critic_orthogonal_gain = args_cli.orthogonal_gain if args_cli.orthogonal_init else critic_cfg.get("orthogonal_gain", 1.0)
    critic_use_layer_norm = critic_cfg.get("use_layer_norm", False)
    
    models["critic_1"] = Critic(env.observation_space, env.action_space, device,
                                orthogonal_init=critic_orthogonal_init,
                                orthogonal_gain=critic_orthogonal_gain,
                                use_layer_norm=critic_use_layer_norm)
    models["critic_2"] = Critic(env.observation_space, env.action_space, device,
                                orthogonal_init=critic_orthogonal_init,
                                orthogonal_gain=critic_orthogonal_gain,
                                use_layer_norm=critic_use_layer_norm)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, device,
                                       orthogonal_init=critic_orthogonal_init,
                                       orthogonal_gain=critic_orthogonal_gain,
                                       use_layer_norm=critic_use_layer_norm)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, device,
                                       orthogonal_init=critic_orthogonal_init,
                                       orthogonal_gain=critic_orthogonal_gain,
                                       use_layer_norm=critic_use_layer_norm)

    # Configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg.update(agent_cfg.get("agent", {}))
    
    # Update state preprocessor with environment-specific info
    if cfg.get("state_preprocessor") == "RunningStandardScaler":
        cfg["state_preprocessor"] = RunningStandardScaler
        cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    
    # Use the same experiment name and directory that was already set up
    # (already includes timestamp and initialization info)
    cfg["experiment"]["directory"] = agent_cfg["agent"]["experiment"]["directory"]
    cfg["experiment"]["experiment_name"] = agent_cfg["agent"]["experiment"]["experiment_name"]

    agent = SAC(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

    # Load BC checkpoint if provided (for warm-start from behavioral cloning)
    if args_cli.bc_checkpoint is not None:
        print(f"\n[INFO] Loading BC checkpoint: {args_cli.bc_checkpoint}")
        try:
            checkpoint = torch.load(args_cli.bc_checkpoint, map_location=device)
            
            # Load policy weights
            if 'model_state_dict' in checkpoint:
                # BC checkpoint format
                bc_state_dict = checkpoint['model_state_dict']
                
                # Map BC model weights to SAC policy network
                # BC model structure: net.0, net.2, net.4, ... (Linear layers with ELU in between)
                # SAC policy structure: net.0, net.1, net.2, ... (same structure)
                models["policy"].net.load_state_dict(bc_state_dict, strict=False)
                
                print(f"[INFO] BC policy weights loaded successfully")
                print(f"[INFO] BC training loss: {checkpoint.get('val_loss', 'N/A')}")
                print(f"[INFO] BC trained for {checkpoint.get('epoch', 'N/A')} epochs")
                
                if 'metadata' in checkpoint:
                    print(f"[INFO] BC dataset info:")
                    for key, val in checkpoint['metadata'].items():
                        print(f"  - {key}: {val}")
            else:
                # Standard checkpoint format
                models["policy"].load_state_dict(checkpoint, strict=False)
                print(f"[INFO] Policy weights loaded from checkpoint")
            
            print(f"[INFO] Starting RL finetuning from BC initialization...\n")
            
        except Exception as e:
            print(f"[WARNING] Failed to load BC checkpoint: {e}")
            print(f"[WARNING] Continuing with random initialization...")

    # Configure and instantiate the RL trainer
    cfg_trainer = agent_cfg.get("trainer", {})
    if args_cli.max_iterations:
        cfg_trainer["timesteps"] = args_cli.max_iterations
    cfg_trainer["headless"] = True
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # Start training
    print(f"[INFO] Starting SAC training for {cfg_trainer['timesteps']} timesteps")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Num envs: {args_cli.num_envs}")
    print(f"[INFO] Seed: {seed}")
    print(f"[INFO] Policy configuration:")
    print(f"  - Initial log std: {policy_cfg.get('initial_log_std', -4.0)}")
    print(f"  - Log std range: [{policy_cfg.get('min_log_std', -20)}, {policy_cfg.get('max_log_std', 2)}]")
    print(f"  - Zero init last layer: {zero_init}")
    print(f"  - Orthogonal initialization: {orthogonal_init}")
    if orthogonal_init:
        print(f"  - Orthogonal gain: {orthogonal_gain}")
    print(f"[INFO] Critic configuration:")
    print(f"  - Orthogonal initialization: {critic_orthogonal_init}")
    if critic_orthogonal_init:
        print(f"  - Orthogonal gain: {critic_orthogonal_gain}")
    print(f"  - LayerNorm: {critic_use_layer_norm} {'(RLPD)' if critic_use_layer_norm else ''}")
    print(f"[INFO] Training configuration:")
    print(f"  - Gradient steps (UTD ratio): {cfg.get('gradient_steps', 1)}")
    print(f"  - Batch size: {cfg.get('batch_size', 64)}")
    print(f"  - Learning starts: {cfg.get('learning_starts', 0)}")
    print(f"  - Random timesteps: {cfg.get('random_timesteps', 0)}")
    print(f"[INFO] State preprocessor: {cfg.get('state_preprocessor', 'None')}")
    
    trainer.train()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
