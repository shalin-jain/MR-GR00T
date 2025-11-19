"""Curriculum manager for progressively expanding spawn distribution."""

from __future__ import annotations

import torch
from dataclasses import dataclass


class SpawnCurriculumManager:
    """
    Manages curriculum learning by progressively expanding object and bin spawn distributions.

    The manager tracks success rate and expands the spawn range for both object and bin
    when performance exceeds a threshold.
    """

    def __init__(self, cfg, device: torch.device):
        """
        Initialize the curriculum manager.

        Args:
            cfg: Configuration for curriculum parameters
            device: Device to create tensors on
        """
        self.cfg = cfg
        self.device = device

        # Current spawn ranges (start at initial values)
        self.object_x_min = cfg.object_x_min_start
        self.object_x_max = cfg.object_x_max_start
        self.object_y_min = cfg.object_y_min_start
        self.object_y_max = cfg.object_y_max_start

        self.bin_x_min = cfg.bin_x_min_start
        self.bin_x_max = cfg.bin_x_max_start
        self.bin_y_min = cfg.bin_y_min_start
        self.bin_y_max = cfg.bin_y_max_start

        # Tracking metrics
        self.reset_count = 0  # Total resets across all curriculum levels
        self.resets_since_last_update = 0  # Resets since last curriculum check
        self.success_history = []
        self.num_envs_history = []  # Track number of envs in each reset
        self.curriculum_level = 0
        self.weighted_success_rate = 0.0  # Cache the last computed weighted success rate

        # Store previous ranges for frontier sampling
        self.prev_object_x_min = cfg.object_x_min_start
        self.prev_object_x_max = cfg.object_x_max_start
        self.prev_object_y_min = cfg.object_y_min_start
        self.prev_object_y_max = cfg.object_y_max_start
        self.prev_bin_x_min = cfg.bin_x_min_start
        self.prev_bin_x_max = cfg.bin_x_max_start
        self.prev_bin_y_min = cfg.bin_y_min_start
        self.prev_bin_y_max = cfg.bin_y_max_start

    def update(self, success_rate: float, num_envs: int) -> bool:
        """
        Update curriculum based on recent performance.

        Args:
            success_rate: Recent success rate (0-1)
            num_envs: Number of environments that were reset

        Returns:
            True if curriculum was expanded, False otherwise
        """
        self.reset_count += num_envs  # Track total number of environment resets (episodes)

        # Check if we're still in warmup period
        if self.reset_count <= self.cfg.warmup_resets:
            return False

        self.resets_since_last_update += num_envs  # Track resets since last update
        self.success_history.append(success_rate)
        self.num_envs_history.append(num_envs)

        # Check if we've accumulated enough episodes since last update
        if self.resets_since_last_update < self.cfg.update_frequency:
            return False

        # Calculate weighted average success rate over the last update_frequency episodes
        # Work backwards through history to get exactly update_frequency episodes
        total_episodes = 0
        episodes_to_use = []
        successes_to_use = []

        for i in range(len(self.num_envs_history) - 1, -1, -1):
            if total_episodes >= self.cfg.update_frequency:
                break
            episodes_to_use.insert(0, self.num_envs_history[i])
            successes_to_use.insert(0, self.success_history[i])
            total_episodes += self.num_envs_history[i]

        # Weighted average: each success_rate is weighted by how many envs it represents
        if total_episodes == 0:
            return False

        self.weighted_success_rate = sum(s * n for s, n in zip(successes_to_use, episodes_to_use)) / total_episodes
        print(f"[CURRICULUM]: Weighted success rate over last {total_episodes} episodes ({len(episodes_to_use)} resets): {self.weighted_success_rate:.3f}")

        # Expand if performance is good and we haven't reached final ranges
        if self.weighted_success_rate >= self.cfg.success_threshold and not self._at_final_ranges():
            self._expand_ranges()
            self.curriculum_level += 1

            # Clear history buffers after expansion so previous level performance doesn't affect next level
            self.success_history.clear()
            self.num_envs_history.clear()

            # Reset the counter for next level
            self.resets_since_last_update = 0

            return True

        return False

    def sample_object_position(self, base_pos: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Sample random object positions within current curriculum ranges.
        Uses 50/50 mix of frontier (new region) and uniform (full range) sampling.

        Args:
            base_pos: Base position to offset from [x, y, z]
            num_samples: Number of positions to sample

        Returns:
            Sampled positions of shape (num_samples, 3)
        """
        # Split samples: 50% from frontier, 50% from full range
        num_frontier = num_samples // 2
        num_uniform = num_samples - num_frontier

        # Sample uniformly from full current range
        x_uniform = torch.rand(num_uniform, device=self.device) * (self.object_x_max - self.object_x_min) + self.object_x_min
        y_uniform = torch.rand(num_uniform, device=self.device) * (self.object_y_max - self.object_y_min) + self.object_y_min

        # Sample from frontier (new expanded region)
        # Frontier is the difference between current and previous range
        x_frontier = self._sample_frontier_1d(
            num_frontier,
            self.object_x_min, self.object_x_max,
            self.prev_object_x_min, self.prev_object_x_max
        )
        y_frontier = self._sample_frontier_1d(
            num_frontier,
            self.object_y_min, self.object_y_max,
            self.prev_object_y_min, self.prev_object_y_max
        )

        # Combine samples
        x_offset = torch.cat([x_uniform, x_frontier])
        y_offset = torch.cat([y_uniform, y_frontier])
        z_offset = torch.zeros(num_samples, device=self.device)

        offsets = torch.stack([x_offset, y_offset, z_offset], dim=1)
        return base_pos.unsqueeze(0) + offsets

    def sample_bin_position(self, base_pos: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Sample random bin positions within current curriculum ranges.
        Uses 50/50 mix of frontier (new region) and uniform (full range) sampling.

        Args:
            base_pos: Base position to offset from [x, y, z]
            num_samples: Number of positions to sample

        Returns:
            Sampled positions of shape (num_samples, 3)
        """
        # Split samples: 50% from frontier, 50% from full range
        num_frontier = num_samples // 2
        num_uniform = num_samples - num_frontier

        # Sample uniformly from full current range
        x_uniform = torch.rand(num_uniform, device=self.device) * (self.bin_x_max - self.bin_x_min) + self.bin_x_min
        y_uniform = torch.rand(num_uniform, device=self.device) * (self.bin_y_max - self.bin_y_min) + self.bin_y_min

        # Sample from frontier (new expanded region)
        x_frontier = self._sample_frontier_1d(
            num_frontier,
            self.bin_x_min, self.bin_x_max,
            self.prev_bin_x_min, self.prev_bin_x_max
        )
        y_frontier = self._sample_frontier_1d(
            num_frontier,
            self.bin_y_min, self.bin_y_max,
            self.prev_bin_y_min, self.prev_bin_y_max
        )

        # Combine samples
        x_offset = torch.cat([x_uniform, x_frontier])
        y_offset = torch.cat([y_uniform, y_frontier])
        z_offset = torch.zeros(num_samples, device=self.device)

        offsets = torch.stack([x_offset, y_offset, z_offset], dim=1)
        return base_pos.unsqueeze(0) + offsets

    def _sample_frontier_1d(self, num_samples: int, curr_min: float, curr_max: float,
                           prev_min: float, prev_max: float) -> torch.Tensor:
        """
        Sample from the frontier region (new area outside previous range).

        Args:
            num_samples: Number of samples to generate
            curr_min, curr_max: Current range bounds
            prev_min, prev_max: Previous range bounds

        Returns:
            Samples from frontier region
        """
        # If ranges are the same (first level or no expansion), sample uniformly
        if abs(curr_min - prev_min) < 1e-6 and abs(curr_max - prev_max) < 1e-6:
            return torch.rand(num_samples, device=self.device) * (curr_max - curr_min) + curr_min

        # Calculate frontier regions: [curr_min, prev_min] and [prev_max, curr_max]
        left_frontier_size = max(0, prev_min - curr_min)
        right_frontier_size = max(0, curr_max - prev_max)
        total_frontier_size = left_frontier_size + right_frontier_size

        # If no frontier (shouldn't happen), fall back to uniform
        if total_frontier_size < 1e-6:
            return torch.rand(num_samples, device=self.device) * (curr_max - curr_min) + curr_min

        # Sample proportionally from left and right frontiers
        samples = []
        for _ in range(num_samples):
            if torch.rand(1, device=self.device).item() < (left_frontier_size / total_frontier_size):
                # Sample from left frontier [curr_min, prev_min]
                samples.append(torch.rand(1, device=self.device) * left_frontier_size + curr_min)
            else:
                # Sample from right frontier [prev_max, curr_max]
                samples.append(torch.rand(1, device=self.device) * right_frontier_size + prev_max)

        return torch.cat(samples)

    def get_info(self) -> dict:
        """
        Get current curriculum information for logging.

        Returns:
            Dictionary with curriculum state information
        """
        return {
            "curriculum/level": torch.tensor(self.curriculum_level, device=self.device),
            "curriculum/weighted_success_rate": torch.tensor(self.weighted_success_rate * 100.0, device=self.device),  # as percentage
            "curriculum/object_x_range": torch.tensor(self.object_x_max - self.object_x_min, device=self.device),
            "curriculum/object_y_range": torch.tensor(self.object_y_max - self.object_y_min, device=self.device),
            "curriculum/bin_x_range": torch.tensor(self.bin_x_max - self.bin_x_min, device=self.device),
            "curriculum/bin_y_range": torch.tensor(self.bin_y_max - self.bin_y_min, device=self.device),
            "curriculum/reset_count": torch.tensor(self.reset_count, device=self.device),
        }

    def _expand_ranges(self):
        """Expand spawn ranges by configured expansion rate."""
        # Store previous ranges before expanding
        self.prev_object_x_min = self.object_x_min
        self.prev_object_x_max = self.object_x_max
        self.prev_object_y_min = self.object_y_min
        self.prev_object_y_max = self.object_y_max
        self.prev_bin_x_min = self.bin_x_min
        self.prev_bin_x_max = self.bin_x_max
        self.prev_bin_y_min = self.bin_y_min
        self.prev_bin_y_max = self.bin_y_max

        # Calculate expansion amounts
        obj_x_expand = (self.cfg.object_x_max_final - self.cfg.object_x_min_final) * self.cfg.expansion_rate
        obj_y_expand = (self.cfg.object_y_max_final - self.cfg.object_y_min_final) * self.cfg.expansion_rate
        bin_x_expand = (self.cfg.bin_x_max_final - self.cfg.bin_x_min_final) * self.cfg.expansion_rate
        bin_y_expand = (self.cfg.bin_y_max_final - self.cfg.bin_y_min_final) * self.cfg.expansion_rate

        # Expand object ranges (clamped to final values)
        self.object_x_min = max(self.object_x_min - obj_x_expand / 2, self.cfg.object_x_min_final)
        self.object_x_max = min(self.object_x_max + obj_x_expand / 2, self.cfg.object_x_max_final)
        self.object_y_min = max(self.object_y_min - obj_y_expand / 2, self.cfg.object_y_min_final)
        self.object_y_max = min(self.object_y_max + obj_y_expand / 2, self.cfg.object_y_max_final)

        # Expand bin ranges (clamped to final values)
        self.bin_x_min = max(self.bin_x_min - bin_x_expand / 2, self.cfg.bin_x_min_final)
        self.bin_x_max = min(self.bin_x_max + bin_x_expand / 2, self.cfg.bin_x_max_final)
        self.bin_y_min = max(self.bin_y_min - bin_y_expand / 2, self.cfg.bin_y_min_final)
        self.bin_y_max = min(self.bin_y_max + bin_y_expand / 2, self.cfg.bin_y_max_final)

        print(f"[CURRICULUM]: Expanded ranges - Object X: [{self.object_x_min:.3f}, {self.object_x_max:.3f}], "
              f"Object Y: [{self.object_y_min:.3f}, {self.object_y_max:.3f}], "
              f"Bin X: [{self.bin_x_min:.3f}, {self.bin_x_max:.3f}], "
              f"Bin Y: [{self.bin_y_min:.3f}, {self.bin_y_max:.3f}]")

    def _at_final_ranges(self) -> bool:
        """Check if curriculum has reached final spawn ranges."""
        obj_x_final = (abs(self.object_x_min - self.cfg.object_x_min_final) < 1e-4 and
                       abs(self.object_x_max - self.cfg.object_x_max_final) < 1e-4)
        obj_y_final = (abs(self.object_y_min - self.cfg.object_y_min_final) < 1e-4 and
                       abs(self.object_y_max - self.cfg.object_y_max_final) < 1e-4)
        bin_x_final = (abs(self.bin_x_min - self.cfg.bin_x_min_final) < 1e-4 and
                       abs(self.bin_x_max - self.cfg.bin_x_max_final) < 1e-4)
        bin_y_final = (abs(self.bin_y_min - self.cfg.bin_y_min_final) < 1e-4 and
                       abs(self.bin_y_max - self.cfg.bin_y_max_final) < 1e-4)

        return obj_x_final and obj_y_final and bin_x_final and bin_y_final
