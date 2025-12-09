#!/usr/bin/env python3
"""
Script to parse and plot Wasserstein distances per action dimension for base vs residual policies.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set seaborn style
sns.set_theme(style="whitegrid")

# Set monospace font (Courier-like)
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.monospace'] = ['DejaVu Sans Mono', 'Courier', 'monospace']


def load_action_stats(filepath):
    """Load action statistics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_wasserstein_per_dim(action_stats, level=1):
    """Extract Wasserstein distances per dimension for a specific curriculum level."""
    level_key = str(level)
    if level_key in action_stats:
        return action_stats[level_key].get('wasserstein_per_dim', [])
    return []


def plot_wasserstein_comparison(base_wd, residual_wd, output_path=None):
    """
    Plot Wasserstein distances per dimension as horizontal bar graphs.

    Args:
        base_wd: List of Wasserstein distances for base policy
        residual_wd: List of Wasserstein distances for residual policy
        output_path: Optional path to save the plot
    """
    # Make plot 4:3 aspect ratio
    fig, ax = plt.subplots(figsize=(12, 9))

    # Define colors (matching plot_curriculum_results)
    base_color = '#F6CF95'
    residual_color = '#3F784C'

    # Number of dimensions (joints)
    n_dims = len(base_wd)

    # Create joint labels
    joint_labels = [f'joint_{i}' for i in range(n_dims)]

    # Set up bar positions
    y_positions = np.arange(n_dims)
    bar_height = 0.35

    # Plot horizontal bars
    bars_base = ax.barh(y_positions - bar_height/2, base_wd, bar_height,
                        label='base', color=base_color, alpha=0.9, edgecolor='black', linewidth=1.5)
    bars_residual = ax.barh(y_positions + bar_height/2, residual_wd, bar_height,
                            label='residual', color=residual_color, alpha=0.9, edgecolor='black', linewidth=1.5)

    # Formatting with larger fonts
    ax.set_xlabel('Wasserstein Distance', fontsize=24, fontweight='bold')
    ax.set_title('Wasserstein Distance per Joint', fontsize=28, fontweight='bold')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(joint_labels)
    ax.legend(fontsize=20, loc='best')
    ax.grid(True, alpha=0.3, axis='x')

    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Invert y-axis so joint_0 is at the top
    ax.invert_yaxis()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Plot saved to: {output_path}")

    plt.show()


def main():
    """Main function to parse action stats and generate comparison plot."""
    # File paths
    base_stats_path = Path('base_action_stats.json')
    residual_stats_path = Path('res_action_stats.json')

    # Check if files exist
    if not base_stats_path.exists():
        print(f"[ERROR] Base action stats not found: {base_stats_path}")
        return

    if not residual_stats_path.exists():
        print(f"[ERROR] Residual action stats not found: {residual_stats_path}")
        return

    print(f"[INFO] Loading action statistics...")

    # Load action statistics
    base_stats = load_action_stats(base_stats_path)
    residual_stats = load_action_stats(residual_stats_path)

    # Extract Wasserstein distances per dimension (using level 1)
    base_wd = extract_wasserstein_per_dim(base_stats, level=1)
    residual_wd = extract_wasserstein_per_dim(residual_stats, level=1)

    if not base_wd or not residual_wd:
        print(f"[ERROR] No Wasserstein distance data found in action stats")
        return

    if len(base_wd) != len(residual_wd):
        print(f"[WARNING] Dimension mismatch: base has {len(base_wd)}, residual has {len(residual_wd)}")
        # Truncate to minimum length
        min_len = min(len(base_wd), len(residual_wd))
        base_wd = base_wd[:min_len]
        residual_wd = residual_wd[:min_len]

    print(f"[INFO] Comparing Wasserstein distances across {len(base_wd)} dimensions")
    print(f"[INFO] Base policy average WD: {np.mean(base_wd):.4f}")
    print(f"[INFO] Residual policy average WD: {np.mean(residual_wd):.4f}")

    # Print per-dimension comparison
    print(f"\n[INFO] Per-dimension comparison:")
    for i, (base_val, res_val) in enumerate(zip(base_wd, residual_wd)):
        diff = res_val - base_val
        sign = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"  joint_{i}: base={base_val:.4f}, residual={res_val:.4f}, diff={diff:+.4f} {sign}")

    # Generate plot
    output_path = Path('wasserstein_comparison.png')
    plot_wasserstein_comparison(base_wd, residual_wd, output_path=output_path)


if __name__ == "__main__":
    main()
