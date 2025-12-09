#!/usr/bin/env python3
"""
Script to parse curriculum evaluation results and generate plots with std bars.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set seaborn style
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Set monospace font (Courier-like)
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.monospace'] = ['DejaVu Sans Mono', 'Courier', 'monospace']


def parse_results_file(filepath):
    """
    Parse a single results txt file to extract curriculum data.

    Returns:
        list of dict: Each dict contains curriculum level info and success rates
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Split by curriculum level
    curriculum_blocks = re.split(r'\[INFO\] Curriculum expanded to level \d+', content)

    results = []

    # Find all curriculum level headers and their positions
    level_matches = list(re.finditer(r'\[INFO\] Curriculum expanded to level (\d+)', content))

    for i, level_match in enumerate(level_matches):
        level = int(level_match.group(1))
        start_pos = level_match.end()

        # Find end position (next level or end of file)
        end_pos = level_matches[i + 1].start() if i + 1 < len(level_matches) else len(content)
        block = content[start_pos:end_pos]

        # Extract object XY range
        xy_match = re.search(r'Object XY ranges: \[([-\d.]+), ([-\d.]+)\] x \[([-\d.]+), ([-\d.]+)\]', block)
        if not xy_match:
            continue

        x_min, x_max, y_min, y_max = map(float, xy_match.groups())
        # Use max absolute value as the randomization amount (in meters, convert to cm)
        randomization_cm = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)) * 100

        # Extract all intermediate success rates
        success_rates = []
        for match in re.finditer(r'Success Rate: ([\d.]+)%', block):
            success_rates.append(float(match.group(1)))

        # Extract weighted success rate (final for this level)
        weighted_match = re.search(r'Weighted success rate over last \d+ episodes[^:]*: ([\d.]+)', block)
        if weighted_match:
            weighted_success = float(weighted_match.group(1)) * 100  # Convert to percentage
        else:
            # If no weighted success, use mean of intermediate rates
            weighted_success = np.mean(success_rates) if success_rates else None

        if success_rates and weighted_success is not None:
            results.append({
                'level': level,
                'randomization_cm': (randomization_cm*2)**2,
                'success_rates': success_rates,
                'mean_success': np.mean(success_rates),
                'std_success': np.std(success_rates),
                'weighted_success': weighted_success,
            })

    return results


def plot_curriculum_results(results_dict, output_path=None):
    """
    Plot curriculum results with std bars.

    Args:
        results_dict: Dictionary mapping experiment name to list of curriculum results
        output_path: Optional path to save the plot
    """
    # Make plot 4:3 aspect ratio
    fig, ax = plt.subplots(figsize=(12, 9))

    # Define colors for specific experiments (hex colors)
    color_map = {
        'residual': '#3F784C',
        'base': '#F6CF95',
        'curr': '#3F784C',
        'no-curr': '#3F784C',
        'dense': '#3F784C',
        'sparse': '#3F784C',
    }

    line_style = {
        'residual': '-',
        'base': '-',
        'curr': '-',
        'no-curr': '--',
        'dense': '-',
        'sparse': '--',
    }

    # Track min and max values for y-axis
    all_min_values = []
    all_max_values = []

    for exp_name, results in results_dict.items():
        if not results:
            continue

        # Filter to only levels 0-10
        results = [r for r in results if r['level'] <= 10]

        # Extract data
        randomizations = [r['randomization_cm'] for r in results]
        mean_success = [r['mean_success'] for r in results]
        std_success = [r['std_success'] for r in results]

        # Track min/max for y-axis (including error bars)
        for mean, std in zip(mean_success, std_success):
            all_min_values.append(mean - std)
            all_max_values.append(mean + std)

        # Get color for this experiment (default to auto if not in map)
        color = color_map.get(exp_name, None)

        # Use dashed line for "no-curr" or "no curr"
        linestyle = line_style.get(exp_name, '-')

        # Plot line with error bars (thicker lines)
        ax.errorbar(
            randomizations,
            mean_success,
            yerr=std_success,
            marker='o',
            markersize=12,
            linewidth=4,
            capsize=8,
            capthick=3,
            label=exp_name,
            color=color,
            linestyle=linestyle,
            alpha=0.9
        )

    # Set y-axis limits dynamically based on data
    if all_min_values and all_max_values:
        y_min = min(all_min_values)
        y_max = max(all_max_values)
        # Add 5% padding
        y_range = y_max - y_min
        y_min = max(0, y_min - 0.05 * y_range)  # Don't go below 0
        y_max = min(100, y_max + 0.05 * y_range)  # Don't go above 100
        ax.set_ylim([y_min, y_max])

    # Formatting with larger fonts
    ax.set_xlabel('Object Randomization (cm^2)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=24, fontweight='bold')
    ax.set_title('Effect of Curriculum', fontsize=28, fontweight='bold')

    # Create custom legend handles to show just lines (no markers/error bars)
    from matplotlib.lines import Line2D
    legend_handles = []
    for exp_name in results_dict.keys():
        color = color_map.get(exp_name, 'black')
        linestyle = line_style.get(exp_name, '-')
        handle = Line2D([0], [0], color=color, linewidth=4, linestyle=linestyle,
                       label=exp_name)
        legend_handles.append(handle)

    ax.legend(handles=legend_handles, fontsize=20, loc='best')
    ax.grid(True, alpha=0.3)

    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Plot saved to: {output_path}")

    plt.show()


def main():
    """Main function to parse all results and generate plot."""
    # Find all txt files in results directory
    results_dir = Path('./curriculum')

    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return

    txt_files = list(results_dir.glob('*.txt'))

    if not txt_files:
        print(f"[ERROR] No txt files found in: {results_dir}")
        return

    print(f"[INFO] Found {len(txt_files)} result file(s)")

    # Parse all files
    all_results = {}
    for txt_file in txt_files:
        exp_name = txt_file.stem  # Filename without extension
        print(f"[INFO] Parsing: {txt_file.name}")

        results = parse_results_file(txt_file)

        if results:
            all_results[exp_name] = results
            print(f"  - Extracted {len(results)} curriculum levels")
            print(f"  - Randomization range: {results[0]['randomization_cm']:.1f} - {results[-1]['randomization_cm']:.1f} cm")
        else:
            print(f"  - No valid data found")

    if not all_results:
        print("[ERROR] No valid results parsed from any file")
        return

    # Print summary statistics
    print("\n[INFO] Summary Statistics:")
    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        for r in results:
            print(f"  Level {r['level']:2d} ({r['randomization_cm']:4.1f} cm): "
                  f"Success = {r['mean_success']:5.1f}% ± {r['std_success']:4.1f}% "
                  f"(n={len(r['success_rates'])} resets)")

    # Generate plot
    output_path = results_dir / 'plot.png'
    plot_curriculum_results(all_results, output_path=output_path)


if __name__ == "__main__":
    main()
