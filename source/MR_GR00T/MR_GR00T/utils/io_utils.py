# Copied from https://github.com/isaac-sim/IsaacLabEvalTasks/blob/main/scripts/io_utils.py

import collections
import json
import numpy as np
import yaml
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2


def dump_jsonl(data, file_path):
    """
    Write a sequence of data to a file in JSON Lines format.

    Args:
        data: Sequence of items to write, one per line.
        file_path: Path to the output file.

    Returns:
        None
    """
    assert isinstance(data, collections.abc.Sequence) and not isinstance(data, str)
    if isinstance(data, (np.ndarray, np.number)):
        data = data.tolist()
    with open(file_path, "w") as fp:
        for line in data:
            print(json.dumps(line), file=fp, flush=True)


def dump_json(data, file_path, **kwargs):
    """
    Write data to a file in standard JSON format.

    Args:
        data: Data to write to the file.
        file_path: Path to the output file.
        **kwargs: Additional keyword arguments for json.dump.

    Returns:
        None
    """
    if isinstance(data, (np.ndarray, np.number)):
        data = data.tolist()
    with open(file_path, "w") as fp:
        json.dump(data, fp, **kwargs)


def load_json(file_path: str | Path, **kwargs) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        file_path: Path to the JSON file to load.
        **kwargs: Additional keyword arguments for the JSON loader.

    Returns:
        Dictionary loaded from the JSON file.
    """
    with open(file_path) as fp:
        return json.load(fp, **kwargs)


def load_gr1_joints_config(yaml_path: str | Path) -> Dict[str, Any]:
    """Load GR1 joint configuration from YAML file"""
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("joints", {})


class VideoWriter:
    """
    A class for writing videos from images.
    """

    def __init__(self, out_video_path: str, video_size: Tuple, fps: int = 20):
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.fps = fps
        self.video_size = video_size
        self.writer = cv2.VideoWriter(out_video_path, self.fourcc, fps, video_size)
        print(f"Writing video to: {out_video_path}")

    def add_image(self, img):
        # Permute to (BGR) and resize
        img_bgr = img.squeeze()[:, :, [2, 1, 0]]
        resized_img = cv2.resize(img_bgr.cpu().numpy(), self.video_size)
        self.writer.write(resized_img)

    def change_file_path(self, out_video_path: str):
        self.writer.release()
        self.writer = cv2.VideoWriter(out_video_path, self.fourcc, self.fps, self.video_size)
        print(f"Writing video to: {out_video_path}")

    def close(self):
        self.writer.release()