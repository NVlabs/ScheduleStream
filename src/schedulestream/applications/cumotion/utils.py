#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
import colorsys
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
from urllib.parse import urlparse

# Third Party
import numpy as np
from _cumotion import Pose3, Rotation3

Vector = Union[np.ndarray, List[float]]


def get_cumotion_dir() -> str:
    dist = importlib.metadata.distribution("cumotion")
    direct_url = json.loads(dist.read_text("direct_url.json"))
    wheel_path = Path(urlparse(direct_url["url"]).path)
    return str(wheel_path.parent.parent)


def get_config_dir() -> str:
    return os.path.join(get_cumotion_dir(), "content", "nvidia", "shared")


def get_third_party_dir() -> str:
    return os.path.join(get_cumotion_dir(), "content", "third_party")


BLACK = (0.0, 0.0, 0.0)
GREY = (0.5, 0.5, 0.5)
WHITE = (1.0, 1.0, 1.0)
RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)
Color = Tuple[float, float, float]


def rgb_from_hsv(hue: float, saturation: float = 1.0, value: float = 1.0) -> Color:
    return colorsys.hsv_to_rgb(hue, saturation, value)


def hsv_colors(num: int, hue1: float = 0.0, hue2: float = 1.0, **kwargs: Any) -> List[Color]:
    return [rgb_from_hsv(hue, **kwargs) for hue in np.linspace(hue1, hue2, num, endpoint=False)]


def create_pose(position: Optional[Vector] = None, orientation: Optional[Vector] = None) -> Pose3:
    if position is None:
        position = np.zeros(3)
    assert len(position) == 3
    if orientation is None:
        orientation = np.zeros(3)
    assert len(orientation) == 3
    rotation = Rotation3.from_scaled_axis(orientation)
    return Pose3(rotation, position)
