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
import datetime
import itertools
import math
import os
import random
from io import BytesIO
from typing import Annotated, Any, Iterable, Iterator, List, Literal, Optional, Union

# Third Party
import numpy as np
import pyglet
import trimesh
from trimesh.viewer import SceneViewer

# NVIDIA
from schedulestream.common.utils import create_seed, current_time, elapsed_time


def set_random_seed(seed=None, numpy_seed=None):
    if numpy_seed is None:
        numpy_seed = seed
    if seed is None:
        seed = numpy_seed
    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        np.random.seed(numpy_seed)


def set_seed(**kwargs: Any) -> int:
    seed = create_seed(**kwargs)
    set_random_seed(seed)
    print(f"Seed: {seed}")
    return seed


def inclusive_range(start: float, stop: float, step: float) -> np.ndarray:
    return np.append(np.arange(start, stop, step), stop)


def get_video_path(video: Optional[str], name: str = "video") -> Optional[str]:
    if video is None:
        return video
    if video == "":
        video = "mp4"
    root_dir = os.path.abspath(os.path.join(__file__, *itertools.repeat(os.pardir, times=5)))
    video_dir = os.path.join(root_dir, "videos")
    date_name = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    if video == "mp4":
        video = os.path.join(video_dir, f"{name}_{date_name}.mp4")
    elif video == "gif":
        video = os.path.join(video_dir, f"{name}_{date_name}.gif")
    return os.path.abspath(video)


def create_video(
    frames: Iterable[bytes], video_path: str = "video.mp4", frequency: float = 30.0
) -> Optional[str]:
    if not frames:
        return None
    # Third Party
    import imageio
    from PIL import Image

    video_path = os.path.abspath(video_path)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    kwargs = dict(loop=100) if video_path.endswith(".gif") else dict()
    video_writer = imageio.get_writer(video_path, fps=frequency, **kwargs)
    for frame in frames:
        image_pil = Image.open(BytesIO(frame))
        image_np = np.array(image_pil)
        video_writer.append_data(image_np)
    video_writer.close()
    return video_path


def save_frames(frames: List[bytes], video_path: str = "video.mp4", **kwargs: Any) -> Optional[str]:
    if not frames:
        return None
    start_time = current_time()
    video_path = os.path.abspath(video_path)
    print(f"Saving {len(frames)} frames to: {video_path}")
    video_path = create_video(frames, video_path=video_path, **kwargs)
    print(f"Saved {len(frames)} frames ({elapsed_time(start_time):.3f} sec): {video_path}")
    return video_path


Color = Annotated[np.ndarray, Literal[3]]


def to_rgba(red: float = 0.0, green: float = 0.0, blue: float = 0.0, alpha: float = 1.0) -> Color:
    return np.array([red, green, blue, alpha])


def to_uint8_color(float_color: Color) -> np.ndarray:
    return ((2**8 - 1) * np.array(float_color)).astype(np.uint8)


def rgb_from_hsv(hue, saturation: float = 1.0, value: float = 1.0) -> Color:
    return np.array(colorsys.hsv_to_rgb(hue, saturation, value))


def spaced_colors(num: int, hue1: float = 0.0, hue2: float = 1.0) -> List[Color]:
    return [rgb_from_hsv(hue) for hue in np.linspace(hue1, hue2, num, endpoint=False)]


def apply_alpha(color: Color, **kwargs: Any) -> np.ndarray:
    return to_rgba(*color[:3], **kwargs)


COLORS = {
    "black": to_rgba(red=0.0, green=0.0, blue=0.0),
    "grey": to_rgba(*0.5 * np.ones(3)),
    "white": to_rgba(red=1.0, green=1.0, blue=1.0),
    "red": to_rgba(red=1.0),
    "green": to_rgba(green=1.0),
    "blue": to_rgba(blue=1.0),
}


def get_color(color: Optional[Union[Color, str]]) -> Color:
    if not isinstance(color, str):
        return color
    assert color in COLORS, color
    return COLORS[color]


def is_category(name: str, categories: Optional[List[str]] = None) -> bool:
    if categories is None:
        return True
    return any(category.lower() in str(name).lower() for category in categories)


def animate_scene(
    scene: trimesh.Scene,
    iterator: Iterator[Any],
    height: Optional[int] = 480,
    frequency: float = 30.0,
    start: bool = True,
    record: bool = False,
    **kwargs: Any,
) -> List[bytes]:
    def callback(_):
        if callback.pause:
            return
        try:
            next(iterator)
        except StopIteration:
            pass

    callback.pause = not start

    fixed = None
    resolution = 4**2 / 3**2
    width = int(math.ceil(resolution * height))
    window = SceneViewer(
        scene=scene,
        callback=None if record else callback,
        callback_period=1.0 / frequency,
        start_loop=False,
        resolution=(width, height),
        fixed=fixed,
        visible=True,
        background=(200, 200, 225, 255),
        record=record,
        **kwargs,
    )

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            callback.pause = not callback.pause

    if record:
        for _ in iterator:
            pyglet.clock.tick()
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event("on_draw")
            window.flip()
        window.close()
    else:
        pyglet.app.run()
    return scene.metadata.get("recording", [])
