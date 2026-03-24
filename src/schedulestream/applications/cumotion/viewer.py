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
from __future__ import annotations

# Standard Library
import os.path
import time
from itertools import count
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, Union

# Third Party
import numpy as np
from cumotion import Pose3
from cumotion_vis.visualizer import CollisionSphereVisualization, RenderableType, Visualizer

# NVIDIA
from schedulestream.applications.cumotion.objects import Object
from schedulestream.applications.cumotion.utils import GREEN, Color, create_pose, rgb_from_hsv

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.applications.cumotion.command import Attachment
    from schedulestream.applications.cumotion.world import World


def add_visualizer_frame(
    visualizer: Visualizer, name: str, pose: Optional[Pose3] = None, size: float = 1e-1
) -> None:
    if pose is None:
        pose = Pose3.identity()
    config = {"size": size, "position": np.zeros(3)}
    if name not in visualizer._renderables:
        visualizer.add(RenderableType.COORDINATE_FRAME, name, config)
    visualizer.set_pose(name, pose)


def add_visualizer_sphere(
    visualizer: Visualizer,
    name: str,
    radius: float,
    color: Color = GREEN,
    pose: Optional[Pose3] = None,
):
    if pose is None:
        pose = Pose3.identity()
    config = {
        "position": np.zeros(3),
        "radius": radius,
        "color": color,
    }
    if name not in visualizer._renderables:
        visualizer.add(RenderableType.MARKER, name, config)
    visualizer.set_pose(name, pose)


class WorldViewer:
    def __init__(
        self,
        world: World,
        robot_spheres: bool = True,
        object_spheres: bool = True,
        time_step: float = 1 / 30,
        video_path: Optional[str] = None,
    ):
        self.world = world
        self.visualizer = Visualizer()
        add_visualizer_frame(self.visualizer, "origin")
        self.visualizer.set_background_color(rgb_from_hsv(hue=2 / 3, saturation=0.25, value=1.0))

        self.robot_visualization = self.world.robot_visualization(self.visualizer)
        if robot_spheres:
            self.collision_sphere_visualization = CollisionSphereVisualization(
                self.world.robot_description,
                self.visualizer,
                world_collision_spheres_color=GREEN,
                self_collision_spheres_color=[0.6, 0.0, 0.4],
                show_world_collision_spheres=True,
                show_self_collision_spheres=False,
            )
        else:
            self.collision_sphere_visualization = None
        self.object_spheres = object_spheres
        self.sync()
        self.set_view()
        self.time_step = time_step
        self.step = 0
        self.attachments = {}

        self.video_path = video_path
        self.video_writer = None
        if self.video_path is not None:
            # Third Party
            import imageio

            frequency = 1 / self.time_step
            kwargs = dict(loop=100) if video_path.endswith(".gif") else dict()
            self.video_writer = imageio.get_writer(video_path, fps=frequency, **kwargs)

    def set_view(self) -> None:
        control = self.visualizer._visualizer.get_view_control()
        camera_position = np.array([1.0, 0.0, 0.5])
        target_position = np.array([0.0, 0.0, 0.0])
        delta_position = camera_position - target_position
        control.set_up((0, 0, 1))
        control.set_front(delta_position)
        control.set_lookat(target_position)

    def get_configuration(self) -> np.ndarray:
        return self.robot_visualization.get_joint_positions()

    def set_configuration(self, configuration: np.ndarray) -> None:
        self.robot_visualization.set_joint_positions(configuration)
        if self.collision_sphere_visualization is not None:
            self.collision_sphere_visualization.set_joint_positions(configuration)

        for frame in self.world.tool_frames:
            pose = self.world.get_frame_pose(frame, configuration=configuration)
            add_visualizer_frame(self.visualizer, frame, pose=pose)

    def set_object_pose(self, obj: Object, pose: Pose3) -> None:
        if obj.name not in self.visualizer._renderables:
            self.visualizer.add(obj.visualizer_type, obj.name, obj.visualizer_config)
        self.visualizer.set_pose(obj.name, pose)

        if self.object_spheres and obj.movable:
            spheres = obj.spheres.transform(pose)
            for i in range(spheres.num):
                sphere_pose = create_pose(spheres.centers[i])
                add_visualizer_sphere(
                    self.visualizer,
                    name=f"{obj.name}_sphere{i}",
                    radius=spheres.radii[i],
                    color=obj.color,
                    pose=sphere_pose,
                )

    def sync(self):
        self.set_configuration(self.world.configuration)
        for obj in self.world.objects:
            pose = self.world.get_object_pose(obj)
            self.set_object_pose(obj, pose)

    def get_pose(self, frame: Union[str, Object]) -> Pose3:
        if isinstance(frame, Object):
            frame = frame.name
        if frame in self.visualizer._renderables:
            return self.visualizer._renderables[frame].pose
        return self.world.get_frame_pose(frame, configuration=self.get_configuration())

    def attach(self, obj: Object, frame: str) -> Attachment:
        # NVIDIA
        from schedulestream.applications.cumotion.command import Attachment

        obj_pose = self.get_pose(obj)
        frame_pose = self.get_pose(frame)
        relative_pose = frame_pose.inverse() * obj_pose
        attachment = Attachment(obj, frame, relative_pose)
        self.attachments[obj] = attachment
        return attachment

    def detach(self, obj: Object) -> Optional[Attachment]:
        return self.attachments.pop(obj, None)

    def sync_attachments(self) -> None:
        for attachment in self.attachments.values():
            frame_pose = self.get_pose(attachment.frame)
            obj_pose = frame_pose * attachment.pose
            self.set_object_pose(attachment.obj, obj_pose)

    def render(self, time_step: float = 1 / 30) -> bool:
        if not self.visualizer.is_active():
            return False
        self.sync_attachments()
        self.visualizer.update()

        if self.video_writer is not None:
            # Third Party
            from PIL import Image

            image_path = "/tmp/image.png"
            self.visualizer._visualizer.capture_screen_image(image_path)
            image_pil = Image.open(image_path)
            image_np = np.array(image_pil)
            self.video_writer.append_data(image_np)

        if time_step is not None:
            time.sleep(time_step)
        self.step += 1
        return True

    def close(self) -> None:
        self.visualizer.close()
        if self.video_writer is not None:
            self.video_writer.close()
            print(f"Saved: {os.path.abspath(self.video_path)}")

    def animate(self, generator: Iterable[Any], time_step: float = 1 / 30) -> None:
        for _ in generator:
            if not self.render():
                break
        self.close()

    def show(self, **kwargs: Any) -> None:
        return self.animate(count(), **kwargs)

    def loop(self, **kwargs: Any) -> Iterator[None]:
        while self.render(**kwargs):
            yield
        self.close()
