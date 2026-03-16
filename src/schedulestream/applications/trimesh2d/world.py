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
from functools import cache
from typing import Any, Dict, FrozenSet, List, Optional

# Third Party
import trimesh

# NVIDIA
from schedulestream.applications.trimesh2d.geometry import (
    Attachment,
    Body,
    Pose,
    Position2,
    conf_from_pose,
    invert_pose,
    multiply_poses,
    pose_from_conf,
)
from schedulestream.applications.trimesh2d.utils import animate_scene, is_category


class World:
    def __init__(
        self,
        collision_scene: Optional[trimesh.Scene] = None,
        visual_scene: Optional[trimesh.Scene] = None,
    ):
        if collision_scene is None:
            collision_scene = trimesh.Scene()
        if visual_scene is None:
            visual_scene = trimesh.Scene()
        self.scene = trimesh.scene.scene.append_scenes([collision_scene, visual_scene])
        self.manager, _ = trimesh.collision.scene_to_collision(collision_scene)
        self.attachments = {}

    @property
    def graph(self):
        return self.scene.graph

    @property
    def base_frame(self) -> str:
        return self.graph.base_frame

    @property
    def frames(self) -> List[str]:
        return self.graph.nodes

    @property
    def collision_frames(self) -> List[str]:
        return list(self.manager._objs.keys())

    @property
    def object_names(self) -> List[str]:
        return self.collision_frames

    def get_category_names(self, categories: List[str]) -> List[str]:
        return [frame for frame in self.frames if is_category(frame, categories)]

    def current_poses(self) -> Dict[str, Pose]:
        self.propagate()
        return {frame: self.get_pose(frame) for frame in self.frames}

    def current_state(self) -> "State":
        return State(self, self.current_poses())

    def get_geometry(self, frame: str) -> Optional[trimesh.Trimesh]:
        frame = str(frame)
        _, geometry = self.graph.get(frame_to=frame)
        if geometry is None:
            return geometry
        return self.scene.geometry[geometry]

    @property
    def geometries(self) -> List[trimesh.Trimesh]:
        geometries = list(map(self.get_geometry, self.frames))
        return list(filter(lambda g: g is not None, geometries))

    @property
    def bodies(self) -> List[Body]:
        return [geometry for geometry in self.geometries if isinstance(geometry, Body)]

    @property
    def movable_bodies(self) -> List[Body]:
        return [body for body in self.bodies if body.movable]

    @property
    def fixed_bodies(self) -> List[Body]:
        return [body for body in self.bodies if not body.movable]

    def get_object(self, name: str) -> trimesh.Trimesh:
        if not isinstance(name, str):
            return name
        return self.get_geometry(name)

    def get_pose(self, frame: str) -> Pose:
        frame = str(frame)
        pose, _ = self.graph.get(frame_to=frame)
        return pose

    def set_pose(self, frame: str, pose: Pose) -> None:
        frame = str(frame)
        self.graph.update(frame_to=frame, matrix=pose)
        if frame in self.collision_frames:
            self.manager.set_transform(frame, pose)

    def attach(self, child: str, parent: Optional[str] = None) -> Attachment:
        child_pose = self.get_pose(child)
        parent_pose = self.get_pose(parent)
        parent_from_child = multiply_poses(invert_pose(parent_pose), child_pose)
        attachment = Attachment(parent, parent_from_child)
        self.attachments[child] = attachment
        return attachment

    def detach(self, child: str) -> Attachment:
        return self.attachments.pop(child)

    def propagate(self) -> None:
        for child, attachment in self.attachments.items():
            pose = attachment.get(self)
            self.set_pose(child, pose)

    def current_geometry(self, frame: str) -> Optional[trimesh.Trimesh]:
        geometry = self.get_geometry(frame)
        if geometry is None:
            return geometry
        geometry = geometry.copy()
        geometry.apply_transform(self.get_pose(frame))
        return geometry

    def current_box(self, frame: str) -> Optional[trimesh.primitives.Box]:
        geometry = self.get_geometry(frame)
        if geometry is None:
            return geometry
        box = geometry.bounding_box.copy()
        box.apply_transform(self.get_pose(frame))
        return box

    def current_box_scene(self) -> trimesh.Scene:
        scene = trimesh.Scene()
        for frame in self.frames:
            scene.add_geometry(self.get_box(frame))
        return scene

    def check_collisions(self) -> List[FrozenSet[str]]:
        _, collisions = self.manager.in_collision_internal(return_names=True, return_data=False)
        return list(map(frozenset, collisions))

    def check_colliding(self, frame: str) -> List[str]:
        frame = str(frame)
        collisions = self.check_collisions()
        return [list(pair - {frame})[0] for pair in collisions if frame in pair]

    def check_pair(self, frame1: str, frame2: str) -> bool:
        pair = frozenset([str(frame1), str(frame2)])
        return pair in self.check_collisions()

    def __str__(self):
        return f"{self.__class__.__name__}({self.bodies})"

    __repr__ = __str__

    def show(self, **kwargs: Any) -> None:
        self.scene.show(**kwargs)

    def animate(
        self,
        states: List["State"],
        **kwargs: Any,
    ) -> List[bytes]:
        iterator = (state.set() for state in states)
        return animate_scene(self.scene, iterator, **kwargs)


class World2D(World):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def current_state(self) -> "State2D":
        return State2D(self, self.current_poses())

    def get_conf(self, frame: str) -> Position2:
        return conf_from_pose(self.get_pose(frame))

    def set_conf(self, frame: str, conf: Position2) -> None:
        self.set_pose(frame, pose_from_conf(conf))


class State:
    def __init__(self, world: World, poses: Optional[Dict[str, Pose]] = None):
        if poses is None:
            poses = {}
        self.world = world
        self.poses = dict(poses)

    @property
    def frames(self):
        return list(self.poses)

    def get_pose(self, frame: str) -> Pose:
        return self.poses[frame]

    def set(self, frames: Optional[str] = None) -> None:
        if frames is None:
            frames = self.frames
        for frame in frames:
            pose = self.get_pose(frame)
            self.world.set_pose(frame, pose)

    def show(self, **kwargs: Any) -> None:
        self.set()
        self.world.show(**kwargs)


class State2D(State):
    def __init__(self, world: World2D, *args: Any, **kwargs: Any) -> None:
        super().__init__(world, *args, **kwargs)

    @cache
    def get_conf(self, frame: str) -> Position2:
        return conf_from_pose(self.get_pose(frame))
