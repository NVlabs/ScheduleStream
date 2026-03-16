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
from functools import cached_property
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
import trimesh
from curobo.geom.types import Pose

# NVIDIA
from schedulestream.applications.custream.bounding import BoundingBox
from schedulestream.applications.custream.object import Object, SurfaceConfig
from schedulestream.applications.custream.spheres import add_spheres
from schedulestream.applications.custream.state import Attachment
from schedulestream.applications.custream.utils import (
    create_pose,
    matrix_from_pose,
    multiply_poses,
    quat_from_axis_angle,
    sample_uniformly,
    to_cpu,
    transform_spheres,
)
from schedulestream.applications.custream.world import World
from schedulestream.common.graph import topological_sort
from schedulestream.common.utils import INF


class Placement(Attachment):
    def __init__(
        self,
        world: "World",
        obj: str,
        pose: Optional[Pose] = None,
        parent: Optional[str] = None,
        placement: Optional["Placement"] = None,
        **kwargs: Any,
    ):
        if placement is not None:
            assert (parent is None) or (parent == placement.obj)
            parent = placement.obj
        self.placement = placement
        super().__init__(world, obj, pose=pose, parent=parent, **kwargs)

    @property
    def body(self) -> Object:
        return self.world.get_object(self.obj)

    @property
    def objects(self) -> List[str]:
        if self.placement is None:
            return super().objects
        return [self.obj] + self.placement.objects

    def get_parent_pose(self) -> Pose:
        if self.placement is None:
            return super().get_parent_pose()
        return self.placement.get_pose()

    def stack(self, placement: "Placement") -> "Placement":
        return self.__class__(self.world, self.obj, self.pose, placement=placement)

    def set(self) -> None:
        if self.placement is not None:
            self.placement.set()
        self.world.set_object_pose(self.obj, self.get_pose())

    @property
    def spheres(self) -> torch.Tensor:
        spheres_tensor = self.body.get_spheres_tensor()
        return transform_spheres(self.get_pose(), spheres_tensor)

    def scene(self, scene: Optional[trimesh.Scene] = None) -> trimesh.Scene:
        if scene is None:
            scene = self.world.objects_scene()
        if self.placement is not None:
            scene = self.placement.scene(scene)
        scene.graph.update(
            frame_to=self.obj, frame_from=None, matrix=matrix_from_pose(self.get_pose())
        )
        return add_spheres(self.spheres, scene=scene)

    def __str__(self):
        prefix = self.__class__.__name__[0].lower()
        return f"{prefix}{id(self) % 1000}"

    __repr__ = __str__


def get_stacked_objects(world: World, names: List[str] = None) -> Dict[str, str]:
    if names is None:
        names = world.movable_names
    bottom_objects = world.objects
    parent_from_obj = {}
    for top_obj in names:
        top_obj = world.get_object(top_obj)
        [base_center] = to_cpu(top_obj.get_bottom_pose().position)
        candidate_objects = [bottom_obj for bottom_obj in bottom_objects if top_obj != bottom_obj]
        candidate_objects = filter(
            lambda o: (base_center - o.get_bounding_box().top.center)[2] >= -1e-3, candidate_objects
        )
        closest_objects = sorted(
            candidate_objects,
            key=lambda o: o.get_bounding_box().top.distance(base_center),
        )
        if closest_objects:
            parent_from_obj[top_obj.name] = closest_objects[0].name
    return parent_from_obj


def get_ordered_stacks(world: World, **kwargs: Any) -> Dict[str, Optional[str]]:
    stacked_objects = get_stacked_objects(world, **kwargs)
    stacked_orders = {(bottom, top) for top, bottom in stacked_objects.items()}
    object_names = topological_sort(stacked_orders, world.object_names)
    ordered_stacks = {obj: stacked_objects.get(obj, None) for obj in object_names}
    return ordered_stacks


def sample_se2(
    world: World,
    position_interval: Tuple[np.ndarray, np.ndarray] = (-np.zeros(3), +np.zeros(3)),
    axis: Tuple[float] = (0.0, 0.0, 1.0),
    angle_interval: Tuple[float, float] = (-np.pi, +np.pi),
    batch_size: int = 1,
) -> Pose:
    position_lower, position_upper = world.to_device(np.array(position_interval))
    position = sample_uniformly(position_lower, position_upper, num=batch_size)

    axis = world.to_device(np.array(axis)).repeat(batch_size, 1)
    angle_lower, angle_upper = world.to_device(np.expand_dims(angle_interval, axis=-1))
    yaw = sample_uniformly(angle_lower, angle_upper, num=batch_size)
    quaternion = quat_from_axis_angle(axis, yaw)

    return Pose(position=position, quaternion=quaternion)


def placement_generator(
    world: World,
    obj1: Union[str, Object],
    obj2: Union[str, Object],
    batch_size: int = 10,
    debug: bool = False,
) -> Iterable[Pose]:
    obj1 = world.get_object(obj1)
    obj2 = world.get_object(obj2)
    surface_config = obj2.surface_config
    if surface_config is None:
        surface_config = SurfaceConfig(xy_extend=-INF if obj2.movable else -2e-2)

    surface_box = obj2.bounding_box.top.transform(create_pose(z=surface_config.z_offset))
    extent = np.maximum(np.zeros(3), surface_box.extent + surface_config.surface_extend)
    surface_box = BoundingBox(extent, surface_box.pose)
    while True:
        obj2_from_surface = sample_se2(
            world, position_interval=surface_box.bounds, batch_size=batch_size
        )
        obj2_from_obj1 = multiply_poses(
            surface_box.pose, obj2_from_surface, obj1.origin_from_bottom.inverse()
        )
        placements = [
            Placement(world, obj1.name, pose, parent=obj2.name) for pose in obj2_from_obj1
        ]
        if debug:
            for placement in placements:
                placement.show()
            continue
        yield from placements
