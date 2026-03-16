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
from dataclasses import dataclass
from functools import cache, cached_property
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
import trimesh
from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Obstacle, Pose, Sphere, SphereFitType
from curobo.types.math import Pose

# NVIDIA
from schedulestream.applications.custream.bounding import BoundingBox
from schedulestream.applications.custream.spheres import (
    add_obstacles,
    create_sphere_obstacles,
    greedy_sample_mesh,
)
from schedulestream.applications.custream.utils import (
    Color,
    multiply_poses,
    pose_from_pos_quat,
    to_cpu,
    to_float_color,
    to_matrix,
    to_pose,
    transform_points,
    unit_pose,
)
from schedulestream.applications.trimesh2d.utils import get_color, spaced_colors, to_uint8_color
from schedulestream.common.utils import INF, safe_zip, take


@dataclass
class GraspConfig:
    primitive: Optional[str] = None
    pitch_interval: Optional[Union[str, Tuple[float, float]]] = "upper"
    top_offset: float = INF
    max_grasps: float = INF
    reverse: float = False


@dataclass
class PlacementConfig:
    yaw_interval: Optional[Union[str, Tuple[float, float]]] = "upper"


@dataclass
class SurfaceConfig:
    xy_extend: float = -2e-2
    z_offset: float = 1e-3

    @property
    def surface_extend(self) -> np.ndarray:
        return np.array([self.xy_extend, self.xy_extend, 0.0])


class Object:
    primitive = None

    def __init__(
        self,
        grasp_config: Optional[GraspConfig] = None,
        placement_config: Optional[PlacementConfig] = None,
        surface_config: Optional[SurfaceConfig] = None,
        collider: bool = True,
        attributes: Dict[str, Any] = None,
    ) -> None:
        self.attributes = dict(attributes or {})
        self.grasp_config = grasp_config
        self.placement_config = placement_config
        self.surface_config = surface_config
        self.collider = collider
        self.world = None

    def get_attribute(self, attribute: str) -> Any:
        pass

    @property
    def graspable(self) -> bool:
        return self.grasp_config is not None

    @property
    def pushable(self) -> bool:
        return False

    @property
    def movable(self) -> bool:
        return self.graspable

    @property
    def placeable(self) -> bool:
        raise NotImplementedError()

    @property
    def stackable(self) -> bool:
        return self.surface_config is not None

    @cached_property
    def mesh(self) -> trimesh.Trimesh:
        return self.get_trimesh_mesh()

    @property
    def main_color(self) -> np.ndarray:
        return self.mesh.visual.main_color

    @property
    def matrix(self) -> np.ndarray:
        return to_matrix(self.pose)

    @property
    def bounds(self) -> np.ndarray:
        return self.mesh.bounds

    @property
    def lower(self) -> np.ndarray:
        return self.bounds[0]

    @property
    def upper(self) -> np.ndarray:
        return self.bounds[1]

    @property
    def extent(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def dimensions(self) -> np.ndarray:
        return self.extent

    @property
    def area(self) -> float:
        return np.prod(self.dimensions[:2])

    @property
    def volume(self) -> float:
        return np.prod(self.dimensions)

    @property
    def width(self) -> float:
        return float(self.extent[0])

    @property
    def depth(self) -> float:
        return float(self.extent[1])

    @property
    def height(self) -> float:
        return float(self.extent[2])

    @property
    def length(self) -> float:
        return max(self.width, self.depth)

    @property
    def center(self) -> np.ndarray:
        return np.average(self.bounds, axis=0)

    @property
    def bottom_center(self) -> np.ndarray:
        z = self.lower[2]
        return np.append(self.center[:2], [z])

    @property
    def top_center(self) -> np.ndarray:
        z = self.upper[2]
        return np.append(self.center[:2], [z])

    @property
    def origin_from_bottom(self) -> Pose:
        return pose_from_pos_quat(position=self.bottom_center)

    @property
    def origin_from_center(self) -> Pose:
        return pose_from_pos_quat(position=self.center)

    @property
    def origin_from_top(self) -> Pose:
        return pose_from_pos_quat(position=self.top_center)

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(self.extent, self.origin_from_center)

    def get_bounding_box(self) -> BoundingBox:
        return self.bounding_box.transform(self.get_pose())

    def get_pose(self) -> Pose:
        return to_pose(self.pose)

    def set_pose(self, pose: Pose) -> None:
        if self.world is None:
            self.pose = to_pose(pose).tolist()
        else:
            self.world.set_object_pose(self.name, pose)

    def get_bottom_pose(self) -> Pose:
        return multiply_poses(self.get_pose(), self.origin_from_bottom)

    def get_top_pose(self) -> Pose:
        return multiply_poses(self.get_pose(), self.origin_from_top)

    def place(self, bottom_pose: Optional[Pose] = None) -> Pose:
        if bottom_pose is None:
            bottom_pose = unit_pose()
        world_pose = multiply_poses(bottom_pose, self.origin_from_bottom.inverse())
        self.set_pose(world_pose)
        return world_pose

    def stack(self, parent: Optional["Object"] = None, bottom_pose: Optional[Pose] = None) -> Pose:
        if parent is None:
            top_pose = unit_pose()
        else:
            top_pose = parent.get_top_pose()
        if bottom_pose is None:
            bottom_pose = unit_pose()
        return self.place(bottom_pose=multiply_poses(top_pose, bottom_pose))

    @cache
    def _get_bounding_spheres(
        self,
        n_spheres: int = 50,
        surface_sphere_radius: float = 1e-2,
        fit_type: Optional[SphereFitType] = None,
        pre_transform_pose: Optional[Pose] = None,
        display: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[Sphere]:
        if fit_type is None:
            points, radii, _ = greedy_sample_mesh(self.mesh, n_spheres, surface_sphere_radius)
            spheres = create_sphere_obstacles(points, radii)
        else:
            if pre_transform_pose is None:
                pre_transform_pose = self.get_pose().inverse()
            spheres = super().get_bounding_spheres(
                n_spheres=n_spheres,
                surface_sphere_radius=surface_sphere_radius,
                fit_type=fit_type,
                pre_transform_pose=pre_transform_pose,
                **kwargs,
            )
        for sphere in spheres:
            sphere.color = np.append(to_uint8_color(self.color)[:3], [100])
        if verbose:
            print(f"{self.name}) Spheres: {len(spheres)}")
        if display:
            add_obstacles(spheres, scene=self.mesh.scene()).show()
        return spheres

    def get_bounding_spheres(
        self,
        n_spheres: int = 50,
        surface_sphere_radius: float = 1e-2,
        pre_transform_pose: Optional[Pose] = None,
        **kwargs: Any,
    ) -> List[Sphere]:
        spheres = self._get_bounding_spheres(n_spheres, surface_sphere_radius, **kwargs)
        if pre_transform_pose is None:
            return spheres
        positions = [sphere.position for sphere in spheres]
        radii = [sphere.radius for sphere in spheres]
        positions = to_cpu(
            transform_points(pre_transform_pose, self.world.to_device(positions)).squeeze(0)
        )
        return create_sphere_obstacles(positions, radii)

    def get_spheres_tensor(self, **kwargs: Any) -> torch.Tensor:
        spheres = self.get_bounding_spheres(**kwargs)
        sphere_list = [list(s.position) + [s.radius] for s in spheres]
        sphere_tensor = self.world.to_device(np.array(sphere_list))
        return sphere_tensor

    def cuboid_object(self) -> "CuboidObject":
        return CuboidObject(
            self.name,
            width=self.width,
            depth=self.depth,
            height=self.height,
            color=self.color,
            pose=self.pose,
            grasp_config=self.grasp_config,
            surface_config=self.surface_config,
            attributes=self.attributes,
        )

    def mesh_object(self) -> "MeshObject":
        return MeshObject(
            self.name,
            self.mesh,
            pose=self.pose,
            grasp_config=self.grasp_config,
            surface_config=self.surface_config,
            attributes=self.attributes,
        )

    def scene(self, scene: Optional[trimesh.Scene] = None, **kwargs: Any) -> trimesh.Scene:
        if scene is None:
            scene = trimesh.Scene()
        scene.add_geometry(self.mesh, node_name=self.name, transform=self.matrix, **kwargs)
        return scene

    def save(self):
        raise NotImplementedError()

    @staticmethod
    def from_obstacle(obstacle: Obstacle, **kwargs: Any) -> "Object":
        if isinstance(obstacle, Cuboid):
            width, depth, height = obstacle.dims
            return CuboidObject(
                name=obstacle.name,
                width=width,
                depth=depth,
                height=height,
                color=obstacle.color,
                pose=obstacle.pose,
                **kwargs,
            )
        mesh = obstacle.get_trimesh_mesh()
        return MeshObject(name=obstacle.name, mesh=mesh, pose=obstacle.pose, **kwargs)

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and id(self) == id(other)

    def __hash__(self) -> int:
        return hash((type(self), id(self)))

    def __str__(self):
        return self.name

    __repr__ = __str__


class CuboidObject(Object, Cuboid):
    primitive = "cuboid"

    def __init__(
        self,
        name: str,
        width: Optional[float] = None,
        depth: Optional[float] = None,
        height: Optional[float] = None,
        color: Optional[Color] = None,
        pose: Optional[Pose] = None,
        **kwargs: Any,
    ) -> None:
        assert (width is not None) or (depth is not None) or (height is not None)
        depth = depth or width
        width = width or depth
        height = height or width
        depth = depth or height
        width = width or height
        dims = np.array([width, depth, height])
        color = get_color(color)
        if pose is None:
            pose = unit_pose()
        pose = to_pose(pose).tolist()
        Cuboid.__init__(self, name, dims=dims, color=color, pose=pose)
        Object.__init__(self, **kwargs)
        self.cuboid = self


class MeshObject(Object, Mesh):
    def __init__(
        self,
        name: str,
        mesh: trimesh.Trimesh,
        pose: Optional[Pose] = None,
        **kwargs: Any,
    ) -> None:
        pose = to_pose(unit_pose() if pose is None else pose).tolist()
        Mesh.__init__(
            self,
            name,
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=to_float_color(mesh.visual.vertex_colors),
            vertex_normals=mesh.vertex_normals,
            face_colors=to_float_color(mesh.visual.face_colors),
            color=np.average(to_float_color(mesh.visual.vertex_colors), axis=0),
            pose=pose,
        )
        Object.__init__(self, **kwargs)


class SphereObject(MeshObject):
    primitive = "sphere"

    def __init__(
        self,
        name: str,
        diameter: float,
        color: Optional[Color] = None,
        pose: Optional[Pose] = None,
        **kwargs: Any,
    ) -> None:
        color = get_color(color)
        if pose is None:
            pose = unit_pose()
        pose = to_pose(pose).tolist()
        self.sphere = Sphere(name, radius=diameter / 2.0, color=color, pose=unit_pose())
        mesh = self.sphere.get_trimesh_mesh()
        MeshObject.__init__(self, name, mesh, pose=pose, **kwargs)

    @property
    def radius(self) -> float:
        return self.sphere.radius

    def get_bounding_spheres(self, **kwargs: Any) -> List[Sphere]:
        return [self.sphere]


class CylinderObject(MeshObject):
    primitive = "cylinder"

    def __init__(
        self,
        name: str,
        diameter: Optional[float] = None,
        height: Optional[float] = None,
        color: Optional[Color] = None,
        pose: Optional[Pose] = None,
        **kwargs: Any,
    ) -> None:
        assert (diameter is not None) or (height is not None)
        height = height or diameter
        diameter = diameter or height
        color = get_color(color)
        if pose is None:
            pose = unit_pose()
        pose = to_pose(pose).tolist()
        self.cylinder = Cylinder(name, radius=diameter / 2.0, height=height, color=color)
        mesh = self.cylinder.get_trimesh_mesh()
        MeshObject.__init__(self, name, mesh, pose=pose, **kwargs)

    @property
    def radius(self) -> float:
        return self.cylinder.radius


class CapsuleObject(MeshObject):
    primitive = "capsule"

    def __init__(
        self,
        name: str,
        diameter: Optional[float] = None,
        height: Optional[float] = None,
        color: Optional[Color] = None,
        pose: Optional[Pose] = None,
        **kwargs: Any,
    ) -> None:
        assert (diameter is not None) or (height is not None)
        height = height or diameter
        diameter = diameter or height
        color = get_color(color)
        if pose is None:
            pose = unit_pose()
        pose = to_pose(pose).tolist()
        self.capsule = Capsule(
            name,
            radius=diameter / 2.0,
            base=[0, 0, -height / 2],
            tip=[0, 0, +height / 2],
            color=color,
        )
        mesh = self.capsule.get_trimesh_mesh()
        MeshObject.__init__(self, name, mesh, pose=pose, **kwargs)

    @property
    def radius(self) -> float:
        return self.capsule.radius


def create_primitive(
    name: str,
    primitive: str = "cuboid",
    width: Optional[float] = None,
    depth: Optional[float] = None,
    height: Optional[float] = None,
    **kwargs: Any,
) -> Object:
    if primitive == "cuboid":
        return CuboidObject(name, width=width, depth=depth, height=height, **kwargs)
    if primitive == "sphere":
        return SphereObject(name, diameter=width, **kwargs)
    if primitive == "cylinder":
        return CylinderObject(name, diameter=width, height=height, **kwargs)
    if primitive == "capsule":
        return CapsuleObject(name, diameter=width, height=height, **kwargs)
    raise ValueError(primitive)


def create_primitive_objects(
    num_objects: int,
    name: Optional[str] = None,
    primitive: str = "cuboid",
    parent: Optional[Object] = None,
    colors: Optional[List[Color]] = None,
    poses: Optional[List[Pose]] = None,
    start_index: int = 0,
    **kwargs: Any,
) -> List[Object]:
    if name is None:
        name = primitive
    if colors is None:
        colors = spaced_colors(num_objects)
    colors = list(take(cycle(colors), num_objects))
    assert num_objects == len(colors)
    if poses is None:
        poses = [None]
    poses = list(take(cycle(poses), num_objects))
    assert num_objects == len(poses)

    objects = []
    for i, (color, pose) in enumerate(safe_zip(colors, poses)):
        index = start_index + i
        obj = create_primitive(name=f"{name}{index}", primitive=primitive, color=color, **kwargs)
        obj.stack(parent, bottom_pose=pose)
        objects.append(obj)
    return objects
