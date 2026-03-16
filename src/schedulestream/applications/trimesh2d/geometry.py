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
from functools import partial
from typing import Annotated, Any, List, Literal, Optional, Tuple

# Third Party
import numpy as np
import trimesh

# NVIDIA
from schedulestream.applications.trimesh2d.utils import Color, to_uint8_color

Vector = np.ndarray
Vector2 = Annotated[Vector, Literal[2]]
Vector3 = Annotated[Vector, Literal[3]]
Matrix4x4 = Annotated[Vector, Literal[4, 4]]

Position2 = Vector2
Position = Vector3
Conf = Position2
Path = List[Conf]
Pose = Matrix4x4


def invert_pose(pose: Pose) -> Pose:
    return np.linalg.inv(pose)


def multiply_poses(pose: Pose, *poses: Pose) -> Pose:
    for _pose in poses:
        pose = pose @ _pose
    return pose


def to_pose(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> Pose:
    position = np.array([x, y, z])
    matrix = trimesh.transformations.euler_matrix(roll, pitch, yaw, "rxyz")
    matrix[:3, 3] = position
    return matrix


def pose_from_position(position: Position) -> Pose:
    return to_pose(*position)


def position_from_conf(conf: Conf, z: float = 0.0) -> Position:
    assert len(conf) == 2
    position = np.append(conf, [z])
    return position


def pose_from_conf(conf: Conf, **kwargs: Any) -> Pose:
    return pose_from_position(position_from_conf(conf, **kwargs))


def position_from_pose(pose: Pose) -> Position:
    return pose[:3, 3]


def conf_from_position(position: Position) -> Conf:
    return position[:2]


def conf_from_pose(pose: Pose) -> Conf:
    return conf_from_position(position_from_pose(pose))


def get_bounds_center(bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return np.average(bounds, axis=0)


def get_bounds_extent(bounds: Tuple[Vector, Vector]) -> Vector:
    return np.array(bounds[1]) - np.array(bounds[0])


def bounds_from_center_extent(center: Vector, extent: Vector) -> Tuple[Vector, Vector]:
    lower = np.array(center) - np.array(extent) / 2
    upper = np.array(center) + np.array(extent) / 2
    bounds = (lower, upper)
    return bounds


def extend_bounds(bounds: Tuple[Vector, Vector], extension: float) -> Tuple[Vector, Vector]:
    center = get_bounds_center(bounds)
    extent = get_bounds_extent(bounds) + extension * np.ones(len(center))
    return bounds_from_center_extent(center, extent)


def bounds_contain_vector(bounds: Tuple[Vector, Vector], vector: Vector) -> bool:
    lower, upper = bounds
    return bool(np.less_equal(lower, vector).all() and np.less_equal(vector, upper).all())


def get_mesh_bounds(mesh: trimesh.Trimesh) -> np.ndarray:
    return mesh.bounding_box.bounds


def get_mesh_center(mesh: trimesh.Trimesh) -> np.ndarray:
    return get_bounds_center(get_mesh_bounds(mesh))


def get_mesh_bottom(mesh: trimesh.Trimesh) -> np.ndarray:
    bounds = get_mesh_bounds(mesh)
    x, _, _ = get_mesh_center(mesh)
    _, y, _ = bounds[0]
    return np.array([x, y])


def get_mesh_top(mesh: trimesh.Trimesh) -> np.ndarray:
    bounds = get_mesh_bounds(mesh)
    x, _, _ = get_mesh_center(mesh)
    _, y, _ = bounds[1]
    return np.array([x, y])


def set_visual(geometry: trimesh.Trimesh, color: Optional[Color] = None):
    if color is None:
        return None
    color = to_uint8_color(color)
    visual = trimesh.visual.create_visual(mesh=geometry, face_colors=color, vertex_colors=color)
    geometry.visual = visual
    return visual


def create_primitive(
    shape: str,
    width: float,
    depth: Optional[float] = None,
    height: Optional[float] = None,
    color: Optional[Color] = None,
    pose: Optional[Pose] = None,
    **kwargs: Any,
) -> trimesh.Trimesh:
    metadata = kwargs
    if depth is None:
        depth = width
    if height is None:
        height = width
    if shape == "box":
        geometry = trimesh.primitives.Box(
            extents=[width, depth, height],
            transform=pose,
        )
    elif shape == "capsule":
        geometry = trimesh.primitives.Capsule(
            radius=width / 2.0,
            height=height - width,
        )
    elif shape == "cylinder":
        geometry = trimesh.primitives.Cylinder(
            radius=width / 2.0,
            height=height,
        )
    elif shape == "sphere":
        geometry = trimesh.primitives.Sphere(
            radius=width / 2.0,
        )
    else:
        raise ValueError(shape)
    geometry = geometry.to_mesh()
    geometry = Body(metadata=metadata, **geometry.to_dict())
    set_visual(geometry, color)
    return geometry


create_box = partial(create_primitive, "box")
create_capsule = partial(create_primitive, "capsule")
create_cylinder = partial(create_primitive, "cylinder")
create_sphere = partial(create_primitive, "sphere")


def create_bounds(bounds: np.ndarray, **kwargs: Any) -> trimesh.Trimesh:
    extent = bounds[1] - bounds[0]
    center = np.average(bounds, axis=0)
    return create_box(*extent, pose=to_pose(*center), **kwargs)


def create_lines(
    lines: List[Vector3], color: Optional[np.ndarray] = None
) -> Optional[trimesh.path.Path3D]:
    lines = list(lines)
    if len(lines) == 0:
        return None
    geometry = trimesh.path.Path3D(**trimesh.path.exchange.misc.lines_to_path(lines))
    if color is not None:
        geometry.colors = [color] * len(geometry.entities)
    return geometry


def create_line(line: np.ndarray, **kwargs: Any) -> Optional[trimesh.path.Path3D]:
    return create_lines(lines=[line], **kwargs)


def create_path(path: List[Vector3], **kwargs: Any) -> Optional[trimesh.path.Path3D]:
    lines = list(zip(path[:-1], path[1:]))
    return create_lines(lines, **kwargs)


def create_scene(geometries: Optional[List[trimesh.Trimesh]] = None) -> trimesh.Scene:
    if geometries is None:
        geometries = {}
    scene = trimesh.Scene()
    for geometry in geometries:
        name = geometry.metadata.get("name")
        scene.add_geometry(geometry, node_name=name)
    return scene


class Body(trimesh.Trimesh):
    def __init__(
        self,
        *args: Any,
        world: Optional["World"] = None,
        pose: Optional[Pose] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.world = world
        self.pose = pose

    @property
    def name(self) -> str:
        return self.metadata.get("name", self.__class__.__name__.lower())

    @property
    def movable(self) -> bool:
        return self.metadata.get("movable", False)

    @property
    def height(self) -> float:
        return self.extents[2]

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__


@dataclass
class Attachment:
    parent: str
    pose: Pose

    def get(self, world: "World") -> Pose:
        return multiply_poses(world.get_pose(self.parent), self.pose)
