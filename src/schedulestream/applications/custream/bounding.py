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
from typing import List, Optional, Tuple

# Third Party
import numpy as np
import trimesh
from curobo.types.math import Pose

# NVIDIA
from schedulestream.applications.custream.utils import (
    Matrix,
    Vector,
    multiply_poses,
    pose_from_pos_quat,
    to_cpu,
    transform_points,
    unit_pose,
)


class BoundingBox:
    def __init__(self, extent: Vector, pose: Optional[Pose] = None):
        if pose is None:
            pose = unit_pose()
        self.extent = np.array(extent)
        self.pose = pose

    @property
    def dim(self) -> int:
        return len(self.extent)

    @property
    def dimensions(self) -> Vector:
        return self.extent

    @property
    def half_extent(self) -> Vector:
        return self.extent / 2

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
    def center(self) -> Vector:
        return to_cpu(self.pose.position[0])

    @property
    def lower(self) -> Vector:
        return -self.half_extent

    @property
    def upper(self) -> Vector:
        return +self.half_extent

    @property
    def bounds(self) -> Tuple[Vector, Vector]:
        return self.lower, self.upper

    @property
    def corners(self) -> List[Vector]:
        return trimesh.bounds.corners(self.bounds)

    def transform(self, pose: Pose) -> "BoundingBox":
        return BoundingBox(self.extent, multiply_poses(pose, self.pose))

    def extend(self, extent: Vector) -> "BoundingBox":
        if np.isscalar(extent):
            extent *= np.ones(len(self.extent))
        return BoundingBox(self.extent + np.array(extent), self.pose)

    def scale(self, scale: Vector) -> "BoundingBox":
        if np.isscalar(scale):
            scale *= np.ones(len(self.extent))
        return BoundingBox(self.extent * np.array(scale), self.pose)

    @cached_property
    def matrix(self) -> Matrix:
        [matrix] = self.pose.get_numpy_matrix()
        return matrix

    @cached_property
    def box(self) -> trimesh.primitives.Box:
        return trimesh.primitives.Box(self.extent, self.matrix)

    @cached_property
    def outline(self) -> trimesh.path.Path3D:
        return self.box.as_outline()

    @cached_property
    def mesh(self):
        return self.box.to_mesh()

    @cached_property
    def axis_aligned(self) -> "BoundingBox":
        points = to_cpu(transform_points(self.pose, self.corners).squeeze(0))
        return BoundingBox.from_points(points)

    @cached_property
    def top(self) -> "BoundingBox":
        z = self.height / 2
        top_center = np.append(np.zeros(2), [z])
        pose = multiply_poses(self.pose, pose_from_pos_quat(position=top_center))
        return BoundingBox(extent=[self.width, self.depth, 0.0], pose=pose)

    @staticmethod
    def from_bounds(bounds: Tuple[Vector, Vector]) -> "BoundingBox":
        lower, upper = bounds
        extent = upper - lower
        center = (upper + lower) / 2
        return BoundingBox(extent, pose=pose_from_pos_quat(center))

    @staticmethod
    def from_points(points: List[Vector]) -> "BoundingBox":
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        bounds = (lower, upper)
        return BoundingBox.from_bounds(bounds)

    @staticmethod
    def from_point(point: Vector) -> "BoundingBox":
        return BoundingBox.from_points([point])

    def contains(self, points: List[Vector]) -> List[float]:
        batch = len(points)
        points = to_cpu(transform_points(self.pose.inverse(), points).squeeze(0))
        lower = np.tile(self.lower, (batch, 1))
        upper = np.tile(self.upper, (batch, 1))
        contains = (np.less_equal(lower, points) & np.less_equal(points, upper)).all(axis=1)
        return contains

    def contain(self, point: Vector) -> float:
        return self.contains([point])[0]

    def distances(self, points: List[Vector]) -> List[float]:
        batch = len(points)
        points = to_cpu(transform_points(self.pose.inverse(), points).squeeze(0))
        lower = np.tile(self.lower, (batch, 1))
        upper = np.tile(self.upper, (batch, 1))
        lower_difference = np.maximum(lower - points, np.zeros(lower.shape))
        upper_difference = np.maximum(points - upper, np.zeros(lower.shape))
        difference = lower_difference + upper_difference
        distances = np.linalg.norm(difference, axis=1)
        return distances

    def distance(self, point: Vector) -> float:
        return self.distances([point])[0]

    def sample(self, num: int = 1) -> List[Vector]:
        return np.random.uniform(self.lower, self.upper, size=(num, self.dim))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.extent}, {self.center})"

    __repr__ = __str__
