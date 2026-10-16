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
from functools import cached_property
from typing import List, Optional, Tuple

# Third Party
import numpy as np
import trimesh
from curobo.types import Pose

# NVIDIA
from schedulestream.applications.custream2.utils import (
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

    @cached_property
    def center(self) -> Vector:
        """Return the world-frame position of the box centroid."""
        return to_cpu(self.origin_from_center.position[0])

    @cached_property
    def bottom_center(self) -> np.ndarray:
        """Return the world-frame position of the bottom-face center."""
        return to_cpu(self.origin_from_bottom.position[0])

    @cached_property
    def top_center(self) -> np.ndarray:
        """Return the world-frame position of the top-face center."""
        return to_cpu(self.origin_from_top.position[0])

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

    def transform(self, pose: Pose) -> BoundingBox:
        return BoundingBox(self.extent, multiply_poses(pose, self.pose))

    def extend(self, extent: Vector) -> BoundingBox:
        if np.isscalar(extent):
            extent *= np.ones(len(self.extent))
        return BoundingBox(self.extent + np.array(extent), self.pose)

    def scale(self, scale: Vector) -> BoundingBox:
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
    def mesh(self) -> trimesh.Trimesh:
        return self.box.to_mesh()

    @cached_property
    def world_corners(self) -> np.ndarray:
        """Return the 8 corner points of this box in world coordinates."""
        return to_cpu(transform_points(self.pose, self.corners).squeeze(0))

    @cached_property
    def axis_aligned(self) -> BoundingBox:
        return BoundingBox.from_points(self.world_corners)

    @cached_property
    def origin_from_bottom(self) -> Pose:
        """Return the world-frame pose of the bottom-face center."""
        position = np.append(np.zeros(2), [self.lower[2]])
        return multiply_poses(self.pose, pose_from_pos_quat(position=position))

    @cached_property
    def origin_from_center(self) -> Pose:
        """Return the world-frame pose of the box centroid (= ``self.pose``)."""
        return self.pose

    @cached_property
    def origin_from_top(self) -> Pose:
        """Return the world-frame pose of the top-face center."""
        position = np.append(np.zeros(2), [self.upper[2]])
        return multiply_poses(self.pose, pose_from_pos_quat(position=position))

    @cached_property
    def top(self) -> BoundingBox:
        return BoundingBox(extent=[self.width, self.depth, 0.0], pose=self.origin_from_top)

    @cached_property
    def bottom(self) -> BoundingBox:
        return BoundingBox(extent=[self.width, self.depth, 0.0], pose=self.origin_from_bottom)

    @staticmethod
    def from_bounds(bounds: Tuple[Vector, Vector]) -> BoundingBox:
        lower, upper = bounds
        extent = upper - lower
        center = (upper + lower) / 2
        return BoundingBox(extent, pose=pose_from_pos_quat(center))

    @staticmethod
    def from_points(points: List[Vector]) -> BoundingBox:
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        return BoundingBox.from_bounds((lower, upper))

    @staticmethod
    def from_point(point: Vector) -> BoundingBox:
        return BoundingBox.from_points([point])

    @staticmethod
    def from_spheres(spheres: List[Vector]) -> BoundingBox:
        spheres = np.asarray(spheres)
        centers, radii = spheres[:, :3], spheres[:, 3:4]
        lower = np.min(centers - radii, axis=0)
        upper = np.max(centers + radii, axis=0)
        return BoundingBox.from_bounds((lower, upper))

    @staticmethod
    def from_sphere(sphere: Vector) -> BoundingBox:
        return BoundingBox.from_spheres([sphere])

    @staticmethod
    def union(boxes: List[BoundingBox]) -> BoundingBox:
        """Return the union AABB of ``boxes`` expressed in the first box's frame.

        Each box's corners are mapped into the first box's local frame, then
        an axis-aligned bounding box is fit to all those points. The result
        is re-expressed in world coordinates using the first box's pose, so
        the union shares the first box's orientation.
        """
        if not boxes:
            raise ValueError("BoundingBox.union: empty list")
        if len(boxes) == 1:
            return boxes[0]
        ref_pose = boxes[0].pose
        inv_ref = ref_pose.inverse()
        all_points: List[np.ndarray] = []
        for box in boxes:
            ref_corners = transform_points(inv_ref, box.world_corners).squeeze(0)
            all_points.append(to_cpu(ref_corners))
        local = BoundingBox.from_points(np.concatenate(all_points, axis=0))
        return local.transform(ref_pose)

    @staticmethod
    def intersection(boxes: List[BoundingBox]) -> Optional[BoundingBox]:
        """Return the axis-aligned intersection of ``boxes`` in the first box's frame.

        Each box's world-frame corners are mapped into the first box's local
        frame and an AABB is fit (conservative for rotated boxes — the AABB
        of a rotated OBB is a superset of the OBB itself). The intersection
        is the per-axis ``[max-of-lowers, min-of-uppers]``. Returns ``None``
        when the boxes don't all overlap.
        """
        if not boxes:
            raise ValueError("BoundingBox.intersection: empty list")
        if len(boxes) == 1:
            return boxes[0]
        ref_pose = boxes[0].pose
        inv_ref = ref_pose.inverse()
        lower = -np.full(boxes[0].dim, np.inf)
        upper = +np.full(boxes[0].dim, np.inf)
        for box in boxes:
            ref_corners = to_cpu(transform_points(inv_ref, box.world_corners).squeeze(0))
            lower = np.maximum(lower, ref_corners.min(axis=0))
            upper = np.minimum(upper, ref_corners.max(axis=0))
        if np.any(lower > upper):
            return None
        local = BoundingBox.from_bounds((lower, upper))
        return local.transform(ref_pose)

    def contains(self, points: List[Vector]) -> np.ndarray:
        batch = len(points)
        points = to_cpu(transform_points(self.pose.inverse(), points).squeeze(0))
        lower = np.tile(self.lower, (batch, 1))
        upper = np.tile(self.upper, (batch, 1))
        return (np.less_equal(lower, points) & np.less_equal(points, upper)).all(axis=1)

    def contain(self, point: Vector) -> bool:
        return self.contains([point])[0]

    def distances(self, points: List[Vector]) -> np.ndarray:
        batch = len(points)
        points = to_cpu(transform_points(self.pose.inverse(), points).squeeze(0))
        lower = np.tile(self.lower, (batch, 1))
        upper = np.tile(self.upper, (batch, 1))
        lower_difference = np.maximum(lower - points, np.zeros(lower.shape))
        upper_difference = np.maximum(points - upper, np.zeros(lower.shape))
        difference = lower_difference + upper_difference
        return np.linalg.norm(difference, axis=1)

    def distance(self, point: Vector) -> float:
        return self.distances([point])[0]

    def closest_surface_points(self, points: List[Vector]) -> np.ndarray:
        """Return the closest point on the box surface for each query point, in the box's world frame.

        Points outside the box clamp to the box bounds (which lies on the
        surface). Points inside snap to the nearest of the 6 face planes:
        the axis (and ±) whose distance to its corresponding bound is
        smallest. Returns an ``(N, dim)`` array of world-frame points.
        """
        points_arr = np.asarray(points, dtype=float)
        batch = points_arr.shape[0]
        local = to_cpu(transform_points(self.pose.inverse(), points_arr).squeeze(0))
        lower_b = np.tile(self.lower, (batch, 1))
        upper_b = np.tile(self.upper, (batch, 1))
        clamped = np.clip(local, lower_b, upper_b)
        inside = np.all(clamped == local, axis=1)
        if inside.any():
            d_lower = local - self.lower
            d_upper = self.upper - local
            d_all = np.stack([d_lower, d_upper], axis=-1).reshape(batch, -1)
            argmin = np.argmin(d_all, axis=1)
            axis = argmin // 2
            side = argmin % 2
            on_axis = np.arange(self.dim)[None, :] == axis[:, None]
            target = np.where(side[:, None] == 0, self.lower[None, :], self.upper[None, :])
            snapped = np.where(on_axis, target, local)
            clamped = np.where(inside[:, None], snapped, clamped)
        return to_cpu(transform_points(self.pose, clamped).squeeze(0))

    def closest_surface_point(self, point: Vector) -> Vector:
        return self.closest_surface_points([point])[0]

    def surface_vectors(self, points: List[Vector]) -> np.ndarray:
        """Return the vector from each query point to the closest point on the box surface.

        For a point ``p`` and surface-projection ``s``, returns ``s - p``;
        magnitude equals :meth:`distance` for outside points and the depth
        of the nearest face for inside points.
        """
        points_arr = np.asarray(points, dtype=float)
        return self.closest_surface_points(points_arr) - points_arr

    def surface_vector(self, point: Vector) -> Vector:
        return self.surface_vectors([point])[0]

    def sample(self, num: int = 1) -> List[Vector]:
        return np.random.uniform(self.lower, self.upper, size=(num, self.dim))

    def show(self, **kwargs) -> None:
        """Open an interactive viewer showing this box as a wireframe outline."""
        scene = trimesh.Scene([self.outline])
        scene.show(**kwargs)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.extent}, {self.center})"

    __repr__ = __str__
