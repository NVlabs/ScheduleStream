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
import itertools
from typing import Iterable, List, Optional, Set, Tuple, Union

# Third Party
import numpy as np
import torch
import trimesh
from curobo.types import Pose

# NVIDIA
from schedulestream.applications.custream2.utils import (
    Vector,
    matrix_from_pose,
    to_cpu,
    transform_points,
    unit_pose,
)

VoxelIndex = Tuple[int, int, int]


class OccupancyGrid:
    """Sparse 3D occupancy grid keyed by integer voxel indices.

    The grid lives in a frame ``world_from_grid`` (``self.pose``) with per-axis
    voxel side lengths ``self.resolution`` (length-3 array). A point ``p`` in
    world coordinates maps to voxel index ``floor((pose^-1 * p) / resolution)``.
    """

    _PACK_BITS = 21
    _PACK_OFFSET = 1 << 20

    def __init__(
        self,
        pose: Optional[Pose] = None,
        resolution: Union[float, Vector] = 0.05,
        points: Optional[Union[List[Vector], np.ndarray, torch.Tensor]] = None,
    ) -> None:
        self.pose = unit_pose() if pose is None else pose
        res = np.asarray(resolution, dtype=float)
        if res.ndim == 0:
            res = np.full(3, float(res))
        if res.shape != (3,):
            raise ValueError(f"resolution must be a scalar or length-3, got {resolution!r}")
        if np.any(res <= 0):
            raise ValueError(f"resolution must be positive, got {res.tolist()}")
        self.resolution = res
        self.voxels: Set[VoxelIndex] = set()
        self._packed_cache: Optional[np.ndarray] = None
        self._identity_pose: bool = pose is None or self._pose_is_identity(self.pose)
        if points is not None:
            self.add_points(points)

    @staticmethod
    def _pose_is_identity(pose: Pose, atol: float = 1e-7) -> bool:
        """Return True iff ``pose`` is (within ``atol``) the identity transform."""
        pos = to_cpu(pose.position).reshape(-1)
        quat = to_cpu(pose.quaternion).reshape(-1)
        if pos.shape[0] != 3 or quat.shape[0] != 4:
            return False
        return bool(
            np.abs(pos).max() < atol
            and abs(abs(quat[0]) - 1.0) < atol
            and np.abs(quat[1:]).max() < atol
        )

    @classmethod
    def _pack(cls, indices: np.ndarray) -> np.ndarray:
        """Pack ``[N, 3]`` int voxel indices into ``[N]`` int64 keys."""
        shifted = indices.astype(np.int64) + cls._PACK_OFFSET
        return (
            (shifted[:, 0] << (2 * cls._PACK_BITS))
            | (shifted[:, 1] << cls._PACK_BITS)
            | shifted[:, 2]
        )

    def _packed_voxels(self) -> np.ndarray:
        """Return a cached sorted ``int64`` array of packed voxel keys."""
        if self._packed_cache is None:
            if self.voxels:
                arr = np.asarray(list(self.voxels), dtype=np.int64)
                self._packed_cache = np.sort(self._pack(arr))
            else:
                self._packed_cache = np.zeros((0,), dtype=np.int64)
        return self._packed_cache

    def add_points(self, points: Union[List[Vector], np.ndarray, torch.Tensor]) -> None:
        """Mark the voxel(s) containing each point as occupied."""
        indices = self.voxel_indices(points).cpu().numpy()
        self.voxels.update(map(tuple, indices.tolist()))
        self._packed_cache = None

    def voxel_indices(self, points: Union[List[Vector], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Return integer voxel indices ``[N, 3]`` for each world-frame point."""
        if not isinstance(points, torch.Tensor):
            points = torch.as_tensor(np.asarray(points), dtype=torch.float32)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"points must have shape [N, 3], got {tuple(points.shape)}")
        if self._identity_pose:
            local = to_cpu(points)
        else:
            local = to_cpu(transform_points(self.pose.inverse(), points).squeeze(0))
        return torch.from_numpy(np.floor(local / self.resolution).astype(np.int64))

    def contains(self, points: Union[List[Vector], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Return a bool tensor ``[N]`` of whether each point falls in an occupied voxel."""
        indices = self.voxel_indices(points).cpu().numpy()
        if not self.voxels:
            return torch.zeros(len(indices), dtype=torch.bool)
        packed_query = self._pack(indices)
        sorted_voxels = self._packed_voxels()
        positions = np.searchsorted(sorted_voxels, packed_query)
        positions = np.minimum(positions, len(sorted_voxels) - 1)
        mask = sorted_voxels[positions] == packed_query
        return torch.from_numpy(mask)

    def voxel(self) -> trimesh.voxel.VoxelGrid:
        """Return an equivalent ``trimesh.voxel.VoxelGrid``.

        The occupied voxel indices are shifted to a non-negative origin so they
        fit a dense bounding shape; the lost offset is folded into ``transform``
        along with ``self.resolution`` and ``self.pose`` so trimesh's
        ``indices_to_points`` reproduces our world-frame voxel centers.
        """
        # Third Party
        from trimesh.voxel.encoding import SparseBinaryEncoding

        # NVIDIA
        from schedulestream.applications.custream2.utils import matrix_from_pose

        if not self.voxels:
            encoding = SparseBinaryEncoding(np.zeros((0, 3), dtype=np.int64), shape=(1, 1, 1))
            return trimesh.voxel.VoxelGrid(encoding)
        indices = np.asarray(sorted(self.voxels), dtype=np.int64)
        min_idx = indices.min(axis=0)
        sparse_indices = indices - min_idx
        shape = tuple(int(s) for s in (sparse_indices.max(axis=0) + 1))
        local_to_world = np.eye(4)
        local_to_world[:3, :3] = np.diag(self.resolution)
        local_to_world[:3, 3] = (min_idx + 0.5) * self.resolution
        transform = matrix_from_pose(self.pose) @ local_to_world
        encoding = SparseBinaryEncoding(sparse_indices, shape=shape)
        return trimesh.voxel.VoxelGrid(encoding, transform=transform)

    _FACE_NEIGHBOR_OFFSETS = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )

    def neighbors(self, voxel: VoxelIndex) -> Iterable[VoxelIndex]:
        """Yield the 6 face-neighbor indices of ``voxel`` (regardless of occupancy)."""
        i, j, k = voxel
        for di, dj, dk in self._FACE_NEIGHBOR_OFFSETS:
            yield (i + di, j + dj, k + dk)

    def is_boundary(self, voxel: VoxelIndex) -> bool:
        """Return True iff at least one of ``voxel``'s 6 face-neighbors is unoccupied."""
        return any(n not in self.voxels for n in self.neighbors(voxel))

    def convex_hull(self) -> trimesh.Trimesh:
        """Return the convex hull of every occupied voxel's 8 corners as a
        world-frame ``Trimesh``.

        Raises ``ValueError`` when the grid is empty.
        """
        if not self.voxels:
            raise ValueError("OccupancyGrid is empty; cannot compute convex hull")
        voxels = np.asarray(list(self.voxels), dtype=float)
        offsets = np.asarray(list(itertools.product([0.0, 1.0], repeat=3)), dtype=float)
        corners = (voxels[:, None, :] + offsets[None, :, :]).reshape(-1, 3) * self.resolution
        hull = trimesh.PointCloud(corners).convex_hull
        hull.apply_transform(matrix_from_pose(self.pose))
        return hull

    def bounding_box(self) -> "BoundingBox":
        """Return the axis-aligned ``BoundingBox`` (in world frame via ``self.pose``)
        that contains every occupied voxel.

        Raises ``ValueError`` when the grid is empty.
        """
        # NVIDIA
        from schedulestream.applications.custream2.bounding import BoundingBox
        from schedulestream.applications.custream2.utils import (
            multiply_poses,
            pose_from_pos_quat,
        )

        if not self.voxels:
            raise ValueError("OccupancyGrid is empty; cannot compute bounding box")
        indices = np.asarray(list(self.voxels), dtype=float)
        lower = indices.min(axis=0) * self.resolution
        upper = (indices.max(axis=0) + 1) * self.resolution
        extent = upper - lower
        center = (upper + lower) / 2
        pose = multiply_poses(self.pose, pose_from_pos_quat(center))
        return BoundingBox(extent, pose=pose)

    def mesh(
        self,
        color: Tuple[int, int, int, int] = (200, 100, 0, 120),
        outline: bool = False,
    ) -> Optional[trimesh.Trimesh]:
        """Return a single merged ``Trimesh`` of occupied voxels in world frame,
        or ``None`` when no voxels qualify.

        Each emitted voxel contributes a ``self.resolution``-sized box centered
        at its grid-frame center; the boxes are merged and transformed by
        ``self.pose`` so they land in world coordinates. When ``outline`` is
        True, only voxels with at least one unoccupied face-neighbor (i.e.
        boundary cells) are drawn — interior cells are skipped.
        """
        voxels = sorted(v for v in self.voxels if not outline or self.is_boundary(v))
        if not voxels:
            return None
        local_centers = (np.asarray(voxels, dtype=float) + 0.5) * self.resolution
        meshes = []
        for center in local_centers:
            box = trimesh.creation.box(extents=self.resolution)
            box.apply_translation(center)
            box.visual.face_colors = list(color)
            meshes.append(box)
        merged = trimesh.util.concatenate(meshes)
        merged.apply_transform(matrix_from_pose(self.pose))
        return merged

    def scene(
        self,
        scene: Optional[trimesh.Scene] = None,
        color: Tuple[int, int, int, int] = (200, 100, 0, 120),
        outline: bool = False,
        name: str = "occupancy_grid",
    ) -> trimesh.Scene:
        """Return a ``trimesh.Scene`` containing one box per emitted voxel.

        Each voxel becomes an axis-aligned ``self.resolution``-sized box placed
        at the voxel's world-frame center (via ``self.pose``). When ``scene``
        is provided, the boxes are added to it under ``node_name=name``;
        otherwise a new scene is created. When ``outline`` is True, interior
        voxels (those with all 6 face-neighbors occupied) are skipped — see
        :meth:`mesh`.
        """
        if scene is None:
            scene = trimesh.Scene()
        merged = self.mesh(color=color, outline=outline)
        if merged is not None:
            scene.add_geometry(merged, node_name=name)
        return scene

    def __len__(self) -> int:
        return len(self.voxels)

    def __iter__(self) -> Iterable[VoxelIndex]:
        return iter(self.voxels)

    def __contains__(self, voxel: VoxelIndex) -> bool:
        return tuple(voxel) in self.voxels

    def __str__(self) -> str:
        return f"{type(self).__name__}(occupied={len(self.voxels)})"

    __repr__ = __str__
