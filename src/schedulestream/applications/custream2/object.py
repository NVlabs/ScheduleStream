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
from dataclasses import dataclass
from functools import cache, cached_property
from itertools import cycle
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
import trimesh
from curobo._src.geom.sphere_fit.fit_spheres import fit_spheres_to_mesh
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo.scene import Capsule, Cuboid, Cylinder, Mesh, Obstacle, Sphere
from curobo.types import Pose

# NVIDIA
from schedulestream.applications.custream2.bounding import BoundingBox
from schedulestream.applications.custream2.utils import (
    PI,
    Color,
    invert_pose,
    multiply_poses,
    pose_from_pos_quat,
    position_from_pose,
    sample_yaw_quaternions,
    scene_sphere_contacts,
    to_cpu,
    to_float_color,
    to_matrix,
    to_pose,
    transform_spheres,
    unit_pose,
)
from schedulestream.applications.custream.spheres import greedy_sample_mesh
from schedulestream.applications.trimesh2d.utils import get_color, spaced_colors, to_uint8_color
from schedulestream.common.utils import INF, current_time, elapsed_time, safe_zip, take


def _color_to_float(color: Optional[Color]) -> Optional[List[float]]:
    """Return RGBA as float list in [0, 1].  get_color already returns floats."""
    if color is None:
        return None
    rgba = np.array(get_color(color), dtype=np.float32)
    if rgba.shape == (3,):
        rgba = np.append(rgba, 1.0)
    return rgba.tolist()


@dataclass
class GraspConfig:
    enabled: bool = True
    approach: Optional[str] = None
    roll_interval: Optional[Union[str, Tuple[float, float]]] = "upper"
    top_offset: float = INF
    max_grasps: float = INF
    reverse: bool = False
    generator: Optional[Iterator[Pose]] = None


@dataclass
class PlacementConfig:
    world_orientation: bool = True
    yaw_interval: Tuple[float, float] = (-PI, +PI)
    yaw_steps: Optional[int] = None
    stable_orientations: Optional[int] = None

    def sample_yaw_quaternions(self, batch_size: int) -> np.ndarray:
        """Return a ``(batch_size, 4)`` array of ``[w, x, y, z]`` quaternions,
        each a yaw rotation about z sampled from this config's yaw interval (or a
        random choice of ``yaw_steps`` discrete yaws when set)."""
        return sample_yaw_quaternions(batch_size, self.yaw_interval, self.yaw_steps)


@dataclass
class SurfaceConfig:
    enabled: bool = True
    approach: Optional[str] = "cuboid"
    world_orientation: bool = True
    xy_extend: float = -5e-2
    z_offset: float = 1e-3
    min_area: float = 0.1**2

    @property
    def surface_extend(self) -> np.ndarray:
        """Return the 3D extent adjustment applied to surface bounding boxes."""
        return np.array([self.xy_extend, self.xy_extend, 0.0])


@dataclass
class JointConfig:
    """The joint attaching an articulated link to a static parent, in the world frame.

    A ``"prismatic"`` joint translates the link along ``axis``; a ``"revolute"`` one rotates it
    about ``axis`` through ``origin``. ``zero_pose`` is the link's world pose at position 0.
    """

    joint_type: str
    axis: Tuple[float, float, float]
    limits: Tuple[float, float]
    zero_pose: Pose
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def get_poses(self, positions: torch.Tensor) -> Pose:
        """Return the link's world poses ``[N]`` at the joint positions ``[N]``."""
        device = positions.device
        positions = positions.reshape(-1, 1).float()
        axis = torch.tensor(self.axis, dtype=torch.float32, device=device).expand(len(positions), 3)
        if self.joint_type == "prismatic":
            quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(
                len(positions), 4
            )
            motion = Pose(position=positions * axis, quaternion=quaternions.clone())
        elif self.joint_type == "revolute":
            quaternions = torch.cat(
                [torch.cos(positions / 2), torch.sin(positions / 2) * axis], dim=1
            )
            rotation = Pose(position=torch.zeros_like(axis), quaternion=quaternions)
            origin = torch.tensor(self.origin, dtype=torch.float32, device=device)
            rotated = torch.einsum("nij,j->ni", rotation.get_rotation(), origin)
            motion = Pose(position=origin - rotated, quaternion=quaternions)
        else:
            raise ValueError(self.joint_type)
        zero_pose = Pose(
            position=self.zero_pose.position.to(device),
            quaternion=self.zero_pose.quaternion.to(device),
        )
        return multiply_poses(motion, zero_pose)


@dataclass
class SphereConfig:
    fit_type: str = "greedy"
    num_spheres: int = 50
    surface_radius: int = 1e-2
    convexify: bool = False


MAX_DISTANCE: float = 0.5


class Object:
    primitive = None

    def __init__(
        self,
        grasp_config: Optional[GraspConfig] = None,
        placement_config: Optional[PlacementConfig] = None,
        surface_config: Optional[SurfaceConfig] = None,
        sphere_config: Optional[SphereConfig] = None,
        joint_config: Optional[JointConfig] = None,
        virtual: bool = False,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.attributes = dict(attributes or {})
        self.grasp_config = grasp_config
        self.placement_config = placement_config
        self.surface_config = surface_config
        self.sphere_config = sphere_config or SphereConfig()
        self.joint_config = joint_config
        self.virtual = virtual
        self.warmed_up = False
        self.world = None

    @property
    def graspable(self) -> bool:
        """Return True if this object has an enabled grasp configuration."""
        return (self.grasp_config is not None) and self.grasp_config.enabled

    @property
    def pushable(self) -> bool:
        """Return True if this object can be pushed (always False for the base class)."""
        return False

    @property
    def articulated(self) -> bool:
        """Return True if this object is a link moved by a joint (see :class:`JointConfig`)."""
        return self.joint_config is not None

    @property
    def movable(self) -> bool:
        """Return True if this object can be moved (i.e., it is graspable or articulated)."""
        return self.graspable or self.articulated

    @property
    def placeable(self) -> bool:
        """Return True if other objects can be placed on top of this object."""
        raise NotImplementedError()

    @property
    def stackable(self) -> bool:
        """Return True if this object has an enabled surface configuration for stacking."""
        return (self.surface_config is not None) and self.surface_config.enabled

    @cached_property
    def mesh(self) -> trimesh.Trimesh:
        """Return the cached trimesh geometry for this object."""
        return self.get_trimesh_mesh()

    @staticmethod
    def _sort_meshes(meshes: List[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
        """Sort meshes by their (z, y, x) bounding-box center, rounded to 2 decimals."""

        def key(mesh: trimesh.Trimesh) -> Tuple[float, float, float]:
            x, y, z = np.round(mesh.bounds.mean(axis=0), 2)
            return z, y, x

        return sorted(meshes, key=key)

    @cached_property
    def components(self) -> List[trimesh.Trimesh]:
        """Return the mesh's face-connected components as a list of meshes,
        sorted by (z, y, x) center.

        Splits :attr:`mesh` on face adjacency (via ``trimesh.graph.split``).
        """
        return self._sort_meshes(self.mesh.split(only_watertight=False))

    def get_surfaces(
        self, normal: Any = (0.0, 0.0, 1.0), tolerance: float = PI / 16
    ) -> List[trimesh.Trimesh]:
        """Return connected-component meshes of faces whose outward normal is
        within ``tolerance`` radians of ``normal``, sorted by (z, y, x) center.

        Filters :attr:`mesh`'s faces to those aligned with ``normal`` (e.g.
        ``[0, 0, 1]`` for upward-facing surfaces), then splits the retained
        faces on face adjacency into separate surface patches.
        """
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        face_normals = np.asarray(self.mesh.face_normals)
        mask = (face_normals @ normal) >= np.cos(tolerance)
        face_indices = np.flatnonzero(mask)
        if len(face_indices) == 0:
            return []
        submesh = self.mesh.submesh([face_indices], append=True)
        return self._sort_meshes(submesh.split(only_watertight=False))

    @cached_property
    def surfaces(self) -> List[trimesh.Trimesh]:
        """Return the upward-facing :meth:`get_surfaces` patches, dropping any
        whose area is below ``surface_config.min_area`` (when configured). Each
        surface is tinted a distinct color for visualization."""
        surfaces = self.get_surfaces()
        min_area = getattr(self.surface_config, "min_area", None)
        if min_area is not None:
            surfaces = [s for s in surfaces if s.area >= min_area]
        for surface, color in zip(surfaces, spaced_colors(len(surfaces))):
            surface.visual.face_colors = to_uint8_color(color)
        return surfaces

    @cached_property
    def spheres_tensor(self) -> torch.Tensor:
        """Cached (n, 4) local-frame sphere tensor [x, y, z, r]."""
        mesh = self.mesh
        if self.sphere_config.convexify:
            mesh = mesh.convex_hull
        if self.sphere_config.fit_type == "greedy":
            centers, radii, _ = greedy_sample_mesh(
                mesh,
                n_spheres=self.sphere_config.num_spheres,
                sphere_radius=self.sphere_config.surface_radius,
                max_radius=MAX_DISTANCE,
            )
        else:
            result = fit_spheres_to_mesh(
                mesh,
                self.sphere_config.num_spheres,
                self.sphere_config.surface_radius,
                fit_type=SphereFitType(self.sphere_config.fit_type),
            )
            centers = result.centers
            radii = result.radii
        centers = to_cpu(centers)
        radii = to_cpu(radii)
        data = np.column_stack([centers, radii.reshape(-1, 1)])
        return torch.tensor(data, dtype=torch.float32)

    def warmup(self) -> bool:
        """Force sphere fitting and report the sphere count and elapsed time (no-op if already warmed).
        """
        if self.warmed_up:
            return False
        self.warmed_up = True
        t = current_time()
        spheres = self.spheres_tensor
        print(
            f"warmup) Fit {spheres.shape[0]} spheres for {self.name} in {elapsed_time(t):.3f} sec"
        )
        scene_collision = self.scene_collision
        return True

    def _as_local_obstacle(self) -> Obstacle:
        """Return a curobov2 Obstacle representing this object at identity pose."""
        raise NotImplementedError(type(self))

    @cached_property
    def scene_collision(self) -> Any:
        """SceneCollision with this object's mesh at identity pose (for sphere queries).

        ``mesh_max_dist`` controls the BVH search radius and the fallback
        distance returned by ``compute_local_sdf`` when the query point lies
        farther than that from the mesh. cuRobo defaults it to 0.1 m; with
        sphere radii larger than that the cost kernel reports a residual
        ``radius - 0.1`` positive for arbitrarily distant queries (the patch
        clamps the fallback to ``mesh_max_dist`` but doesn't make it any
        bigger). Bump it well past any sphere radius we expect so the
        fallback never triggers a false collision. Note: ``SceneCollisionCfg``
        does not plumb this through, so we build ``SceneData`` directly and
        instantiate ``SceneCollision`` manually.
        """
        # Third Party
        from curobo._src.geom.collision.checker_collision import CollisionChecker
        from curobo._src.geom.collision.collision_scene import SceneCollision
        from curobo._src.geom.data.data_scene import SceneData
        from curobo._src.geom.types import SceneCfg
        from curobo._src.types.device_cfg import DeviceCfg

        device_cfg = DeviceCfg()
        scene_cfg = SceneCfg()
        scene_cfg.add_obstacle(self._as_local_obstacle())
        data = SceneData.from_scene_cfg(scene_cfg, device_cfg, mesh_max_dist=MAX_DISTANCE)
        checker = CollisionChecker(device_cfg=device_cfg, max_distance=MAX_DISTANCE)
        return SceneCollision(
            data=data,
            checker=checker,
            device_cfg=device_cfg,
            scene_model=scene_cfg,
        )

    def get_sphere_contacts(
        self,
        spheres: torch.Tensor,
        activation_distance: float = 0.0,
        collision_distance: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(distances, directions)`` for a batch of spheres
        ``[..., S, 4]`` queried against this object's collision mesh:
        penetration depths ``[..., S]`` (positive means the sphere penetrates)
        and push-out directions ``[..., S, 3]`` in the object frame. Leading
        dims are arbitrary.

        ``collision_distance`` inflates every sphere radius before the query;
        ``activation_distance`` is cuRobo's eta (a smoothed cost ramp)."""
        scene_coll = self.scene_collision
        *lead, num_spheres, _ = spheres.shape
        query = spheres.reshape(-1, 1, num_spheres, 4).to(scene_coll.device_cfg.device)
        distances, directions = scene_sphere_contacts(
            scene_coll,
            query,
            collision_distance=collision_distance,
            activation_distance=activation_distance,
        )
        distances = distances.squeeze(1).reshape(*lead, num_spheres)
        directions = directions.squeeze(1).reshape(*lead, num_spheres, 3)
        return distances, directions

    def get_sphere_distances(
        self,
        spheres: torch.Tensor,
        activation_distance: float = 0.0,
        collision_distance: float = 0.0,
    ) -> torch.Tensor:
        """Per-sphere penetration depths (see :meth:`get_sphere_contacts`)."""
        distances, _ = self.get_sphere_contacts(
            spheres,
            activation_distance=activation_distance,
            collision_distance=collision_distance,
        )
        return distances

    def get_sphere_collisions(
        self,
        spheres: torch.Tensor,
        activation_distance: float = 0.0,
        collision_distance: float = 0.0,
    ) -> torch.Tensor:
        """Return a bool mask ``[..., S]`` (``True`` where the sphere penetrates
        this object's collision mesh) for a batch of spheres ``[..., S, 4]``."""
        return (
            self.get_sphere_distances(
                spheres,
                activation_distance=activation_distance,
                collision_distance=collision_distance,
            )
            > 0.0
        )

    @property
    def main_color(self) -> np.ndarray:
        """Return the dominant RGBA color of the mesh visual."""
        return self.mesh.visual.main_color

    @property
    def average_color(self) -> np.ndarray:
        """Return the average RGBA color (uint8) of the object.

        Uses the explicit obstacle ``color`` when one is set, otherwise the mean
        of the mesh visual's per-vertex colors (falling back to ``main_color``
        when the visual carries no per-vertex colors).
        """
        color = getattr(self, "color", None)
        if color is not None:
            return np.asarray(color)
        colors = getattr(self.mesh.visual, "vertex_colors", None)
        if colors is not None and len(colors) > 0:
            return np.round(np.asarray(colors, dtype=float).mean(axis=0)).astype(np.uint8)
        return np.asarray(self.mesh.visual.main_color)

    @property
    def matrix(self) -> np.ndarray:
        """Return the 4x4 world transform matrix for this object's current pose."""
        return to_matrix(self.pose)

    @property
    def bounds(self) -> np.ndarray:
        """Return the (2, 3) axis-aligned bounding box corners of the mesh."""
        return self.mesh.bounds

    @property
    def lower(self) -> np.ndarray:
        """Return the minimum-corner of the mesh's axis-aligned bounding box."""
        return self.bounds[0]

    @property
    def upper(self) -> np.ndarray:
        """Return the maximum-corner of the mesh's axis-aligned bounding box."""
        return self.bounds[1]

    @property
    def extent(self) -> np.ndarray:
        """Return the (x, y, z) size of the mesh's axis-aligned bounding box."""
        return self.upper - self.lower

    @property
    def dimensions(self) -> np.ndarray:
        """Return the (x, y, z) dimensions of the object (alias for extent)."""
        return self.extent

    @property
    def area(self) -> float:
        """Return the footprint area (x * y) of the object."""
        return np.prod(self.dimensions[:2])

    @property
    def volume(self) -> float:
        """Return the bounding-box volume (x * y * z) of the object."""
        return np.prod(self.dimensions)

    @property
    def width(self) -> float:
        """Return the x-axis extent of the object."""
        return float(self.extent[0])

    @property
    def depth(self) -> float:
        """Return the y-axis extent of the object."""
        return float(self.extent[1])

    @property
    def height(self) -> float:
        """Return the z-axis extent of the object."""
        return float(self.extent[2])

    @property
    def length(self) -> float:
        """Return the larger of width and depth."""
        return max(self.width, self.depth)

    @property
    def center(self) -> np.ndarray:
        """Return the centroid of the mesh bounding box in local coordinates."""
        return np.average(self.bounds, axis=0)

    @property
    def center_mass(self) -> np.ndarray:
        """Return the mesh's center of mass in local coordinates."""
        return np.asarray(self.mesh.center_mass)

    def get_stable_poses(
        self,
        convexify: bool = False,
        max_poses: Optional[int] = None,
        min_probability: float = 1e-3,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Pose:
        """Return a batched Pose of the mesh's stable resting orientations.

        Uses :meth:`trimesh.Trimesh.compute_stable_poses`, which returns the
        homogeneous transforms that rest the mesh on a flat surface ordered by
        descending probability; the transforms are stacked into a single
        batched Pose. Extra kwargs forward to ``compute_stable_poses`` (e.g.
        ``n_samples``, ``sigma``).

        ``min_probability`` is the minimum resting probability a pose must have
        to be returned (forwarded as ``compute_stable_poses``'s ``threshold``).

        ``max_poses``, when set, keeps only the ``max_poses`` most probable
        poses.

        When ``convexify`` is True the stable poses are computed on the mesh's
        convex hull, which is more robust for non-watertight or concave meshes.
        """
        mesh = self.mesh.convex_hull if convexify else self.mesh
        transforms, probabilities = mesh.compute_stable_poses(threshold=min_probability, **kwargs)
        if max_poses is not None:
            transforms = transforms[:max_poses]
            probabilities = probabilities[:max_poses]
        if verbose:
            print(
                f"{self.get_stable_poses.__name__}) Object: {self.name} |"
                f" Poses: {len(transforms)} |"
                f" Probabilities: {np.round(probabilities, 3)}"
            )
        return Pose.from_matrix(np.asarray(transforms))

    @property
    def bottom_center(self) -> np.ndarray:
        """Return the bottom-face center of the bounding box in local coordinates."""
        z = self.lower[2]
        return np.append(self.center[:2], [z])

    @property
    def top_center(self) -> np.ndarray:
        """Return the top-face center of the bounding box in local coordinates."""
        z = self.upper[2]
        return np.append(self.center[:2], [z])

    @property
    def origin_from_bottom(self) -> Pose:
        """Return the pose of the bottom-center relative to the mesh origin."""
        return pose_from_pos_quat(position=self.bottom_center)

    @property
    def origin_from_center(self) -> Pose:
        """Return the pose of the bounding-box centroid relative to the mesh origin."""
        return pose_from_pos_quat(position=self.center)

    @property
    def origin_from_top(self) -> Pose:
        """Return the pose of the top-center relative to the mesh origin."""
        return pose_from_pos_quat(position=self.top_center)

    @property
    def bounding_box(self) -> BoundingBox:
        """Return the axis-aligned BoundingBox in local coordinates."""
        return BoundingBox(self.extent, self.origin_from_center)

    def get_bounding_box(self, pose: Optional[Pose] = None) -> BoundingBox:
        """Return the axis-aligned BoundingBox transformed to world coordinates.

        ``pose`` overrides the object's current world pose; defaults to
        ``self.get_pose()``.
        """
        if pose is None:
            pose = self.get_pose()
        return self.bounding_box.transform(pose)

    def get_aligned_box(self, pose: Optional[Pose] = None, debug: bool = False) -> BoundingBox:
        """Return the axis-aligned bounding box of the mesh transformed by ``pose``.

        Transforms a copy of the mesh by ``pose`` (default: the object's
        current world pose) and takes the AABB of the resulting vertices.
        When ``local=True``, the AABB is transformed back into the object's
        local frame (so the box stays axis-aligned in world coordinates but
        is expressed relative to the object's pose). When ``debug=True``,
        the transformed mesh and the resulting box outline are shown in a
        trimesh viewer before returning.
        """
        if pose is None:
            pose = self.get_pose()
        mesh = self.mesh.copy()
        mesh.apply_transform(to_matrix(pose))
        bounds = mesh.bounds
        center = np.average(bounds, axis=0)
        extent = bounds[1] - bounds[0]
        box = BoundingBox(extent=extent, pose=pose_from_pos_quat(position=center))
        if debug:
            box.show()
            scene = trimesh.Scene([mesh, box.outline])
            scene.show()
        return box

    def get_local_box(self, pose: Optional[Pose] = None) -> BoundingBox:
        if pose is None:
            return self.bounding_box
        box = self.get_aligned_box(pose=pose)
        return box.transform(invert_pose(pose))

    def get_sphere_box(self, pose: Optional[Pose] = None) -> BoundingBox:
        """Return the axis-aligned bounding box of this object's spheres.

        Mirrors :meth:`get_local_box` but the box is built from
        ``self.spheres_tensor`` via :meth:`BoundingBox.from_spheres` instead
        of the mesh vertices. With ``pose=None`` the box is in the object's
        local frame; with ``pose`` provided, the spheres are transformed by
        ``pose`` (via :func:`transform_spheres`) and the resulting AABB is
        re-expressed in the object's local frame.
        """
        spheres = self.spheres_tensor
        if pose is not None:
            spheres = transform_spheres(pose, spheres).squeeze(0)
        box = BoundingBox.from_spheres(to_cpu(spheres))
        if pose is not None:
            box = box.transform(invert_pose(pose))
        return box

    def get_collision_box(self, pose: Optional[Pose] = None) -> BoundingBox:
        """Return the AABB union of the sphere-fit box and the mesh local box.

        Combines :meth:`get_sphere_box` and :meth:`get_local_box` (both
        evaluated at ``pose``, so both are in the object's local frame) via
        :meth:`BoundingBox.union`, so the result encloses both the collision
        spheres and the visual mesh — useful when the sphere fit and the
        mesh disagree on the object's extent (e.g. spheres clipped to a max
        radius leaving a thin tail uncovered).
        """
        return BoundingBox.union([self.get_sphere_box(pose), self.get_local_box(pose)])

    def get_pose(self) -> Pose:
        """Return the current world pose of this object as a Pose."""
        return to_pose(self.pose)

    def set_pose(self, pose: Pose, **kwargs: Any) -> None:
        """Set the world pose of this object, updating the world if one is attached."""
        self.pose = to_pose(pose).tolist()
        if self.world is not None:
            self.world.set_object_pose(self.name, pose, **kwargs)

    def get_bottom_pose(self) -> Pose:
        """Return the world pose of the object's bottom-center."""
        return multiply_poses(self.get_pose(), self.origin_from_bottom)

    def get_top_pose(self) -> Pose:
        """Return the world pose of the object's top-center."""
        return multiply_poses(self.get_pose(), self.origin_from_top)

    def place(self, orientation: Optional[Pose] = None, bottom_pose: Optional[Pose] = None) -> Pose:
        """Position the object so its bottom aligns with bottom_pose and return its world pose."""
        if bottom_pose is None:
            bottom_pose = unit_pose()
        box = self.get_local_box(pose=orientation)
        world_pose = multiply_poses(bottom_pose, box.origin_from_bottom.inverse())
        self.set_pose(world_pose)
        return world_pose

    def stack(
        self,
        parent: Optional[Object] = None,
        parent_orientation: Optional[Pose] = None,
        bottom_pose: Optional[Pose] = None,
        **kwargs: Any,
    ) -> Pose:
        """Stack this object on top of parent (or at the origin) and return its world pose."""
        if parent is None:
            top_pose = unit_pose()
        else:
            parent_box = parent.get_local_box(pose=parent_orientation)
            top_pose = multiply_poses(parent.get_pose(), parent_box.origin_from_top)
        if bottom_pose is None:
            bottom_pose = unit_pose()
        return self.place(bottom_pose=multiply_poses(top_pose, bottom_pose), **kwargs)

    def cuboid_object(self) -> CuboidObject:
        """Return a CuboidObject approximation of this object using its bounding-box dimensions."""
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

    def mesh_object(self) -> MeshObject:
        """Return a MeshObject wrapping this object's trimesh geometry."""
        return MeshObject(
            self.name,
            self.mesh,
            pose=self.pose,
            grasp_config=self.grasp_config,
            surface_config=self.surface_config,
            attributes=self.attributes,
        )

    def scene(
        self,
        scene: Optional[trimesh.Scene] = None,
        local: bool = False,
        **kwargs: Any,
    ) -> trimesh.Scene:
        """Add this object's mesh to scene (creating one if needed) and return it.

        ``local=True`` adds the mesh at identity transform (object-local frame)
        instead of the object's current world pose.
        """
        if scene is None:
            scene = trimesh.Scene()
        transform = np.eye(4) if local else self.matrix
        scene.add_geometry(
            self.mesh, node_name=self.name, geom_name=self.name, transform=transform, **kwargs
        )
        return scene

    def show(self, local: bool = False, **kwargs: Any) -> None:
        """Open an interactive trimesh viewer with this object's mesh.

        ``local=True`` shows the mesh in its object-local frame.
        """
        self.scene(local=local).show(**kwargs)

    def description(self) -> str:
        """Return a human-readable string describing this object's properties and pose."""
        position = position_from_pose(self.get_pose())
        bounding_box = self.get_bounding_box()
        return (
            f"Object: {self.name} | Movable: {self.movable} | Color (RGB): {self.main_color[:3]} |"
            f" Dimensions (m): {np.round(self.dimensions, 2)} | Position (m):"
            f" {np.round(position, 2)} | Lower Corner (m): {np.round(bounding_box.lower, 2)} |"
            f" Upper Corner (m): {np.round(bounding_box.upper, 2)}"
        )

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and id(self) == id(other)

    def __hash__(self) -> int:
        return hash((type(self), id(self)))

    def __str__(self) -> str:
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
        dims = [float(width), float(depth), float(height)]
        color_float = _color_to_float(color)
        if pose is None:
            pose = unit_pose()
        pose_list = to_pose(pose).tolist()
        Cuboid.__init__(self, name=name, dims=dims, color=color_float, pose=pose_list)
        Object.__init__(self, **kwargs)
        self.cuboid = self

    def _as_local_obstacle(self) -> Cuboid:
        """Return a Cuboid obstacle with identity pose for use in a SceneCollision."""
        return Cuboid(
            name=self.name,
            dims=list(self.dims),
            color=self.color,
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )


class MeshObject(Object, Mesh):
    def __init__(
        self,
        name: str,
        mesh: trimesh.Trimesh,
        pose: Optional[Pose] = None,
        **kwargs: Any,
    ) -> None:
        pose_list = to_pose(unit_pose() if pose is None else pose).tolist()
        vc = mesh.visual.vertex_colors if hasattr(mesh.visual, "vertex_colors") else None
        fc = mesh.visual.face_colors if hasattr(mesh.visual, "face_colors") else None
        avg_color = np.average(to_float_color(vc), axis=0).tolist() if vc is not None else None
        Mesh.__init__(
            self,
            name=name,
            vertices=mesh.vertices.tolist(),
            faces=mesh.faces.tolist(),
            vertex_colors=to_float_color(vc) if vc is not None else None,
            vertex_normals=mesh.vertex_normals.tolist()
            if mesh.vertex_normals is not None
            else None,
            face_colors=to_float_color(fc) if fc is not None else None,
            color=avg_color,
            pose=pose_list,
        )
        Object.__init__(self, **kwargs)

    def _as_local_obstacle(self) -> Mesh:
        """Return a Mesh obstacle with identity pose for use in a SceneCollision."""
        return Mesh(
            name=self.name,
            vertices=self.vertices,
            faces=self.faces,
            vertex_colors=self.vertex_colors,
            vertex_normals=self.vertex_normals,
            face_colors=self.face_colors,
            color=self.color,
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )


class SceneObject(MeshObject):
    """MeshObject backed by a ``trimesh.Scene`` (concatenated to a single mesh).

    Useful for assets that load as multi-geometry scenes (e.g. GLB files
    containing several primitives); the scene's geometries are merged via
    ``scene.dump(concatenate=True)`` and fed to :class:`MeshObject`. The
    original ``trimesh.Scene`` is retained on ``self._scene`` so the
    multi-geometry structure (and any per-primitive transforms/colors) can
    be re-exported later.
    """

    def __init__(
        self,
        name: str,
        scene: trimesh.Scene,
        pose: Optional[Pose] = None,
        **kwargs: Any,
    ) -> None:
        self._scene = scene
        mesh = scene.dump(concatenate=True)
        super().__init__(name=name, mesh=mesh, pose=pose, **kwargs)

    @property
    def geometry(self) -> Dict[str, trimesh.Trimesh]:
        """Return the source scene's ``{name: Trimesh}`` geometry mapping."""
        return self._scene.geometry

    @property
    def geometry_names(self) -> List[str]:
        """Return the names of the source scene's component geometries."""
        return list(self.geometry.keys())

    @property
    def graph(self) -> Any:
        """Return the source scene's transform graph (node hierarchy)."""
        return self._scene.graph

    @property
    def node_names(self) -> List[str]:
        """Return the names of the nodes in the source scene's transform graph."""
        return list(self.graph.nodes)

    @property
    def node_geometry_names(self) -> List[str]:
        """Return the names of the nodes that carry geometry."""
        return list(self.graph.nodes_geometry)


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
        color_float = _color_to_float(color)
        self.sphere = Sphere(
            name=name,
            radius=diameter / 2.0,
            color=color_float,
            pose=[0, 0, 0, 1, 0, 0, 0],
        )
        mesh = self.sphere.get_trimesh_mesh()
        MeshObject.__init__(self, name, mesh, pose=pose, **kwargs)

    @property
    def radius(self) -> float:
        """Return the sphere radius."""
        return self.sphere.radius


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
        color_float = _color_to_float(color)
        self.cylinder = Cylinder(
            name=name,
            radius=diameter / 2.0,
            height=height,
            color=color_float,
            pose=[0, 0, 0, 1, 0, 0, 0],
        )
        mesh = self.cylinder.get_trimesh_mesh()
        MeshObject.__init__(self, name, mesh, pose=pose, **kwargs)

    @property
    def radius(self) -> float:
        """Return the cylinder radius."""
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
        color_float = _color_to_float(color)
        self.capsule = Capsule(
            name=name,
            radius=diameter / 2.0,
            base=[0, 0, -height / 2],
            tip=[0, 0, +height / 2],
            color=color_float,
            pose=[0, 0, 0, 1, 0, 0, 0],
        )
        mesh = self.capsule.get_trimesh_mesh()
        MeshObject.__init__(self, name, mesh, pose=pose, **kwargs)

    @property
    def radius(self) -> float:
        """Return the capsule radius."""
        return self.capsule.radius


def create_primitive(
    name: str,
    primitive: str = "cuboid",
    width: Optional[float] = None,
    depth: Optional[float] = None,
    height: Optional[float] = None,
    **kwargs: Any,
) -> Object:
    """Instantiate and return a primitive Object of the given shape type."""
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
    """Create and return a list of num_objects primitive objects stacked on parent."""
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
