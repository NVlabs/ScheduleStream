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
from itertools import chain, count
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

# Third Party
import numpy as np
import torch
import trimesh
from curobo.types import Pose

# NVIDIA
from schedulestream.applications.custream2.bounding import BoundingBox
from schedulestream.applications.custream2.object import PlacementConfig, SurfaceConfig
from schedulestream.applications.custream2.state import Attachment
from schedulestream.applications.custream2.utils import (
    WORLD,
    concatenate_poses,
    create_pose,
    matrix_from_pose,
    multiply_poses,
    perpendicular_basis,
    pose_from_axes,
    transform_spheres,
)
from schedulestream.common.utils import batched, current_time, elapsed_time

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.applications.custream2.world import World


class Placement(Attachment):
    """Relative pose (parent_from_obj) of an object resting on a parent surface."""

    def __init__(
        self,
        world: World,
        obj: str,
        parent: Optional[str] = WORLD,
        pose: Optional[Pose] = None,
        placement: Optional[Placement] = None,
        world_pose: Optional[Pose] = None,
        **kwargs: Any,
    ) -> None:
        if placement is not None:
            assert (parent is WORLD) or (parent == placement.obj)
            parent = placement.obj
        self.placement = placement
        super().__init__(world, obj, pose=pose, parent=parent, **kwargs)
        if parent is WORLD and world_pose is None:
            world_pose = self.pose
        self.world_pose = world_pose

    @property
    def objects(self) -> List[str]:
        if self.placement is None:
            return super().objects
        return [self.obj] + self.placement.objects

    def get_pose(self) -> Pose:
        if self.world_pose is not None:
            return self.world_pose
        return super().get_pose()

    def get_parent_pose(self) -> Pose:
        if self.placement is None:
            return super().get_parent_pose()
        return self.placement.get_pose()

    def set(self, **kwargs: Any) -> None:
        if self.placement is not None:
            self.placement.set(**kwargs)
        super().set(**kwargs)

    def stack(self, placement: Placement, world_pose: Optional[Pose] = None) -> Placement:
        """Return a new Placement of self.obj on top of placement."""
        return Placement(
            self.world, self.obj, pose=self.pose, placement=placement, world_pose=world_pose
        )


def concatenate_placement_poses(placements: List[Any]) -> Pose:
    """Return the batched world poses of ``placements``.

    When every placement shares one parent (the common stream case), the
    parent pose is looked up once and composed in a single batched multiply
    instead of per-placement ``get_pose`` calls.
    """
    if all(
        isinstance(p, Placement) and (p.placement is None) and (p.world_pose is None)
        for p in placements
    ) and (len({p.parent for p in placements}) == 1):
        parent_pose = placements[0].get_parent_pose()
        return multiply_poses(parent_pose, concatenate_poses([p.pose for p in placements]))
    return concatenate_poses([p.get_pose() for p in placements])


def sample_box_placements(
    bounding_box1: BoundingBox,
    bounding_box2: BoundingBox,
    placement_config: Optional[PlacementConfig] = None,
    surface_config: Optional[SurfaceConfig] = None,
    batch_size: int = 1000,
    device: Any = None,
) -> Pose:
    """Sample a batch of ``obj2_from_obj1`` poses resting ``bounding_box1``'s
    bottom (with a random yaw) on ``bounding_box2``'s top surface.

    ``placement_config`` controls the sampled yaw; ``surface_config`` controls
    the surface inset/offset (both default to their config defaults). Returns a
    batched Pose of size ``batch_size``.
    """
    if placement_config is None:
        placement_config = PlacementConfig()
    if surface_config is None:
        surface_config = SurfaceConfig()
    surface_box = bounding_box2.top.transform(create_pose(z=surface_config.z_offset))
    extent = np.maximum(np.zeros(3), surface_box.extent + surface_config.surface_extend)
    surface_box = BoundingBox(extent, surface_box.pose)
    c_inv = bounding_box1.origin_from_bottom.inverse()

    positions = np.random.uniform(surface_box.lower, surface_box.upper, (batch_size, 3))
    quats = placement_config.sample_yaw_quaternions(batch_size)

    pos_t = torch.tensor(positions, dtype=torch.float32, device=device)
    quat_t = torch.tensor(quats, dtype=torch.float32, device=device)
    obj2_from_surface_batch = Pose(position=pos_t, quaternion=quat_t)
    return multiply_poses(surface_box.pose, obj2_from_surface_batch, c_inv)


def sample_surface_placements(
    bounding_box1: BoundingBox,
    mesh: trimesh.Trimesh,
    placement_config: Optional[PlacementConfig] = None,
    batch_size: int = 1000,
    device: Any = None,
) -> Pose:
    """Sample ``obj2_from_obj1`` poses placing ``bounding_box1``'s bottom on
    random points of ``mesh``, aligned to the local face normal.

    For each sampled surface point the local +z axis is set to the face
    normal; a yaw drawn from ``placement_config.yaw_interval`` is applied
    around that normal. The bbox's bottom-face center is anchored to the
    sampled surface point. Returns a batched Pose of length ``batch_size``.
    """
    if placement_config is None:
        placement_config = PlacementConfig()

    points, face_indices = mesh.sample(batch_size, return_index=True)
    points = np.asarray(points)
    normals = np.asarray(mesh.face_normals)[face_indices]

    x_ref, y_ref = perpendicular_basis(normals)
    obj2_from_surface = pose_from_axes(x_ref, y_ref, normals, points)

    quats = placement_config.sample_yaw_quaternions(batch_size)
    yaw_pose = Pose(
        position=torch.zeros((batch_size, 3), dtype=torch.float32, device=device),
        quaternion=torch.tensor(quats, dtype=torch.float32, device=device),
    )
    c_inv = bounding_box1.origin_from_bottom.inverse()
    return multiply_poses(obj2_from_surface, yaw_pose, c_inv)


def placement_generator(
    world: World,
    obj1: str,
    obj2: str,
    orientation1: Optional[Pose] = None,
    orientation2: Optional[Pose] = None,
    batch_size: int = 1000,
    collisions: bool = True,
    verbose: bool = True,
    debug: Union[bool, int] = False,
) -> Iterator[Placement]:
    """Yield random Placements of obj1 on the top surface of obj2.

    ``orientation1`` / ``orientation2`` are the world poses used to orient
    obj1's and obj2's local bounding boxes; ``None`` (default) uses each
    object's canonical (identity) orientation.

    When ``collisions=True`` (default), each candidate is checked against
    obj2's collision geometry (obj1's spheres transformed into obj2's frame
    via the sampled ``obj2_from_obj1`` pose) and colliding candidates are
    dropped before being wrapped into ``Placement``.

    When ``verbose=True`` (default), prints per-iteration stats: which obj
    pair / surface approach is being sampled, how many candidates survived
    the collision filter (out of ``batch_size``), and wall-clock elapsed.

    When ``debug=True``, first opens a viewer with obj1's mesh + its local
    bounding box, then applies and shows each of the first ``debug_count``
    sampled placements via ``world.show()`` before yielding them.
    """
    body1 = world.get_object(obj1)
    placement_config = body1.placement_config
    if placement_config is None:
        placement_config = PlacementConfig()
    bounding_box1 = body1.get_collision_box(pose=orientation1)

    body2 = world.get_object(obj2)
    surface_config = body2.surface_config
    if surface_config is None:
        surface_config = SurfaceConfig()
    bounding_box2 = body2.get_local_box(pose=orientation2)

    for iteration in count():
        start_time = current_time()
        if surface_config.approach == "cuboid":
            obj2_from_obj1_batch = sample_box_placements(
                bounding_box1,
                bounding_box2,
                placement_config,
                surface_config,
                batch_size=batch_size,
                device=world.device,
            )
        elif surface_config.approach == "mesh":
            mesh = trimesh.util.concatenate(body2.surfaces)
            obj2_from_obj1_batch = sample_surface_placements(
                bounding_box1,
                mesh,
                placement_config,
                batch_size=batch_size,
                device=world.device,
            )
        else:
            raise NotImplementedError(surface_config.approach)

        if collisions:
            spheres_obj2 = transform_spheres(obj2_from_obj1_batch, body1.spheres_tensor)
            colliding = body2.get_sphere_collisions(spheres_obj2)
            safe = ~colliding.any(dim=-1)
            keep_indices = torch.nonzero(safe, as_tuple=False).flatten().tolist()
        else:
            keep_indices = list(range(batch_size))

        if verbose:
            print(
                f"{placement_generator.__name__}) Object: {obj1} | Parent: {obj2} |"
                f" Iteration: {iteration} | Approach: {surface_config.approach} |"
                f" Safe: {len(keep_indices)}/{batch_size} |"
                f" Elapsed: {elapsed_time(start_time):.3f} sec"
            )
        if not keep_indices:
            break

        placements = []
        for i in keep_indices:
            pose_i = Pose(
                position=obj2_from_obj1_batch.position[i : i + 1].clone(),
                quaternion=obj2_from_obj1_batch.quaternion[i : i + 1].clone(),
            )
            placements.append(Placement(world, obj1, obj2, pose_i))

        if debug:
            scene = body2.scene(local=True)
            for p in placements[:debug]:
                mesh = body1.mesh.copy()
                scene.add_geometry(mesh, transform=matrix_from_pose(p.pose))
            scene.show()

        yield from placements


def stable_placement_generator(
    world: World,
    obj1: str,
    obj2: str,
    **kwargs: Any,
) -> Iterator[Placement]:
    """Like :func:`placement_generator`, but orients both bounding boxes by the
    objects' current world poses (subject to their ``world_orientation`` flags),
    so obj1 keeps its current stable resting orientation on obj2's surface.
    """
    body1 = world.get_object(obj1)
    placement_config = body1.placement_config
    if placement_config is None:
        placement_config = PlacementConfig()

    body2 = world.get_object(obj2)
    surface_config = body2.surface_config
    if surface_config is None:
        surface_config = SurfaceConfig()
    orientation2 = body2.get_pose() if surface_config.world_orientation else None

    if placement_config.stable_orientations is None:
        orientation1s = [body1.get_pose() if placement_config.world_orientation else None]
    else:
        stable_poses = body1.get_stable_poses(max_poses=placement_config.stable_orientations)
        orientation1s = [stable_poses[i : i + 1] for i in range(stable_poses.batch_size)]

    generators = [
        placement_generator(
            world, obj1, obj2, orientation1=orientation1, orientation2=orientation2, **kwargs
        )
        for orientation1 in orientation1s
    ]
    yield from chain.from_iterable(zip(*generators))


def placement_stream(
    world: Any,
    obj1: str,
    obj2: str,
    placement2: Placement,
    batch_size: int = 1000,
    object_proximity: Optional[float] = 0.0,
    debug: bool = False,
    **kwargs: Any,
) -> Iterator[Placement]:
    """Yield collision-free Placements of obj1 on obj2, given obj2's planned placement.

    When ``debug=True``, each placement is applied to the world and ``world.show()``
    is called before yielding so the caller can inspect candidates interactively.
    """
    world_pose2 = placement2.get_pose()
    fixed_names = [
        obj.name
        for obj in world.fixed_objects
        if not obj.virtual and obj.name != obj2 and obj.name not in placement2.objects
    ]
    fixed_placements = {name: Placement(world, name) for name in fixed_names}
    local_placement_batches = batched(
        stable_placement_generator(world, obj1, obj2, **kwargs), batch_size
    )
    for i in count():
        start_time = current_time()
        local_placements = next(local_placement_batches, None)
        if local_placements is None:
            break

        local_poses1 = concatenate_poses([p.pose for p in local_placements])
        world_poses1 = multiply_poses(world_pose2, local_poses1)
        world_placements1 = [
            local_placements[j].stack(
                placement2,
                world_pose=Pose(
                    position=world_poses1.position[j : j + 1].clone(),
                    quaternion=world_poses1.quaternion[j : j + 1].clone(),
                ),
            )
            for j in range(len(local_placements))
        ]

        valid_indices = list(range(len(local_placements)))
        if object_proximity is not None and fixed_names:
            # NVIDIA
            from schedulestream.applications.custream2.collision import object_object_collision

            any_collision = np.zeros(len(local_placements), dtype=bool)
            for name in fixed_names:
                fixed_placement = fixed_placements[name]
                batch = [(obj1, p, name, fixed_placement) for p in world_placements1]
                result = object_object_collision(
                    world,
                    batch,
                    collision_distance=object_proximity,
                    verbose=False,
                )
                any_collision |= np.array(result, dtype=bool)
            valid_indices = [j for j, col in enumerate(any_collision) if not col]

        print(
            f"{placement_stream.__name__}) Object: {obj1} | Parent: {obj2} | Iteration: {i} |"
            f" Success: {len(valid_indices)}/{len(local_placements)} |"
            f" Elapsed: {elapsed_time(start_time):.3f} sec"
        )
        for j in valid_indices:
            placement = world_placements1[j]
            if debug:
                placement.set()
                world.show()
                continue
            yield placement
