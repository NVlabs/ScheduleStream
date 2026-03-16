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
from typing import Any, List, Optional, Tuple

# Third Party
import numpy as np
import torch
from curobo.types.math import Pose
from curobo.types.state import JointState

# NVIDIA
from schedulestream.applications.custream.command import ArmPath, Commands, Trajectory
from schedulestream.applications.custream.grasp import Grasp
from schedulestream.applications.custream.placement import Placement
from schedulestream.applications.custream.spheres import add_cloud, add_spheres
from schedulestream.applications.custream.state import Configuration
from schedulestream.applications.custream.utils import (
    matrix_from_pose,
    multiply_poses,
    to_cpu,
    transform_spheres,
)
from schedulestream.applications.custream.world import World
from schedulestream.common.utils import batched, current_time, elapsed_time


def compute_sphere_proximity(spheres1: torch.Tensor, spheres2: torch.Tensor) -> float:
    _spheres1 = torch.repeat_interleave(spheres1, repeats=len(spheres2), dim=0)
    _spheres2 = spheres2.repeat(len(spheres1), 1)
    differences = _spheres2[:, :3] - _spheres1[:, :3]
    distances = torch.norm(differences, p=2, dim=-1)
    radii = _spheres1[:, 3] + _spheres2[:, 3]
    proximities = distances - radii
    return torch.min(proximities).item()


def spheres_collision(
    world: World, spheres: torch.Tensor, collision_distance: float = 0.0, debug: bool = False
) -> bool:
    if spheres.nelement() == 0:
        return False
    distances = -world.get_sphere_distances(spheres)
    distance = torch.min(distances)
    collision = distance <= collision_distance

    if collision and debug:
        active_names = world.get_active_objects()
        print(
            f"Spheres: {to_cpu(spheres.shape)} | Obstacles: {active_names} | Distance:"
            f" {distance:.3f} m | Collision: {collision}"
        )
        scene = world.objects_scene(names=active_names)

        spheres = to_cpu(torch.flatten(spheres, end_dim=-2))
        spheres = spheres[spheres[:, -1] > 0.0]
        centers = spheres[..., :3]
        radii = spheres[..., 3]
        points = []
        for k in range(3):
            vector = np.zeros(centers.shape)
            vector[:, k] = radii
            for sign in [-1, 1]:
                points.extend(centers + sign * vector)
        scene = add_cloud(points, scene=scene)
        scene.show()
    return bool(collision)


def trajectory_collision(world: World, arm: str, traj: Trajectory, **kwargs) -> bool:
    return spheres_collision(world, traj.spheres, **kwargs)


def arm_path_collision(world: World, arm: str, arm_path: ArmPath, **kwargs) -> bool:
    return spheres_collision(world, arm_path.spheres, **kwargs)


def sphere_object_distances(
    world: World, spheres: torch.Tensor, obj: str, pose: Optional[Pose] = False, debug: bool = False
) -> torch.Tensor:
    world_from_obj = pose
    if world_from_obj is None:
        world_from_obj = world.get_object_pose(obj)
    collision_from_obj = world.get_object_pose(obj)
    collision_from_world = multiply_poses(collision_from_obj, world_from_obj.inverse())

    spheres = transform_spheres(collision_from_world, spheres)
    with world.active_context(objects=None):
        world.enable_objects_active(active_names=[obj])
        distances = -world.get_sphere_distances(spheres)

    if debug:
        min_distance = torch.min(distances)
        print(f"Spheres: {spheres.shape} | Object: {obj} | Min Distance: {min_distance:.3f} m")
        obj = world.get_object(obj)
        scene = add_spheres(spheres)
        scene.add_geometry(obj.mesh, transform=matrix_from_pose(collision_from_obj))
        scene.show()
    return distances


def trajectory_placement_collision(
    world: World,
    arm: str,
    traj: Trajectory,
    obj: str,
    placement: Placement,
    collision_distance: float = 0.0,
    debug: bool = False,
    **kwargs: Any,
) -> bool:
    if (obj in traj.objects) or (placement is None):
        return False
    distances = sphere_object_distances(world, traj.spheres, obj, placement.get_pose(), **kwargs)
    distance = torch.min(distances)
    collision = distance <= collision_distance
    if debug and collision:
        traj.scene().show()
    return bool(collision)


def commands_placement_collision(
    world: World,
    arm: str,
    commands: Commands,
    obj: str,
    placement: Placement,
    **kwargs: Any,
) -> bool:
    if isinstance(placement, Grasp):
        return False

    for traj in commands.commands:
        if isinstance(traj, Trajectory) and trajectory_placement_collision(
            world, arm, traj, obj, placement, **kwargs
        ):
            return True
    return False


def _joint_state_pair_collision(
    world: World,
    joint_state1: JointState,
    joint_state2: JointState,
    strict: bool = False,
    debug: bool = False,
) -> List[Tuple[int, int]]:
    joint_indices1 = world.get_joint_indices(joint_state1.joint_names)
    path1 = joint_state1.position

    joint_indices2 = world.get_joint_indices(joint_state2.joint_names)
    path2 = joint_state2.position

    num = len(path1) * len(path2)
    confs = world.retract_conf.repeat(num, 1)
    confs[:, joint_indices1] = torch.repeat_interleave(path1, repeats=len(path2), dim=0)
    confs[:, joint_indices2] = path2.repeat(len(path1), 1)

    start_time = current_time()
    collisions = world.get_self_collisions(confs=confs)
    colliding_indices = to_cpu(torch.nonzero(collisions)[:, 0])

    colliding_pairs = []
    for index in colliding_indices:
        i = index // len(path2)
        j = index % len(path2)
        colliding_pairs.append((i, j))

    if len(colliding_indices) != 0:
        print(
            f"Joint1: {joint_state1.joint_names[0]} | Path1: {len(path1)} | Joint2:"
            f" {joint_state2.joint_names[0]} | Path2: {len(path2)} | Colliding:"
            f" {len(colliding_indices)}/{len(confs)} | Elapsed: {elapsed_time(start_time):.3f} sec"
        )

    if strict:
        min_index2_per_index1 = {}
        for index1, index2 in colliding_pairs:
            if (index1 not in min_index2_per_index1) or (index2 < min_index2_per_index1[index1]):
                min_index2_per_index1[index1] = index2
        colliding_pairs = list(min_index2_per_index1.items())

    if debug:
        for conf in confs[colliding_indices]:
            world.set_conf(conf)
            world.show()

    return colliding_pairs


def joint_state_pair_collision(
    world: World, joint_state1: JointState, joint_state2: JointState, batch_size=1e5, **kwargs
) -> List[Tuple[int, int]]:
    index1 = world.get_joint_index(joint_state1.joint_names[0])
    index2 = world.get_joint_index(joint_state2.joint_names[0])
    if index1 > index2:
        colliding_pairs = joint_state_pair_collision(
            world, joint_state2, joint_state1, batch_size=batch_size, **kwargs
        )
        return [(i2, i1) for i1, i2 in colliding_pairs]

    if batch_size is None:
        return _joint_state_pair_collision(world, joint_state1, joint_state2, **kwargs)
    num1 = len(joint_state1)
    num2 = len(joint_state2)
    batch1 = int(batch_size / num2)

    colliding_pairs = []
    for indices1 in batched(range(num1), batch1):
        local_pairs = _joint_state_pair_collision(
            world, joint_state1[indices1], joint_state2, **kwargs
        )
        colliding_pairs.extend((indices1[index1], index2) for index1, index2 in local_pairs)
    return colliding_pairs


def commands_conf_collision(
    world: World,
    arm1: str,
    commands1: Commands,
    arm2: str,
    conf2: Configuration,
    conf_collisions: bool = False,
    **kwargs: Any,
) -> bool:
    if (arm1 == arm2) or (conf2 is None):
        return False
    if not conf_collisions and isinstance(conf2, Configuration):
        return False
    joint_state1 = commands1.joint_state
    joint_state2 = conf2.joint_state
    colliding_indices = joint_state_pair_collision(world, joint_state1, joint_state2, **kwargs)
    return len(colliding_indices) != 0


def object_object_collision(
    world: World,
    obj1: str,
    placement1: Placement,
    obj2: str,
    placement2: Placement,
    collision_distance: float = 0.0,
    verbose: bool = True,
    **kwargs: Any,
) -> bool:
    if obj1 == obj2:
        return False
    start_time = current_time()
    distances = sphere_object_distances(
        world, placement1.spheres, obj2, placement2.get_pose(), **kwargs
    )
    distance = torch.min(distances)
    collision = distance <= collision_distance
    if verbose and collision:
        world.logger.debug(
            f"Object1: {obj1} | Spheres: {placement1.spheres.shape[-2]} | Object2: {obj2} |"
            f" Collision: {collision} | Distance: {distance:.3f} m | Elapsed:"
            f" {elapsed_time(start_time):.3f} sec"
        )

    return bool(collision)
