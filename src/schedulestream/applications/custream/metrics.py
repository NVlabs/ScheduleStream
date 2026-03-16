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
from typing import Any, Optional

# Third Party
import torch
from curobo.types.robot import JointState

# NVIDIA
from schedulestream.applications.custream.utils import INF
from schedulestream.applications.custream.world import World


def compute_trajectory_length(trajectory: Optional[JointState]) -> int:
    if trajectory is None:
        return INF
    return len(trajectory)


def compute_trajectory_duration(world: World, trajectory: Optional[JointState]) -> float:
    if trajectory is None:
        return INF
    return len(trajectory) * world.time_step


def compute_joint_distances(trajectory: JointState, **kwargs: Any) -> float:
    assert len(trajectory) >= 1
    differences = trajectory.position[:-1, :] - trajectory.position[1:, :]
    return torch.norm(differences, dim=1, **kwargs)


def compute_joint_distance(trajectory: Optional[JointState], **kwargs: Any) -> float:
    if trajectory is None:
        return INF
    if len(trajectory) == 0:
        return 0.0
    distances = compute_joint_distances(trajectory, **kwargs)
    return distances.sum().item()


def compute_ee_distance(cu_world: World, trajectory: Optional[JointState], **kwargs) -> float:
    if trajectory is None:
        return INF
    if len(trajectory) == 0:
        return 0.0
    ee_poses = cu_world.get_ee_poses(confs=trajectory.position)
    differences = ee_poses.position[:-1, :] - ee_poses.position[1:, :]
    distances = torch.norm(differences, dim=1, **kwargs)
    return distances.sum().item()


def compute_sphere_distance(
    world: World, trajectory: Optional[JointState], weighted: bool = True, **kwargs: Any
) -> float:
    if trajectory is None:
        return INF
    if len(trajectory) == 0:
        return 0.0
    link_spheres = world.get_spheres(confs=trajectory.position)

    link_positions = link_spheres[..., :3]
    differences = link_positions[:-1, ...] - link_positions[1:, ..., :3]
    distances = torch.norm(differences, dim=-1, **kwargs).sum(dim=0)
    if not weighted:
        return distances.sum()

    radii = torch.mean(link_spheres[..., 3], dim=0)
    radii = torch.maximum(torch.zeros(radii.shape, device=world.device), radii)
    volumes = 4.0 / 3.0 * torch.pi * torch.pow(radii, exponent=3)

    weighted_distances = torch.mul(volumes, distances)
    return weighted_distances.sum()


def compute_linear_durations(
    world: World, joint_state: JointState, speed: Optional[float] = None
) -> torch.Tensor:
    if len(joint_state) == 0:
        return world.to_device([])
    if len(joint_state) == 1:
        return world.to_device([0.0])
    joints = joint_state.joint_names
    differences = joint_state.position[:-1, :] - joint_state.position[1:, :]
    velocities = world.extract_conf(world.velocity_limit, joints)
    if speed is not None:
        velocities *= speed
    durations = torch.max(torch.absolute(differences / velocities), dim=1).values
    return durations


def compute_linear_duration(world: World, trajectory: Optional[JointState], **kwargs: Any) -> float:
    if trajectory is None:
        return INF
    if len(trajectory) == 0:
        return 0.0
    durations = compute_linear_durations(world, trajectory, **kwargs)
    return durations.sum()


def dump_trajectory_statistics(world: World, trajectory: Optional[JointState]) -> None:
    print(
        f"Solved: {trajectory is not None} | "
        f"Length: {compute_trajectory_length(trajectory)} | "
        f"Duration: {compute_trajectory_duration(world, trajectory):.3f} sec | "
        f"Joint Distance: {compute_joint_distance(trajectory):.3f} rad | "
        f"EE Distance: {compute_ee_distance(world, trajectory):.3f} m | "
        f"Sphere Distance: {compute_sphere_distance(world, trajectory):.3f} m"
    )
