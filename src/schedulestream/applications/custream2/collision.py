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
from typing import Any, Dict, List, Optional, Tuple

# Third Party
import numpy as np
import torch
from curobo.types import JointState, Pose

# NVIDIA
from schedulestream.applications.custream2.command import Trajectory
from schedulestream.applications.custream2.placement import Placement, concatenate_placement_poses
from schedulestream.applications.custream2.state import Configuration
from schedulestream.applications.custream2.utils import (
    concatenate_joint_states,
    scene_sphere_contacts,
    transform_spheres,
)
from schedulestream.applications.custream2.world import World
from schedulestream.common.utils import batched, current_time, elapsed_time


def joint_state_collision(
    world: World,
    joint_state: JointState,
    scene_coll: Optional[Any] = None,
    scene_pose: Optional[Pose] = None,
    collision_distance: float = 0.0,
) -> torch.Tensor:
    """Return [N] bool tensor; True if any waypoint in that sequence collides.

    joint_state.position must have shape [N, T, dof].
    If scene_coll is None, uses world.scene_collision_checker.
    If scene_pose is provided, spheres are transformed into scene_pose's frame before querying.
    """
    if scene_coll is None:
        scene_coll = world.scene_collision_checker
    N, T, dof = joint_state.position.shape
    flat_js = JointState.from_position(
        joint_state.position.reshape(N * T, dof), joint_names=list(joint_state.joint_names)
    )
    flat_js = world.complete_joint_state(flat_js)
    full_js = world.trajopt_solver.get_full_js(flat_js).reorder(world.active_joints)
    kin_state = world.compute_kinematics(full_js)
    spheres = kin_state.robot_spheres
    if scene_pose is not None:
        NT, hz, S, _ = spheres.shape
        flat = spheres.reshape(1, NT * hz * S, 4).to(scene_pose.position.device)
        spheres = transform_spheres(scene_pose.inverse(), flat).reshape(NT * hz, 1, S, 4)
    distances, _ = scene_sphere_contacts(scene_coll, spheres, collision_distance=collision_distance)
    return (distances > 0.0).reshape(N, -1).any(dim=1)


def _asymmetric_pair_collision(
    world: World,
    obj1: str,
    placement1s: List[Any],
    obj2: str,
    placement2s: List[Any],
    collision_distance: float = 0.0,
    verbose: bool = True,
) -> List[bool]:
    """Check obj1 spheres against obj2's collision scene for each placement pair."""
    t0 = current_time()
    num = len(placement1s)
    if obj1 == obj2:
        return num * [False]
    body1 = world.get_object(obj1)
    body2 = world.get_object(obj2)
    if not body1.movable and body2.movable:
        return num * [False]
    pose1 = concatenate_placement_poses(placement1s)
    scene_pose = concatenate_placement_poses(placement2s)
    spheres_world = transform_spheres(pose1, body1.spheres_tensor)
    spheres_obj2 = transform_spheres(scene_pose.inverse(), spheres_world)
    distances = body2.get_sphere_distances(spheres_obj2, collision_distance=collision_distance)
    collisions = (distances.amax(dim=-1) > 0.0).tolist()
    if verbose and any(collisions):
        print(
            f"object_object_collision) Object 1: {obj1} | Object 2: {obj2} | Colliding:"
            f" {sum(collisions)}/{num} | Elapsed: {elapsed_time(t0):.3f} sec"
        )
    return collisions


def asymmetric_object_object_collision(
    world: World,
    batch_inputs: List[Tuple[str, Any, str, Any]],
    collision_distance: float = 0.0,
    verbose: bool = False,
) -> List[bool]:
    """Check obj1-sphere vs obj2-collision-scene; True where obj1 spheres penetrate obj2."""
    num = len(batch_inputs)
    if num == 0:
        return []
    obj1s, placement1s, obj2s, placement2s = zip(*batch_inputs)
    results = [False] * num
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, (o1, p1, o2, p2) in enumerate(batch_inputs):
        if not all(isinstance(p, Placement) for p in (p1, p2)):
            continue
        groups.setdefault((o1, o2), []).append(i)
    for (obj1, obj2), indices in groups.items():
        sub_results = _asymmetric_pair_collision(
            world,
            obj1,
            [placement1s[i] for i in indices],
            obj2,
            [placement2s[i] for i in indices],
            collision_distance=collision_distance,
            verbose=verbose,
        )
        for i, r in zip(indices, sub_results):
            results[i] = r
    return results


def object_object_collision(
    world: World,
    batch_inputs: List[Tuple[str, Any, str, Any]],
    collision_distance: float = 0.0,
    verbose: bool = True,
) -> List[bool]:
    """Return symmetric collision results by ORing both directional checks."""
    collisions1 = asymmetric_object_object_collision(
        world, batch_inputs, collision_distance=collision_distance, verbose=verbose
    )
    reverse = [(obj2, p2, obj1, p1) for obj1, p1, obj2, p2 in batch_inputs]
    collisions2 = asymmetric_object_object_collision(
        world, reverse, collision_distance=collision_distance, verbose=verbose
    )
    return np.logical_or(collisions1, collisions2).tolist()


def commands_placement_collision(
    world: World,
    batch_inputs: List[Tuple[str, Any, str, Any]],
    collision_distance: float = 0.0,
    max_waypoints: int = 16384,
) -> List[bool]:
    """Batched trajectory-vs-placement collision over (arm, commands, obj, placement) tuples.

    Each ``commands``' designated ``collider`` trajectory is checked against
    ``obj``'s collision scene at ``placement``; inputs without a collider (or
    without a Placement) are False. All inputs must share the same ``arm`` and
    ``obj`` (matching the batched stream convention), so every waypoint chunk
    costs a single kinematics pass and collision query.
    """
    if not batch_inputs:
        return []
    arms, all_commands, objs, placements = zip(*batch_inputs)
    [arm] = set(arms)
    [obj] = set(objs)
    body = world.get_object(obj)

    entries: List[Tuple[int, Trajectory, Placement]] = []
    for i, (commands, placement) in enumerate(zip(all_commands, placements)):
        collider = getattr(commands, "collider", None)
        if body.virtual or (collider is None) or not isinstance(placement, Placement):
            continue
        assert isinstance(collider, Trajectory), collider
        if obj in collider.grasped:
            continue
        entries.append((i, collider, placement))
    results = [False] * len(batch_inputs)
    if not entries:
        return results
    [joints] = {tuple(collider.joints) for _, collider, _ in entries}

    max_length = max(collider.length for _, collider, _ in entries)
    for chunk in batched(entries, max(1, max_waypoints // max_length)):
        positions = torch.stack(
            [
                torch.cat([c.positions, c.positions[-1:].expand(max_length - c.length, -1)])
                for _, c, _ in chunk
            ]
        )
        n, T, dof = positions.shape
        flat_js = world.complete_joint_state(
            JointState.from_position(positions.reshape(n * T, dof), joint_names=list(joints))
        )
        full_js = world.trajopt_solver.get_full_js(flat_js).reorder(world.active_joints)
        spheres = world.compute_kinematics(full_js).robot_spheres
        inverse_poses = concatenate_placement_poses(
            [placement for _, _, placement in chunk]
        ).inverse()
        rows = transform_spheres(inverse_poses, spheres.reshape(n, -1, 4))
        distances = body.get_sphere_distances(rows, collision_distance=collision_distance)
        chunk_collides = (distances > 0.0).reshape(n, -1).any(dim=-1)
        for (i, _, _), collided in zip(chunk, chunk_collides.tolist()):
            results[i] = collided
    return results


def joint_state_pair_collision(
    world: World,
    arm1: str,
    joint_state1: JointState,
    arm2: str,
    joint_state2: JointState,
    collision_distance: float = 0.0,
    verbose: bool = False,
) -> List[Tuple[int, int]]:
    """Return (i, j) pairs where arm1 at joint_state1[i] sphere-collides with arm2 at joint_state2[j].
    """
    arm1_joints = world.get_arm_joints(arm1)
    arm2_joints = world.get_arm_joints(arm2)
    N1 = joint_state1.position.shape[0]
    N2 = joint_state2.position.shape[0]

    all_joints = world.active_joints
    current = np.array(world.get_joint_positions(world.active_joints), dtype=np.float32)
    full = (
        torch.tensor(current, dtype=torch.float32, device=world.device)
        .unsqueeze(0)
        .expand(N1 * N2, -1)
        .clone()
    )
    jnames1 = list(joint_state1.joint_names)
    jnames2 = list(joint_state2.joint_names)
    pos1 = joint_state1.position.to(world.device)
    pos2 = joint_state2.position.to(world.device)
    arm1_idx = [all_joints.index(j) for j in arm1_joints if j in all_joints]
    arm2_idx = [all_joints.index(j) for j in arm2_joints if j in all_joints]
    src1_idx = [jnames1.index(j) for j in arm1_joints if j in all_joints]
    src2_idx = [jnames2.index(j) for j in arm2_joints if j in all_joints]
    full[:, arm1_idx] = pos1[:, src1_idx].repeat_interleave(N2, dim=0)
    full[:, arm2_idx] = pos2[:, src2_idx].repeat(N1, 1)

    full_js = JointState.from_position(full, joint_names=all_joints).reorder(world.active_joints)
    kin_state = world.compute_kinematics(full_js)
    if kin_state.robot_spheres is None:
        return []

    self_col_cost = world.self_collision_cost
    self_col_cost.setup_batch_tensors(N1 * N2, 1)
    d_self = self_col_cost.forward(kin_state.robot_spheres)
    pair_collides = d_self.view(-1) > -collision_distance

    colliding_pairs = [
        (flat // N2, flat % N2) for flat in torch.nonzero(pair_collides)[:, 0].tolist()
    ]
    if verbose:
        total = N1 * N2
        print(
            f"joint_state_pair_collision) {arm1} ({N1}) x {arm2} ({N2})"
            f" | colliding: {len(colliding_pairs)}/{total} ({len(colliding_pairs)/total:.1%})"
        )
    return colliding_pairs


def conf_pair_collision(
    world: World,
    arm1: str,
    arm2: str,
    confs1: List[Configuration],
    confs2: List[Configuration],
    collision_distance: float = 0.0,
    verbose: bool = True,
    debug: bool = False,
) -> List[Tuple[int, int]]:
    """Return (i, j) pairs where arm1 at confs1[i] sphere-collides with arm2 at confs2[j]."""
    if not confs1 or not confs2:
        return []
    t = current_time()
    js1 = concatenate_joint_states(c.joint_state for c in confs1)
    js2 = concatenate_joint_states(c.joint_state for c in confs2)
    colliding_pairs = joint_state_pair_collision(
        world, arm1, js1, arm2, js2, collision_distance=collision_distance
    )
    if verbose:
        total = len(confs1) * len(confs2)
        print(
            f"conf_pair_collision) {arm1} ({len(confs1)}) x {arm2} ({len(confs2)})"
            f" | colliding: {len(colliding_pairs)}/{total} ({len(colliding_pairs)/total:.1%})"
            f" | {elapsed_time(t):.3f}s"
        )
    if debug:
        arm1_joints = world.get_arm_joints(arm1)
        arm2_joints = world.get_arm_joints(arm2)
        for i, j in colliding_pairs:
            world.set_joint_positions(
                arm1_joints, confs1[i].joint_state.position.squeeze(0).tolist()
            )
            world.set_joint_positions(
                arm2_joints, confs2[j].joint_state.position.squeeze(0).tolist()
            )
            world.show()
    return colliding_pairs


def commands_conf_collision(
    world: World,
    arm1: str,
    commands1: Any,
    arm2: str,
    conf2: Any,
    collision_distance: float = 0.0,
    configurations: bool = True,
) -> bool:
    """Check arm1's collider trajectory against arm2 at ``conf2``.

    ``conf2`` may be a Configuration or another Commands/Trajectory (checked
    via its collider). Returns False when either collider is missing, or when
    ``conf2`` is a Configuration and ``configurations`` is False.
    """
    if (arm1 == arm2) or (conf2 is None):
        return False
    collider1 = getattr(commands1, "collider", None)
    if collider1 is None:
        return False
    if isinstance(conf2, Configuration):
        if not configurations:
            return False
        joint_state2 = conf2.joint_state
    else:
        collider2 = getattr(conf2, "collider", None)
        if collider2 is None:
            return False
        joint_state2 = collider2.joint_state
    colliding_pairs = joint_state_pair_collision(
        world,
        arm1,
        collider1.joint_state,
        arm2,
        joint_state2,
        collision_distance=collision_distance,
    )
    return bool(colliding_pairs)
