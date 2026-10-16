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
from typing import Any, List, Optional, Tuple

# Third Party
import numpy as np
import torch
from curobo.types import GoalToolPose, JointState, Pose

# NVIDIA
from schedulestream.applications.custream2.collision import joint_state_collision
from schedulestream.applications.custream2.command import (
    Attach,
    Close,
    Commands,
    Detach,
    HandCommand,
    Open,
    Trajectory,
)
from schedulestream.applications.custream2.grasp import Grasp
from schedulestream.applications.custream2.placement import Placement
from schedulestream.applications.custream2.retime import retime_waypoints_batch
from schedulestream.applications.custream2.state import Configuration
from schedulestream.applications.custream2.utils import (
    INF,
    add_spheres,
    batch_pose,
    concatenate_poses,
    create_pose,
    expand_joint_state,
    extract_joint_state,
    interpolate_joint_states,
    multiply_poses,
    stack_joint_states,
)
from schedulestream.applications.custream2.world import World
from schedulestream.common.utils import current_time, elapsed_time, safe_zip
from schedulestream.language.generator import Output, Pair

INTERPOLATION_STEPS = 10


def _get_goal_tool_pose(
    world: World, arm: str, ee_pose: Pose, tool_frames: Optional[List[str]] = None
) -> GoalToolPose:
    """Wrap an EE Pose into a GoalToolPose for the v2 planner.

    For multi-arm robots, other arms' current poses are included as secondary goals so the IK
    solver receives a goal for every tool frame. ``tool_frames`` restricts which of the world's
    tool frames get a secondary goal -- pass e.g. ``ik_solver.kinematics.tool_frames`` when
    ``ik_solver`` was built for just one arm (see ``World.get_arm_ik_solver``), so poses for
    tool frames it doesn't know about aren't included; defaults to every tool frame in the world.
    """
    if tool_frames is None:
        tool_frames = world.tool_frames
    tool_frame = world.get_tool_link(arm)
    ee_pose = batch_pose(ee_pose)
    poses_dict: dict = {tool_frame: ee_pose}
    if len(tool_frames) <= 1:
        return GoalToolPose.from_poses(poses_dict, ordered_tool_frames=[tool_frame])

    batch = ee_pose.position.shape[0]
    tool_poses = world.get_tool_poses()
    for other_link, other_pose in tool_poses.items():
        if (other_link in poses_dict) or (other_link not in tool_frames):
            continue
        poses_dict[other_link] = batch_pose(other_pose, batch=batch)
    return GoalToolPose.from_poses(poses_dict, ordered_tool_frames=list(poses_dict))


def _solve_ik_sequence(
    world: World,
    arm: str,
    ee_poses: List[Pose],
    current_js: Optional[JointState] = None,
    p: float = INF,
    max_distance: float = INF,
    debug: bool = False,
) -> Tuple[List[JointState], torch.Tensor]:
    """Iteratively solve IK for a sequence of batched poses.

    Returns (joint_states, success_mask) where joint_states[i] holds all batch results at step i
    and success_mask[j] is True if batch element j succeeded every step.
    Seeds are sorted by max inter-step joint distance (p-norm) so the smoothest
    trajectory comes first, matching iterative_inverse_kinematics in custream. A seed whose
    distance exceeds ``max_distance`` fails: its jump between two waypoints, such as a wrist
    flip, would leave the path between them.
    """
    t0 = current_time()
    arm_joints = world.get_arm_joints(arm)
    if current_js is None:
        batch_size = ee_poses[0].position.shape[0]
        current_js = expand_joint_state(world.get_joint_state(arm_joints), batch_size)
    batch_size = current_js.position.shape[0]
    world_js = expand_joint_state(
        world.complete_joint_state(current_js),
        batch_size,
    )
    device = world_js.position.device
    world.set_arm_ik_weights(arm)
    all_arm_positions: List[torch.Tensor] = []
    all_seed_success: List[torch.Tensor] = []
    seed_config = None
    world_js.velocity = torch.zeros_like(world_js.position)
    world_js.acceleration = torch.zeros_like(world_js.position)
    sequence_failed = False
    for ee_pose in ee_poses:
        goal_tool_pose = _get_goal_tool_pose(world, arm, ee_pose)
        result = world.ik_solver.solve_pose(
            goal_tool_pose, world_js, seed_config=seed_config, return_seeds=world.ik_seeds
        )
        js = None if result is None else result.js_solution
        if result is None or js is None or not result.success.any():
            sequence_failed = True
            break
        succ = result.success
        if succ.ndim == 1:
            succ = succ.unsqueeze(-1)
        all_seed_success.append(succ)
        pos = js.position
        seed_pos = pos if pos.ndim == 3 else pos.unsqueeze(1)
        b, s, _ = seed_pos.shape
        flat_js = JointState.from_position(
            seed_pos.reshape(b * s, -1), joint_names=list(js.joint_names)
        )
        seed_config = world.ik_solver.get_active_js(flat_js).position.reshape(b, s, -1)
        arm_pos = extract_joint_state(flat_js, arm_joints).position.reshape(b, s, -1)
        all_arm_positions.append(arm_pos.clone())
    if not all_arm_positions:
        arm_idx = [world.active_joints.index(j) for j in arm_joints]
        dummy = world_js.position[:, arm_idx].unsqueeze(1)
        all_arm_positions.append(dummy)
        all_seed_success.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=device))

    batch = all_arm_positions[0].shape[0]
    n_seeds = all_arm_positions[0].shape[1]

    seed_success = all_seed_success[0].clone()
    for s_mask in all_seed_success[1:]:
        seed_success = seed_success & s_mask
    if sequence_failed:
        seed_success = torch.zeros_like(seed_success)

    max_dist = torch.zeros(batch, n_seeds, device=device)
    for t in range(len(all_arm_positions) - 1):
        diff = (all_arm_positions[t + 1] - all_arm_positions[t]).abs()
        dist = torch.linalg.vector_norm(diff, ord=p, dim=-1)
        max_dist = torch.max(max_dist, dist)
    seed_success = seed_success & (max_dist <= max_distance)
    max_dist = torch.where(seed_success, max_dist, torch.full_like(max_dist, INF))
    _, sort_idx = torch.sort(max_dist, dim=-1)

    success_mask = seed_success.any(dim=-1)

    joint_states = []
    best_idx = sort_idx[:, :1].unsqueeze(-1)
    for arm_pos in all_arm_positions:
        best = arm_pos.gather(1, best_idx.expand(-1, 1, arm_pos.shape[-1])).squeeze(1)
        joint_states.append(JointState.from_position(best, joint_names=arm_joints))
    solved_dists = max_dist.gather(1, sort_idx)[:, 0][success_mask]
    dist_range = (
        f" | Dist: [{solved_dists.min():.2f}, {solved_dists.max():.2f}]"
        if solved_dists.numel() > 0
        else ""
    )
    print(
        f"_solve_ik_sequence) Arm: {arm} | Solved: {int(success_mask.sum().item())}/{batch_size} |"
        f" Poses: {len(ee_poses)} | Seeds: {n_seeds}{dist_range} | Elapsed:"
        f" {elapsed_time(t0):.3f} sec"
    )

    if debug and not bool(success_mask.any().item()):
        spheres = world.get_hand_spheres(arm, ee_poses[-1])
        scene = world.active_scene()
        add_spheres(spheres, scene=scene)
        scene.show()

    return joint_states, success_mask


def _approach_poses_seq(
    world: World,
    grasps: List[Grasp],
    placements: List[Placement],
    num_steps: int = 2,
) -> List[Pose]:
    """Batched pose sequence from approach (weight=1) to contact (weight=0) for all pairs."""
    cfg = world.tamp_config
    pairs = safe_zip(grasps, placements)
    grasp_poses = concatenate_poses([grasp.pose for grasp, _ in pairs])
    obj_poses = concatenate_poses([placement.get_pose() for _, placement in pairs])
    seq = []
    for weight in np.linspace(1.0, 0.0, num_steps):
        approach_poses = multiply_poses(
            create_pose(z=weight * cfg.approach_link_distance),
            grasp_poses,
        )
        lifted_obj_poses = multiply_poses(
            create_pose(z=weight * cfg.approach_world_distance), obj_poses
        )
        seq.append(multiply_poses(lifted_obj_poses, approach_poses.inverse()))
    return seq


def num_hand_pose_collisions(
    world: World,
    arm: str,
    poses: List[Pose],
    collision_distance: float = 0.0,
) -> torch.Tensor:
    """Return ``[B]`` int tensor counting the hand spheres in collision with the
    world scene across each candidate's EE-pose sequence.

    Each element of ``poses`` is a batched Pose (one entry per candidate);
    counts sum over the sequence's poses and spheres, so 0 means the whole
    sequence is collision-free. Deactivated objects (e.g. the grasp target)
    are excluded by the scene checker as usual.
    """
    num_poses = len(poses)
    flat_poses = concatenate_poses(poses)
    spheres = world.get_hand_spheres(arm, flat_poses, inactive=False)
    distances = world.get_sphere_distances(spheres, collision_distance=collision_distance)
    batch = flat_poses.batch_size // num_poses
    collides = (distances > 0.0).reshape(num_poses, batch, -1)
    return collides.sum(dim=-1).sum(dim=0)


def solve_tool_paths(
    world: World,
    arm: str,
    tool_paths: List[Pose],
    collisions: bool = True,
    hand_collisions: Optional[str] = "none",
    disable_hand: bool = False,
    **kwargs: Any,
) -> Tuple[List[JointState], JointState, torch.Tensor]:
    """The pipeline shared by manipulation streams: check the hand along each candidate's tool
    path, solve IK along it, and check the interpolated joint path.

    ``tool_paths`` holds one batched Pose per waypoint, with a candidate per batch entry. Only
    static objects are checked; the transitions' tests check the movable ones. The hand check
    keeps the candidates whose hand is collision-free along the path (``"none"``), or those
    with the fewest collisions (``"fewest"``), when contact with a support is unavoidable;
    ``None`` skips it. ``disable_hand`` removes the hand's spheres from IK and the joint-path
    check, for paths that bring it into contact. Returns the IK waypoints ``[B, dof]`` per
    step, their interpolation ``[B, T, dof]`` and the ``[B]`` success mask.
    """
    inactive_names = set(world.movable_names) if collisions else set(world.object_names)
    hand_links = world.get_hand_links(arm)
    with world.active_context(objects=inactive_names), world.sphere_context(links=hand_links):
        world.set_objects_active(active=False, names=inactive_names)
        hand_mask = None
        if collisions and (hand_collisions is not None):
            finger_links = world.get_finger_links(arm)
            world.set_sphere_inflation(
                world.tamp_config.approach_sphere_inflation, links=finger_links
            )
            counts = num_hand_pose_collisions(world, arm, tool_paths)
            if hand_collisions == "none":
                hand_mask = counts == 0
            elif hand_collisions == "fewest":
                hand_mask = counts == counts.min()
            else:
                raise ValueError(hand_collisions)
        if disable_hand:
            world.set_sphere_inflation(-INF, links=hand_links)
        joint_states, success_mask = _solve_ik_sequence(world, arm, tool_paths, **kwargs)
        interpolated_states = interpolate_joint_states(
            joint_states, INTERPOLATION_STEPS, dt=world.time_step
        )
        if hand_mask is not None:
            success_mask = success_mask & hand_mask
        if collisions:
            success_mask = success_mask & ~joint_state_collision(world, interpolated_states)
    return joint_states, interpolated_states, success_mask


def pick_stream(
    world: World,
    batch_inputs: List[Tuple[str, Grasp, Placement, str]],
    collisions: bool = True,
    **kwargs: Any,
) -> List[Pair]:
    """Solve IK and build pick commands for a batch of (obj, grasp, placement, arm) inputs."""
    start_time = current_time()
    objs, grasps, placements, arms = safe_zip(*batch_inputs)
    [obj] = set(objs)
    [arm] = set(arms)
    tool_link = world.get_tool_link(arm)
    rest_steps = world.tamp_config.approach_rest_steps

    tool_paths = _approach_poses_seq(world, list(grasps), list(placements))
    joint_states, interpolated_states, success_mask = solve_tool_paths(
        world,
        arm,
        tool_paths,
        collisions=collisions,
        hand_collisions="fewest",
        disable_hand=True,
        **kwargs,
    )

    batch_pairs = []
    retimed_states = retime_waypoints_batch(
        world,
        stack_joint_states(joint_states),
        velocity_scale=world.tamp_config.approach_velocity_scale,
    )
    for batch_idx, (grasp, placement) in enumerate(zip(grasps, placements)):
        if not success_mask[batch_idx].item():
            continue
        traj = Trajectory(world, retimed_states[batch_idx], grasps=[], placement=placement)
        retreat = traj.reverse(grasps=[grasp], placement=placement.placement)
        contact_conf = traj.configuration(index=-1)
        start_conf = traj.configuration(index=0)
        end_conf = retreat.configuration(index=-1)
        rest_traj = contact_conf.rest_trajectory(steps=rest_steps) if rest_steps > 0 else None
        pick_cmds: List[Any] = [Open(world, arm), traj]
        if rest_traj is not None:
            pick_cmds.append(rest_traj)
        finger_conf = grasp.joint_state.position.detach().cpu().numpy().reshape(-1)
        pick_cmds += [HandCommand(world, arm, conf=finger_conf), Attach(world, obj, tool_link)]
        if rest_traj is not None:
            pick_cmds.append(rest_traj)
        pick_cmds.append(retreat)
        command = Commands(
            world,
            pick_cmds,
            collider=Trajectory(world, interpolated_states[batch_idx], grasps=[grasp]),
        )
        output = Output(start_conf, end_conf, command)
        batch_pairs.append((batch_inputs[batch_idx], output))
    print(
        f"{pick_stream.__name__}) Arm: {arm} | Object: {obj} |"
        f" Successes: {int(success_mask.sum().item())}/{len(grasps)} |"
        f" Elapsed: {elapsed_time(start_time):.3f} sec"
    )
    return batch_pairs


def place_stream(
    world: World,
    batch_inputs: List[Tuple[str, Grasp, Placement, str]],
    collisions: bool = True,
    **kwargs: Any,
) -> List[Pair]:
    """Solve IK and build place commands for a batch of (obj, grasp, placement, arm) inputs."""
    start_time = current_time()
    objs, grasps, placements, arms = safe_zip(*batch_inputs)
    [obj] = set(objs)
    [arm] = set(arms)
    tool_link = world.get_tool_link(arm)
    rest_steps = world.tamp_config.approach_rest_steps

    tool_paths = _approach_poses_seq(world, list(grasps), list(placements))
    joint_states, interpolated_states, success_mask = solve_tool_paths(
        world,
        arm,
        tool_paths,
        collisions=collisions,
        hand_collisions="none",
        disable_hand=True,
        **kwargs,
    )

    batch_pairs = []
    retimed_states = retime_waypoints_batch(
        world,
        stack_joint_states(joint_states),
        velocity_scale=world.tamp_config.approach_velocity_scale,
    )
    for batch_idx, (grasp, placement) in enumerate(zip(grasps, placements)):
        if not success_mask[batch_idx].item():
            continue
        traj = Trajectory(
            world, retimed_states[batch_idx], grasps=[grasp], placement=placement.placement
        )
        retreat = traj.reverse(grasps=[], placement=placement)
        contact_conf = traj.configuration(index=-1)
        start_conf = Configuration(world, traj.configuration(index=0).joint_state, grasps=[grasp])
        end_conf = traj.configuration(index=0)
        rest_traj = contact_conf.rest_trajectory(steps=rest_steps) if rest_steps > 0 else None
        place_cmds: List[Any] = [traj]
        if rest_traj is not None:
            place_cmds.append(rest_traj)
        place_cmds += [Detach(world, tool_link), Open(world, arm)]
        if rest_traj is not None:
            place_cmds.append(rest_traj)
        place_cmds.append(retreat)
        command = Commands(
            world,
            place_cmds,
            collider=Trajectory(world, interpolated_states[batch_idx], grasps=[grasp]),
        )
        output = Output(start_conf, end_conf, command)
        batch_pairs.append((batch_inputs[batch_idx], output))
    print(
        f"{place_stream.__name__}) Arm: {arm} | Object: {obj} |"
        f" Successes: {int(success_mask.sum().item())}/{len(grasps)} |"
        f" Elapsed: {elapsed_time(start_time):.3f} sec"
    )
    return batch_pairs
