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
import math
from itertools import count
from typing import Any, Iterator, List, Optional, Tuple

# Third Party
import numpy as np
import torch
import trimesh
from curobo.types.math import Pose
from curobo.types.state import JointState

# NVIDIA
from schedulestream.applications.blocksworld.visualize import linear_curve
from schedulestream.applications.custream.command import (
    ArmPath,
    Attach,
    Close,
    Commands,
    Detach,
    Open,
    Trajectory,
)
from schedulestream.applications.custream.grasp import Grasp, grasp_generator
from schedulestream.applications.custream.metrics import (
    compute_joint_distance,
    compute_linear_duration,
    compute_linear_durations,
)
from schedulestream.applications.custream.placement import Placement, placement_generator
from schedulestream.applications.custream.state import Configuration, State
from schedulestream.applications.custream.utils import (
    concatenate_joint_states,
    create_pose,
    extract_joint_state,
    interpolate_poses,
    multiply_poses,
    to_cpu,
    transform_spheres,
)
from schedulestream.applications.custream.world import World
from schedulestream.common.utils import (
    EPSILON,
    INF,
    batched,
    current_time,
    elapsed_time,
    safe_zip,
    take,
)
from schedulestream.language.generator import Output, Pair
from schedulestream.language.stream import Context


def grasp_stream(
    world: World, obj: str, arm: Optional[str] = None, **kwargs: Any
) -> Iterator[Grasp]:
    return grasp_generator(world, obj, arm=arm, **kwargs)


def placement_stream(
    world: World,
    obj1: str,
    obj2: str,
    placement2: Placement,
    batch_size: int = 1000,
    object_proximity: Optional[float] = 0.0,
    robot_proximity: Optional[float] = 0.1,
    display: bool = False,
    debug: bool = False,
    **kwargs: Any,
) -> Iterator[Placement]:
    joint_state = world.to_joint_state(world.retract_conf.unsqueeze(0))
    conf = Configuration(world, joint_state)

    fixed_names = set(world.obstacle_names) - {obj1, obj2} - set(placement2.objects)
    body1 = world.get_object(obj1)
    spheres_tensor = body1.get_spheres_tensor()
    world_pose2 = placement2.get_pose()
    local_placement_batches = batched(
        placement_generator(world, obj1, obj2, batch_size=batch_size, **kwargs), batch_size
    )
    for i in count():
        start_time = current_time()
        local_placement_batch = next(local_placement_batches, None)
        if local_placement_batch is None:
            break
        local_poses1 = Pose.cat([p.pose for p in local_placement_batch])
        world_poses1 = multiply_poses(world_pose2, local_poses1)
        valid = torch.full([len(local_poses1)], True, device=world.device)

        if object_proximity is not None:
            world_spheres1 = transform_spheres(world_poses1, spheres_tensor)
            with world.active_context(objects=None):
                world.enable_objects_active(fixed_names)
                distances = torch.min(-world.get_sphere_distances(world_spheres1), dim=1).values
                valid &= distances >= object_proximity

        indices = to_cpu(torch.nonzero(valid)[:, 0])
        print(
            f"{placement_stream.__name__}) Object: {obj1} | Parent: {obj2} | Iteration: {i} |"
            f" Success: {len(indices)}/{len(valid)} | Rate:"
            f" {elapsed_time(start_time)/len(valid):.3e} | Elapsed:"
            f" {elapsed_time(start_time):.3f} sec"
        )
        if display:
            scene = world.objects_scene(names=fixed_names)
            points = to_cpu(world_poses1.position)
            colors = np.full([len(points), 3], [255, 0, 0])
            colors[indices] = [0, 255, 0]
            scene.add_geometry(trimesh.PointCloud(points, colors))
            scene.show()

        for index in indices:
            local_placement = local_placement_batch[index]
            world_placement = local_placement.stack(placement2)
            if debug:
                world_placement.scene().show()
            yield world_placement


def compute_approach(
    world: World, obj: str, grasp: Grasp, placement: Placement, arm: str, debug: bool = False
) -> ArmPath:
    tamp_config = world.tamp_config
    obj_pose = placement.get_pose()
    contact_pose = grasp.get_link_pose(obj_pose)
    approach_pose = grasp.get_link_pose(
        obj_pose,
        link_distance=tamp_config.approach_link_distance,
        object_distance=tamp_config.approach_object_distance,
    )
    tool_pose = Pose.cat(
        list(
            interpolate_poses(
                approach_pose,
                contact_pose,
                pos_step=tamp_config.approach_position_step,
                ori_step=tamp_config.approach_orientation_step,
            )
        )
    )

    state = State(world, attachments=[grasp])
    arm_path = ArmPath(world, arm, tool_pose, state=state)
    if debug:
        arm_path.scene().show()
    return arm_path


def interpolate_joint_state(
    world: World,
    joint_state: JointState,
    min_duration: Optional[float] = EPSILON,
    **kwargs: Any,
) -> JointState:
    if len(joint_state) <= 1:
        return joint_state
    durations = to_cpu(compute_linear_durations(world, joint_state, **kwargs))
    if min_duration is not None:
        durations = np.maximum(min_duration * np.ones(len(durations)), durations)

    times = np.cumsum([0.0] + list(durations))
    curve = linear_curve(times, to_cpu(joint_state.position))
    steps = int(math.ceil(times[-1] / world.time_step)) + 1
    samples = np.linspace(times[0], times[-1], num=steps, endpoint=True)
    positions = list(map(curve, samples))
    return world.to_joint_state(positions, joint_state.joint_names)


def interpolate_trajectory(traj: Trajectory, speed: Optional[float] = None) -> Trajectory:
    world = traj.world
    if speed is None:
        speed = world.tamp_config.approach_velocity_scale
    joint_state = interpolate_joint_state(world, traj.joint_state, speed=speed)
    return Trajectory(world, joint_state, state=traj.state)


def constrained_motion(
    world: World,
    arm: str,
    arm_paths: List[ArmPath],
    interpolate: bool = True,
    max_distance: float = INF,
    sort: bool = True,
    num: Optional[int] = None,
    **kwargs: Any,
) -> List[List[Trajectory]]:
    assert len(arm_paths) <= world.ik_batch
    tool_link = world.get_arm_link(arm)
    pose_paths = [list(arm_path) for arm_path in arm_paths]
    length = max(map(len, pose_paths))
    batch_poses = [
        Pose.cat([pose_path[min(i, len(pose_path) - 1)] for pose_path in pose_paths]).clone()
        for i in range(length)
    ]
    batch_joint_state, batch_distances = world.link_iterative_inverse_kinematics(
        link=tool_link, poses_list=batch_poses, **kwargs
    )
    batch_valid = batch_distances < max_distance
    batch_distances = world.joint_state_distances(world.retract_conf, batch_joint_state[:, :, 0, :])
    batch_distances = torch.where(
        batch_valid,
        batch_distances,
        torch.full(batch_distances.shape, torch.inf, device=world.device),
    )
    indices = torch.sort(batch_distances[:, 0])

    batch_trajectories = [[] for _ in range(len(arm_paths))]
    for batch, arm_path in enumerate(arm_paths):
        joint_state = batch_joint_state[batch]
        distances = batch_distances[batch]
        valid = batch_valid[batch]
        if not valid.any():
            continue
        joint_state = joint_state[valid, ...]
        distances = distances[valid, ...]

        if sort:
            distances = world.joint_state_distances(world.retract_conf, joint_state[:, 0, :])
        distances, indices = torch.sort(distances, dim=-1)

        for index in take(indices, num):
            index_state = joint_state[index, : len(arm_path)]
            traj = world.trajectory(index_state, state=None)
            if interpolate:
                traj = interpolate_trajectory(traj)
            batch_trajectories[batch].append(traj)
    return batch_trajectories


def ik_stream(
    world: World, arm: str, obj: str, grasp: Grasp, placement: Placement, **kwargs: Any
) -> Iterator[Configuration]:
    arm_pose = multiply_poses(placement.get_pose(), grasp.pose)
    arm_path = ArmPath(world, arm, arm_pose)
    [trajectories] = constrained_motion(world, arm, [arm_path], num=1, **kwargs)
    if not trajectories:
        return
    traj = trajectories[0]
    state = State(world, attachments=[grasp])
    conf = traj.configuration(index=0, state=state)
    yield conf


def pick_stream(
    world: World,
    batch_inputs: List[Tuple[str, Grasp, Placement, str]],
    rest: bool = False,
    collisions: bool = True,
    **kwargs: Any,
) -> List[Pair]:
    objs, grasps, placements, arms = safe_zip(*batch_inputs)
    [obj] = set(objs)
    [arm] = set(arms)
    tool_link = world.get_arm_link(arm)

    inverse_grasp_poses = Pose.cat([grasp.pose for grasp in grasps]).inverse()
    parent_pose = placements[0].get_parent_pose()
    placement_poses = multiply_poses(
        parent_pose, Pose.cat([placement.pose for placement in placements])
    )

    tamp_config = world.tamp_config
    approach_distance = tamp_config.approach_link_distance + tamp_config.approach_object_distance
    num_steps = int(math.ceil(approach_distance)) + 1
    link_poses_list = []
    for weight in np.linspace(start=1.0, stop=0.0, num=num_steps, endpoint=True):
        approach_object = create_pose(z=weight * tamp_config.approach_object_distance)
        approach_link = create_pose(z=weight * -tamp_config.approach_link_distance)
        link_poses = multiply_poses(
            placement_poses, approach_object, inverse_grasp_poses, approach_link
        )
        link_poses_list.append(link_poses)

    arm_paths = []
    for i, grasp in enumerate(grasps):
        tool_pose = Pose.cat([link_poses[i] for link_poses in link_poses_list]).clone()
        state = State(world, attachments=[grasp])
        arm_path = ArmPath(world, arm, tool_pose, state=state)
        arm_paths.append(arm_path)

    with world.active_context(objects=None):
        world.enable_objects_active(world.obstacle_names if collisions else [])
        batch_trajectories = constrained_motion(world, arm, arm_paths, num=1, **kwargs)
    assert len(batch_trajectories) == len(arm_paths)

    batch_pairs = []
    for batch, trajectories in enumerate(batch_trajectories):
        if not trajectories:
            continue
        start_state = State(world, attachments=[placements[batch]])
        end_state = State(world, attachments=[grasps[batch]])
        for traj in trajectories:
            start_conf = traj.configuration(index=0)
            end_conf = traj.configuration(index=0, state=end_state)
            contact_conf = traj.configuration(index=-1)
            commands = [
                Open(world, arm),
                traj.clone(state=start_state),
                Close(world, arm),
                Attach(world, obj, tool_link),
                traj.reverse(state=end_state),
            ]
            if rest:
                commands.insert(2, contact_conf.rest_trajectory(steps=20, state=start_state))
                commands.insert(-1, contact_conf.rest_trajectory(steps=20, state=end_state))
            command = Commands(world, commands)
            output = Output(start_conf, end_conf, command)
            batch_pairs.append((batch_inputs[batch], output))
    batch_pairs.sort(key=lambda pair: compute_retract_duration(world, pair[1][0].joint_state))
    return batch_pairs


def place_stream(
    world: World,
    batch_inputs: List[Tuple[str, Grasp, Placement, str]],
    rest: bool = False,
    collisions: bool = True,
    **kwargs: Any,
) -> List[Pair]:
    objs, grasps, placements, arms = safe_zip(*batch_inputs)
    [obj] = set(objs)
    [arm] = set(arms)
    tool_link = world.get_arm_link(arm)
    arm_paths = [
        compute_approach(world, obj, grasp, placement, arm, **kwargs)
        for grasp, placement in safe_zip(grasps, placements)
    ]

    with world.active_context(objects=None):
        world.enable_objects_active(world.obstacle_names if collisions else [])
        batch_trajectories = constrained_motion(world, arm, arm_paths, num=1, **kwargs)
    assert len(batch_trajectories) == len(arm_paths)

    batch_pairs = []
    for batch, trajectories in enumerate(batch_trajectories):
        if not trajectories:
            continue
        start_state = State(world, attachments=[grasps[batch]])
        end_state = State(world, attachments=[placements[batch]])
        for traj in trajectories:
            start_conf = traj.configuration(index=0, state=start_state)
            end_conf = traj.configuration(index=0)
            contact_conf = traj.configuration(index=-1)
            commands = [
                traj.clone(state=start_state),
                Detach(world, tool_link),
                Open(world, arm),
                traj.reverse(state=end_state),
            ]
            if rest:
                commands.insert(1, contact_conf.rest_trajectory(steps=20, state=start_state))
                commands.insert(-1, contact_conf.rest_trajectory(steps=20, state=end_state))
            command = Commands(world, commands)
            output = Output(start_conf, end_conf, command)
            batch_pairs.append((batch_inputs[batch], output))
    batch_pairs.sort(key=lambda pair: compute_retract_duration(world, pair[1][0].joint_state))
    return batch_pairs


def motion_stream(
    world: World,
    arm: str,
    conf1: Configuration,
    conf2: Configuration,
    grasp: Optional[Grasp] = None,
    context: Optional[Context] = None,
    obstacles: Optional[List[str]] = None,
    linear: bool = False,
    collisions: bool = True,
) -> Optional[Trajectory]:
    if grasp is None:
        grasps = set(conf1.grasps) & set(conf2.grasps)
        if grasps:
            [grasp] = grasps
    state = State(world, attachments=[grasp] if grasp else [])
    if obstacles is None:
        obstacles = world.obstacle_names

    if context is not None:
        for constraint in context.constraints:
            _, _, obj, pose = constraint.expression.term.unwrap_arguments()
            if pose is None:
                continue
            if isinstance(pose, Placement):
                obstacles.append(obj)
                pose.set()
            elif isinstance(pose, Grasp):
                grasp = pose
            else:
                raise NotImplementedError(pose)

    if linear:
        joint_state = concatenate_joint_states([conf1.joint_state, conf2.joint_state])
        joint_state = interpolate_joint_state(world, joint_state)
        return world.trajectory(joint_state, state=state)

    with world.active_context(objects=None):
        if grasp is not None:
            grasp.attach()

        world.enable_objects_active(obstacles if collisions else [])
        joint_state = world.plan_motion(
            start_state=conf1.joint_state,
            goal_state=conf2.joint_state,
        )
        if grasp is not None:
            grasp.detach()

    if joint_state is None:
        return None
    return world.trajectory(joint_state, state=state)


def distance_function(arm: str, conf1: Configuration, conf2: Configuration) -> float:
    joint_state = concatenate_joint_states([conf1.joint_state, conf2.joint_state])
    return compute_joint_distance(joint_state)


def duration_function(
    world: World, arm: str, conf1: Configuration, conf2: Configuration, **kwargs: Any
) -> float:
    joint_state = concatenate_joint_states([conf1.joint_state, conf2.joint_state])
    return compute_linear_duration(world, joint_state, **kwargs)


def compute_retract_duration(world: World, joint_state: JointState) -> float:
    joints = joint_state.joint_names
    retract_joint_state = world.to_joint_state(confs=world.retract_conf.unsqueeze(0))
    retract_joint_state = extract_joint_state(retract_joint_state, joints)
    initial_joint_state = concatenate_joint_states([retract_joint_state, joint_state])
    initial_joint_state = interpolate_joint_state(world, initial_joint_state)
    initial_trajectory = world.trajectory(initial_joint_state)
    return initial_trajectory.duration
