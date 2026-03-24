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
from typing import Any, Iterator, Optional, Tuple

# Third Party
import numpy as np
from cumotion import Pose3
from numpy import floating

# NVIDIA
from schedulestream.applications.cumotion.command import (
    Attach,
    Commands,
    Conf,
    Detach,
    Grasp,
    Path,
    Placement,
)
from schedulestream.applications.cumotion.objects import Object
from schedulestream.applications.cumotion.utils import create_pose
from schedulestream.applications.cumotion.world import World
from schedulestream.language.stream import Context


def grasp_sampler(obj: Object) -> Iterator[Optional[Tuple[Grasp]]]:
    for pose in obj.grasps():
        pose = create_pose(position=[0, -0.03, 0]) * pose
        outputs = (Grasp(pose),)
        yield outputs


def placement_sampler(
    world: World,
    obj: Object,
    support: Object,
    support_pose: Optional[Pose3] = None,
    collisions: bool = True,
    **kwargs: Any,
) -> Iterator[Optional[Tuple[Placement]]]:
    if support_pose is None:
        support_pose = world.get_pose(support)
    obstacles = set(world.fixed_objects) - {obj, support}
    if not collisions:
        obstacles.clear()
    for placement in obj.placements(support):
        pose = support_pose * placement
        if any(
            world.object_object_collision(obj, obstacle, pose, **kwargs) for obstacle in obstacles
        ):
            yield None
        else:
            outputs = (Placement(pose),)
            yield outputs


def inverse_kinematics_sampler(
    world: World,
    obj: Object,
    grasp: Pose3,
    placement: Pose3,
    object_z: float = 2e-2,
    robot_z: float = 3e-2,
    collisions: bool = True,
    **kwargs: Any,
) -> Iterator[Optional[Tuple[Conf]]]:
    contact_pose = placement * grasp
    approach_pose = (
        placement * create_pose([0, 0, object_z]) * grasp * create_pose([0, 0, -robot_z])
    )
    poses = [approach_pose, contact_pose]
    world.set_enabled_objects(world.fixed_objects if collisions else [])
    enabled_context = world.enabled_context()
    while True:
        path = world.iterative_inverse_kinematics(poses, **kwargs)
        if path is None:
            yield None
        else:
            path = world.interpolate_path(path)
            outputs = (Path(path),)
            yield outputs
            break
        enabled_context.set()


def pick_sampler(
    world: World,
    obj: Object,
    grasp: Pose3,
    placement: Pose3,
    tool_frame: Optional[str] = None,
    **kwargs: Any,
) -> Iterator[Optional[Tuple[Conf, Commands, Conf]]]:
    if tool_frame is None:
        tool_frame = world.tool_frames[0]
    for outputs in inverse_kinematics_sampler(
        world, obj, grasp, placement, frame=tool_frame, **kwargs
    ):
        if outputs is None:
            yield None
        else:
            (path,) = outputs
            command = Commands([path, Attach(obj, tool_frame), path.reverse()])
            outputs = (path.start(), path.start(), command)
            yield outputs


def place_sampler(
    world: World,
    obj: Object,
    grasp: Pose3,
    placement: Pose3,
    tool_frame: Optional[str] = None,
    **kwargs: Any,
) -> Iterator[Optional[Tuple[Conf, Commands, Conf]]]:
    if tool_frame is None:
        tool_frame = world.tool_frames[0]
    for outputs in inverse_kinematics_sampler(
        world, obj, grasp, placement, frame=tool_frame, **kwargs
    ):
        if outputs is None:
            yield None
        else:
            (path,) = outputs
            command = Commands([path, Detach(obj), path.reverse()])
            outputs = (path.start(), path.start(), command)
            yield outputs


def motion_sampler(
    world: World,
    conf1: np.ndarray,
    conf2: np.ndarray,
    context: Optional[Context] = None,
    collisions: bool = True,
    **kwargs: Any,
) -> Iterator[Optional[Tuple[Path]]]:
    state_context = world.state_context()
    obstacles = world.fixed_objects
    if context is not None:
        for constraint in context.constraints:
            _, obj, pose = constraint.expression.term.unwrap_arguments()
            if pose is None:
                continue
            if isinstance(pose, Grasp):
                pass
            elif isinstance(pose, Placement):
                world.set_object_pose(obj, pose)
                obstacles.append(obj)
            else:
                raise NotImplementedError(pose)

    world.set_enabled_objects(obstacles if collisions else [])
    path = world.plan_to_conf(start_conf=conf1, goal_conf=conf2, **kwargs)
    state_context.set()
    if path is None:
        return
    outputs = (Path(path),)
    yield outputs


def distance_cost(conf1: np.ndarray, conf2: np.ndarray) -> floating[Any]:
    return np.linalg.norm(np.array(conf2) - np.array(conf1))
