#!/usr/bin/env python3
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
import math
from typing import Any, List, Optional

# Third Party
import numpy as np
import torch
from curobo.types import JointState, Pose

# NVIDIA
from schedulestream.applications.custream2.asset import ASSETS
from schedulestream.applications.custream2.franka import load_franka_config
from schedulestream.applications.custream2.object import (
    CuboidObject,
    GraspConfig,
    Object,
    SceneObject,
    SurfaceConfig,
    create_primitive_objects,
)
from schedulestream.applications.custream2.utils import (
    PI,
    create_pose,
    interpolate_joint_states,
    multiply_poses,
    set_seed,
)
from schedulestream.applications.custream2.world import World
from schedulestream.applications.custream2.ycb import create_ycb_objects
from schedulestream.common.utils import current_time, elapsed_time

CAMERA_POSE = create_pose(roll=3 * np.pi / 8)

FRANKA_POSITIONS = np.pi * np.array([0, -1.0 / 4, 0, -2.0 / 4, 0, 2.0 / 4, 1.0 / 4])

PRIMITIVES = ["cuboid", "sphere", "cylinder", "capsule", "ycb"]


def load_robot_config(
    robot: str = "franka",
    base_poses: Optional[List[Pose]] = None,
    **kwargs: Any,
) -> dict:
    """Dispatch to the appropriate ``load_*_config`` for the named robot.

    Supported ``robot`` values:
      * ``"droid"`` — Franka Panda + Robotiq 2F-85 (not in the public release); calls
        :func:`load_droid_config`. ``base_poses`` must be ``None``; the
        droid loader does not support multi-robot base placement.
      * ``"franka"`` (default) — Franka Panda with the stock gripper; calls
        :func:`load_franka_config(base_poses=...)`.

    Extra ``**kwargs`` are forwarded to the chosen loader.
    """
    key = robot.lower()
    if key == "franka":
        return load_franka_config(base_poses=base_poses, **kwargs)
    if key == "droid":
        assert (
            not base_poses
        ), f"load_robot_config: 'droid' does not support base_poses (got {base_poses!r})"
        # NVIDIA
        from schedulestream.applications.custream2.internal.droid import load_droid_config

        return load_droid_config(**kwargs)
    raise NotImplementedError(f"load_robot_config: unsupported robot {robot!r}")


def create_objects(
    num_objects: int,
    primitive: str = "cuboid",
    width: float = 0.05,
    parent: Optional[Object] = None,
    **kwargs: Any,
) -> List[Object]:
    """Create a list of objects stacked on parent, dispatching on ``primitive``.

    ``primitive="ycb"`` samples YCB meshes via :func:`create_ycb_objects`; any
    other value creates primitive shapes via :func:`create_primitive_objects`.
    """
    if primitive == "ycb":
        return create_ycb_objects(num_objects, parent=parent)
    return create_primitive_objects(
        num_objects, primitive=primitive, width=width, parent=parent, **kwargs
    )


def create_circle_poses(num: int, distance: float) -> List[Pose]:
    """Return num poses arranged evenly around a circle of the given radius."""
    yaws = [0.0, PI, -PI / 2, PI / 2]
    assert 0 <= num <= len(yaws)
    base_poses = []
    for i in range(num):
        yaw = yaws[i]
        x = -distance * math.cos(yaw)
        y = -distance * math.sin(yaw)
        base_poses.append(create_pose(x, y, yaw=yaw))
    return base_poses


def create_line_poses(num: int, distance: float, **kwargs: Any) -> List[Pose]:
    """Return num poses evenly spaced along the y-axis spanning the given distance."""
    return [
        create_pose(y=y, **kwargs)
        for y in np.linspace(start=-distance / 2, stop=+distance / 2, num=num, endpoint=True)
    ]


def create_franka_platforms_world(
    num_robots: int = 1,
    robot: str = "franka",
    num_platforms: int = 2,
    num_objects: Optional[int] = None,
    primitive: str = "cuboid",
    floor_size: float = 1.5,
    platform_size: float = 0.1,
    platform_height: float = 0.01,
    object_width: float = 0.05,
    forward_distance: float = 0.5,
    **kwargs: Any,
) -> World:
    """Single-robot scene with a floor, N platforms, and N graspable cubes.

    Mirrors custream.scene.create_franka_platforms_world but uses curobov2
    (MotionPlanner / Scene) in place of curobov1 (MotionGen / WorldConfig).

    Platform layout:
        Platforms are placed at the corners/edges of the floor using a
        fixed sign pattern so they stay within reach of the Franka arm.

    Args:
        num_robots: Number of robots (this scene supports a single robot).
        num_platforms: Number of platforms (≤ 8).
        num_objects: Number of graspable cubes (defaults to num_platforms).
        floor_size: Side length of the square floor in metres.
        platform_size: Side length of each square platform in metres.
        platform_height: Thickness of each platform in metres.
        object_width: Side length of each graspable cube in metres.
        **kwargs: Forwarded to World / MotionPlannerCfg.create.

    Returns:
        Configured World instance ready for motion planning.
    """
    platform_signs = [
        [0, +1],
        [0, -1],
        [+1, +1],
        [+1, -1],
        [+1, 0],
        [-1, +1],
        [-1, -1],
        [-1, 0],
    ]
    assert num_robots == 1, f"platforms scene supports a single robot, got num_robots={num_robots}"
    assert num_platforms <= len(platform_signs)
    if num_objects is None:
        num_objects = num_platforms

    robot_config = load_robot_config(robot=robot)

    floor = CuboidObject(
        name="floor",
        width=floor_size,
        height=0.1,
        color="grey",
        surface_config=SurfaceConfig(),
    )
    floor.place(bottom_pose=create_pose(x=forward_distance, z=-floor.height))

    platform_poses = [
        create_pose(
            x=sx * (floor_size - platform_size) / 2, y=sy * (floor_size - platform_size) / 2
        )
        for sx, sy in platform_signs[:num_platforms]
    ]
    platforms = create_objects(
        num_objects=num_platforms,
        name="platform",
        width=platform_size,
        height=platform_height,
        parent=floor,
        poses=platform_poses,
    )

    movables = create_objects(
        num_objects=num_objects,
        primitive=primitive,
        width=object_width,
        grasp_config=GraspConfig(),
    )
    for movable in movables:
        movable.stack(floor)

    objects: List[Object] = [floor] + platforms + movables
    world = World(robot_config, objects, **kwargs)
    for arm in world.arms:
        world.set_joint_positions(world.get_arm_joints(arm), FRANKA_POSITIONS)
    world.set_camera_pose(CAMERA_POSE)

    return world


def create_franka_platform_world(
    num_robots: int = 1,
    robot: str = "franka",
    num_objects: int = 1,
    primitive: str = "cuboid",
    floor_width: Optional[float] = None,
    floor_depth: Optional[float] = None,
    platform_width: Optional[float] = 0.2,
    object_width: float = 0.05,
    object_depth: Optional[float] = None,
    robot_distance: float = 0.5,
    **kwargs: Any,
) -> World:
    """Scene with a shared floor, an optional platform, and N graspable cubes.

    Robots are arranged in a circle of radius robot_distance around the shared
    workspace (matching custream.scene.create_franka_world). When
    ``platform_width`` is None there is no platform and the objects rest
    directly on the floor.
    """
    assert num_robots >= 1, f"num_robots must be >= 1, got {num_robots}"
    assert num_objects >= 0, f"num_objects must be >= 0, got {num_objects}"
    if floor_width is None:
        floor_width = (robot_distance + 0.15) * 2

    base_poses = create_circle_poses(num_robots, robot_distance)
    robot_config = load_robot_config(robot=robot, base_poses=base_poses)
    floor_pose = create_pose(z=-0.1)

    floor = CuboidObject(
        name="floor",
        width=floor_width,
        depth=floor_depth,
        height=0.1,
        color="grey",
        surface_config=SurfaceConfig(),
    )
    floor.place(bottom_pose=floor_pose)

    if platform_width is not None:
        platform = CuboidObject(name="platform", width=platform_width, height=0.01, color="black")
        platform.stack(floor)
    else:
        platform = floor

    movables = create_objects(
        num_objects=num_objects,
        primitive=primitive,
        parent=platform,
        width=object_width,
        depth=object_depth,
        grasp_config=GraspConfig(),
    )

    objects: List[Object] = [floor] + movables
    if platform is not floor:
        objects.insert(0, platform)
    world = World(robot_config, objects, **kwargs)
    for arm in world.arms:
        world.set_joint_positions(world.get_arm_joints(arm), FRANKA_POSITIONS)

    if num_robots == 4:
        world.set_camera_pose(create_pose(roll=3 * np.pi / 8, yaw=np.pi / 4))
    else:
        world.set_camera_pose(CAMERA_POSE)
    return world


def create_franka_asset_world(
    num_robots: int = 1,
    robot: str = "franka",
    num_objects: int = 1,
    primitive: str = "cuboid",
    object_width: float = 0.05,
    asset: str = "table",
    floor_width: Optional[float] = None,
    asset_size: float = 0.4,
    asset_height: float = 0.2,
    num_shelves: int = 3,
    robot_distance: float = 0.5,
    **kwargs: Any,
) -> World:
    """Like :func:`create_franka_platform_world` but with a :class:`SceneObject`
    asset (a 4-legged ``"table"`` or a multi-board ``"shelf"``) centered on the
    floor; the objects rest on the asset's top surface.
    """
    assert num_robots >= 1, f"num_robots must be >= 1, got {num_robots}"
    assert num_objects >= 0, f"num_objects must be >= 0, got {num_objects}"
    if asset not in ASSETS:
        raise ValueError(f"Unknown asset {asset!r}. Choose from: {list(ASSETS)}")
    if floor_width is None:
        floor_width = (robot_distance + 0.15) * 2

    base_poses = create_circle_poses(num_robots, robot_distance)
    robot_config = load_robot_config(robot=robot, base_poses=base_poses)

    floor = CuboidObject(
        name="floor",
        width=floor_width,
        height=0.1,
        color="grey",
        surface_config=SurfaceConfig(),
    )
    floor.place(bottom_pose=create_pose(z=-0.1))

    asset_kwargs: dict = dict(width=asset_size, height=asset_height)
    if asset == "shelf":
        asset_kwargs["num_shelves"] = num_shelves
    asset_scene = ASSETS[asset](**asset_kwargs)
    body = SceneObject(name=asset, scene=asset_scene, surface_config=SurfaceConfig(approach="mesh"))
    body.stack(floor)

    movables = create_objects(
        num_objects=num_objects,
        primitive=primitive,
        parent=body,
        width=object_width,
        grasp_config=GraspConfig(),
    )

    objects: List[Object] = [floor, body] + movables
    world = World(robot_config, objects, **kwargs)
    for arm in world.arms:
        world.set_joint_positions(world.get_arm_joints(arm), FRANKA_POSITIONS)
    world.set_camera_pose(CAMERA_POSE)
    return world


def create_franka_line_world(
    num_robots: int = 1,
    robot: str = "franka",
    num_objects: int = 1,
    primitive: str = "cuboid",
    distance: float = 1.0,
    object_width: float = 0.05,
    **kwargs: Any,
) -> World:
    """Multi-robot scene with one platform per robot and N graspable cubes."""
    assert num_robots >= 1, f"num_robots must be >= 1, got {num_robots}"
    assert num_objects >= 0, f"num_objects must be >= 0, got {num_objects}"
    base_poses = create_line_poses(num_robots, distance=(num_robots - 1) * distance)
    robot_config = load_robot_config(robot=robot, base_poses=base_poses)

    platform_poses = [multiply_poses(base_pose, create_pose(x=0.5)) for base_pose in base_poses]
    platforms = create_objects(
        num_objects=len(platform_poses),
        name="platform",
        width=0.2,
        height=0.01,
        poses=platform_poses,
    )

    movables = create_objects(
        num_objects=num_objects,
        primitive=primitive,
        width=object_width,
        grasp_config=GraspConfig(roll_interval="top"),
    )
    for i, movable in enumerate(movables):
        movable.stack(parent=platforms[i % len(platforms)])

    objects: List[Object] = platforms + movables
    world = World(robot_config, objects, **kwargs)
    for arm in world.arms:
        world.set_joint_positions(world.get_arm_joints(arm), FRANKA_POSITIONS)
    world.set_camera_pose(create_pose(roll=3 * np.pi / 8, yaw=np.pi / 2))
    return world


def create_franka_motion_world(
    num_robots: int = 1,
    robot: str = "franka",
    num_objects: int = 0,
    primitive: Optional[str] = None,
    distance: float = 1.0,
    obstacle_width: float = 0.1,
    obstacle_depth: float = 0.1,
    obstacle_height: float = 0.1,
    **kwargs: Any,
) -> World:
    """One rectangular obstacle per robot, no platforms or graspable objects.

    Each obstacle is a non-graspable cuboid placed ``obstacle_distance`` in
    front of its robot along the robot's local +x axis, resting on the ground
    plane so the arm has to plan around it.

    ``num_objects`` / ``primitive`` are accepted for signature parity with the
    other ``create_franka_*_world`` scenes but ignored here — this scene has
    no graspable objects.
    """
    del num_objects, primitive
    assert num_robots >= 1, f"num_robots must be >= 1, got {num_robots}"
    base_poses = create_line_poses(num_robots, distance=(num_robots - 1) * distance)
    robot_config = load_robot_config(robot=robot, base_poses=base_poses)

    obstacles: List[Object] = []
    for i, base_pose in enumerate(base_poses):
        obstacle_pose = multiply_poses(
            base_pose,
            create_pose(x=0.2, y=0.0, z=0.5),
        )
        obstacle = CuboidObject(
            name=f"obstacle{i}",
            width=obstacle_width,
            depth=obstacle_depth,
            height=obstacle_height,
            color="gray",
            pose=obstacle_pose,
        )
        obstacles.append(obstacle)

    world = World(robot_config, obstacles, **kwargs)
    for arm in world.arms:
        world.set_joint_positions(world.get_arm_joints(arm), FRANKA_POSITIONS)
    world.set_camera_pose(create_pose(roll=3 * np.pi / 8, yaw=np.pi / 2))

    _sample_motion_state(world)
    return world


def _sample_motion_state(
    world: World,
    num_samples: int = 1000,
    num_steps: int = 25,
    obstructed: bool = True,
    verbose: bool = True,
) -> bool:
    """Move ``world`` to a collision-free configuration whose straight-line
    interpolation to the combined ``FRANKA_POSITIONS`` target either collides
    (``obstructed=True``) or stays collision-free (``obstructed=False``).

    Draws ``num_samples`` collision-free joint configurations via
    :meth:`World.sample_joint_states`, builds a target JointState with every
    arm forced to ``FRANKA_POSITIONS`` (other active joints kept at their
    current values), interpolates via :func:`interpolate_joint_states`, and
    applies the first start whose interpolated path matches ``obstructed``.

    Returns ``True`` if such a start was found and applied, ``False``
    otherwise (the world is left at the target).
    """
    # NVIDIA
    from schedulestream.applications.custream2.collision import joint_state_collision

    t0 = current_time()
    start_js = world.sample_joint_states(num_samples, collisions=True, verbose=False)
    if len(start_js.position) == 0:
        return False
    active_joints = list(start_js.joint_names)

    current = world.get_joint_state(active_joints).position
    target_pos = current.expand_as(start_js.position).clone()
    for arm in world.arms:
        arm_joints = world.get_arm_joints(arm)
        arm_indices = [active_joints.index(j) for j in arm_joints]
        target_pos[:, arm_indices] = torch.tensor(
            FRANKA_POSITIONS[: len(arm_joints)],
            dtype=target_pos.dtype,
            device=target_pos.device,
        )
    target_js = JointState.from_position(target_pos, joint_names=active_joints)

    traj_js = interpolate_joint_states([start_js, target_js], num_steps)
    blocked = joint_state_collision(world, traj_js)
    match = blocked if obstructed else ~blocked
    match_idx = torch.nonzero(match, as_tuple=False).flatten()
    if verbose:
        kind = "non-straight-line" if obstructed else "straight-line-feasible"
        print(
            f"{_sample_motion_state.__name__}) Sampled {len(start_js.position)}/{num_samples}"
            f" collision-free starts, {len(match_idx)} are {kind} |"
            f" Elapsed: {elapsed_time(t0):.3f} sec"
        )
    if len(match_idx) == 0:
        return False
    world.set_joint_positions(active_joints, start_js.position[match_idx[0]].cpu().numpy())
    return True


def create_franka_stand_world(
    num_robots: int = 1,
    robot: str = "franka",
    num_objects: int = 2,
    primitive: str = "cuboid",
    floor_size: float = 1.0,
    stand_height: float = 0.15,
    object_width: float = 0.05,
    **kwargs: Any,
) -> World:
    """Single-robot scene with an elevated stand on a riser and a low platform."""
    assert num_robots == 1, f"stand scene supports a single robot, got num_robots={num_robots}"
    assert num_objects >= 0, f"num_objects must be >= 0, got {num_objects}"
    robot_config = load_robot_config(robot=robot)

    floor = CuboidObject(
        name="floor",
        width=floor_size,
        height=0.1,
        color="grey",
        surface_config=SurfaceConfig(),
    )
    floor.place(bottom_pose=create_pose(z=-floor.height))

    riser = CuboidObject(name="riser", width=0.25, height=stand_height, color="brown")
    riser.place(bottom_pose=create_pose(x=0.3, z=0.0))

    stand = CuboidObject(
        name="stand",
        width=0.3,
        height=0.01,
        color="black",
        surface_config=SurfaceConfig(),
    )
    stand.place(bottom_pose=create_pose(x=0.3, z=stand_height))

    low_platform = CuboidObject(
        name="low_platform",
        width=0.25,
        height=0.01,
        color="black",
        surface_config=SurfaceConfig(),
    )
    low_platform.place(bottom_pose=create_pose(x=-0.3, z=0.0))

    movables = create_objects(
        num_objects=num_objects, primitive=primitive, width=object_width, grasp_config=GraspConfig()
    )
    for movable in movables:
        movable.stack(low_platform)

    objects: List[Object] = [floor, riser, stand, low_platform] + movables
    world = World(robot_config, objects, **kwargs)
    for arm in world.arms:
        world.set_joint_positions(world.get_arm_joints(arm), FRANKA_POSITIONS)
    world.set_camera_pose(CAMERA_POSE)
    return world


def create_franka_relay_world(
    num_robots: int = 2,
    robot: str = "franka",
    num_objects: int = 2,
    primitive: str = "cuboid",
    distance: float = 1.0,
    platform_width: float = 0.2,
    object_width: float = 0.05,
    object_depth: Optional[float] = None,
    **kwargs: Any,
) -> World:
    """Two robots along the y-axis with three platforms (start → mid → end).

    Objects start on the outer platforms; the goal encourages handoff via the shared
    mid_platform, exercising multi-robot coordination.
    """
    assert num_robots == 2, f"relay scene requires exactly two robots, got num_robots={num_robots}"
    assert num_objects >= 0, f"num_objects must be >= 0, got {num_objects}"
    base_poses = create_line_poses(num_robots, distance=distance)
    robot_config = load_robot_config(robot=robot, base_poses=base_poses)

    platform_specs = [
        ("start_platform", -distance / 2),
        ("mid_platform", 0.0),
        ("end_platform", +distance / 2),
    ]
    platforms: List[Object] = []
    for name, y in platform_specs:
        p = CuboidObject(
            name=name,
            width=platform_width,
            height=0.01,
            color="black",
            surface_config=SurfaceConfig(),
        )
        p.place(bottom_pose=create_pose(x=0.5, y=y))
        platforms.append(p)

    movables = create_objects(
        num_objects=num_objects,
        primitive=primitive,
        width=object_width,
        depth=object_depth,
        grasp_config=GraspConfig(),
    )
    for i, movable in enumerate(movables):
        movable.stack(platforms[0] if i % 2 == 0 else platforms[2])

    objects: List[Object] = platforms + movables
    world = World(robot_config, objects, **kwargs)
    for arm in world.arms:
        world.set_joint_positions(world.get_arm_joints(arm), FRANKA_POSITIONS)
    world.set_camera_pose(create_pose(roll=3 * np.pi / 8, yaw=np.pi / 2))
    return world


def create_franka_pack_world(
    num_robots: int = 2,
    num_objects: int = 2,
    floor_width: float = 0.6,
    floor_depth: float = 1.0,
    platform_width: float = 0.4,
    **kwargs: Any,
) -> World:
    """Pack scene: large platform, objects start on the floor and must be moved to the platform."""
    world = create_franka_platform_world(
        num_robots=num_robots,
        num_objects=num_objects,
        floor_width=floor_width,
        floor_depth=floor_depth,
        platform_width=platform_width,
        **kwargs,
    )
    floor = max(world.fixed_objects, key=lambda o: o.area)
    for obj in world.movable_objects:
        obj.stack(floor)
    return world


def create_franka_tight_world(
    num_objects: int = 2,
    object_width: float = 0.05,
    buffer: float = 0.01,
    **kwargs: Any,
) -> World:
    """Pack scene whose platform is only just large enough to hold every object.

    The platform is sized for a ``ceil(sqrt(n)) x ceil(sqrt(n))`` grid of
    cells, each ``object_width`` plus an additive ``buffer`` (m).
    """
    cells = math.ceil(math.sqrt(num_objects))
    platform_width = cells * (object_width + buffer)
    return create_franka_pack_world(
        num_objects=num_objects,
        object_width=object_width,
        platform_width=platform_width,
        **kwargs,
    )


def create_franka_blocked_world(
    num_objects: int = 2,
    object_width: float = 0.05,
    **kwargs: Any,
) -> World:
    """Blocked scene: the platform matches the object footprint exactly; object
    index 1 starts on (and fully blocks) the platform, the rest start on the floor."""
    world = create_franka_pack_world(
        num_objects=num_objects,
        object_width=object_width,
        platform_width=object_width,
        **kwargs,
    )
    platform = min(world.fixed_objects, key=lambda o: o.area)
    movables = world.movable_objects
    if len(movables) > 1:
        movables[1].stack(platform)
    return world


def create_franka_unpack_world(
    num_robots: int = 2,
    num_objects: int = 2,
    floor_width: float = 0.6,
    floor_depth: float = 1.0,
    **kwargs: Any,
) -> World:
    """Unpack scene: medium platform, objects start on the platform and must be moved to the floor.
    """
    world = create_franka_platform_world(
        num_robots=num_robots,
        num_objects=num_objects,
        floor_width=floor_width,
        floor_depth=floor_depth,
        **kwargs,
    )
    platform = min(world.fixed_objects, key=lambda o: o.area)
    for obj in world.movable_objects:
        obj.stack(platform)
    return world


def create_franka_stack_world(
    num_robots: int = 2,
    num_objects: int = 2,
    floor_width: float = 0.6,
    floor_depth: float = 1.0,
    platform_width: float = 0.1,
    **kwargs: Any,
) -> World:
    """Stack scene: small platform, objects start on the floor and must be stacked on each other."""
    world = create_franka_platform_world(
        num_robots=num_robots,
        num_objects=num_objects,
        floor_width=floor_width,
        floor_depth=floor_depth,
        platform_width=platform_width,
        **kwargs,
    )
    floor = max(world.fixed_objects, key=lambda o: o.area)
    for obj in world.movable_objects:
        obj.stack(floor)
    return world


create_franka_ycb_world = lambda primitive="ycb", **kwargs: create_franka_platform_world(
    primitive=primitive, platform_width=None, **kwargs
)

create_franka_floor_world = lambda floor_size=0.5, **kwargs: create_franka_platforms_world(
    num_platforms=0, floor_size=floor_size, **kwargs
)


SCENES = {
    "platforms": create_franka_platforms_world,
    "floor": create_franka_floor_world,
    "platform": create_franka_platform_world,
    "table": lambda **kw: create_franka_asset_world(asset="table", **kw),
    "shelf": lambda **kw: create_franka_asset_world(asset="shelf", asset_height=0.6, **kw),
    "line": create_franka_line_world,
    "motion": create_franka_motion_world,
    "stand": create_franka_stand_world,
    "relay": create_franka_relay_world,
    "pack": create_franka_pack_world,
    "tight": create_franka_tight_world,
    "blocked": create_franka_blocked_world,
    "unpack": create_franka_unpack_world,
    "stack": create_franka_stack_world,
    "ycb": create_franka_ycb_world,
}


def create_franka_world(scene: str = "platforms", **kwargs: Any) -> World:
    """Create a named Franka scene. scene must be one of: platforms, platform, line, stand, relay.
    """
    if scene not in SCENES:
        raise ValueError(f"Unknown scene {scene!r}. Choose from: {list(SCENES)}")
    return SCENES[scene](**kwargs)


if __name__ == "__main__":
    # Standard Library
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scene", nargs="?", default="platforms", choices=list(SCENES), help="Scene to load."
    )
    parser.add_argument(
        "-s", "--seed", nargs="?", const=None, default=0, type=int, help="The random seed."
    )
    parser.add_argument(
        "-n", "--num-scenes", type=int, default=1, help="Number of scenes to create and show."
    )
    parser.add_argument("--num-robots", type=int, default=None, help="Number of robots.")
    parser.add_argument("--num-objects", type=int, default=None, help="Number of objects.")
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        choices=["franka", "droid"],
        help="Robot variant; forwarded to load_robot_config.",
    )
    parser.add_argument(
        "--primitive",
        type=str,
        default=None,
        choices=PRIMITIVES,
        help="Movable object primitive (e.g. cuboid, ycb).",
    )
    parser.add_argument(
        "--spheres",
        action="store_true",
        help="Visualize collision spheres (passes debug=True to World)",
    )
    args = parser.parse_args()

    optional = dict(
        num_robots=args.num_robots,
        num_objects=args.num_objects,
        primitive=args.primitive,
        robot=args.robot,
    )
    kwargs = {key: value for key, value in optional.items() if value is not None}

    set_seed(seed=args.seed)
    for _ in range(args.num_scenes):
        world = create_franka_world(args.scene, debug=args.spheres, **kwargs)
        world.show()
        world.destroy()
