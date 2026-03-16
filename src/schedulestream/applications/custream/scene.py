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
from typing import Any, List, Optional

# Third Party
import numpy as np

# NVIDIA
from schedulestream.applications.custream.franka import load_franka_config
from schedulestream.applications.custream.object import (
    CuboidObject,
    GraspConfig,
    MeshObject,
    Object,
    SurfaceConfig,
    create_primitive_objects,
)
from schedulestream.applications.custream.utils import PI, create_pose, multiply_poses
from schedulestream.applications.custream.world import World
from schedulestream.applications.custream.yourdf import load_robot_yourdf

CAMERA_POSE = create_pose(roll=3 * np.pi / 8, yaw=np.pi / 2)


def create_circle_poses(num: int, distance: float):
    yaws = [0.0, PI, -PI / 2, PI / 2]
    assert 0 <= num <= len(yaws)

    base_poses = []
    for i in range(num):
        yaw = yaws[i]
        x = -distance * math.cos(yaw)
        y = -distance * math.sin(yaw)
        base_poses.append(create_pose(x, y, yaw=yaw))
    return base_poses


def create_line_poses(num: int, distance: float, **kwargs):
    return [
        create_pose(y=y, **kwargs)
        for y in np.linspace(start=-distance / 2, stop=+distance / 2, num=num, endpoint=True)
    ]


def create_robot_objects(robot_config: dict) -> List[Object]:
    urdf_path = robot_config["robot_cfg"]["kinematics"]["urdf_path"]
    yourdf = load_robot_yourdf(urdf_path)
    fixed_nodes = yourdf.get_node_cluster(yourdf.base_link)
    objects = []
    for node in fixed_nodes:
        matrix, geometry = yourdf.scene.graph.get(frame_to=node, frame_from=None)
        if geometry is not None:
            mesh = yourdf.scene.geometry[geometry]
            objects.append(MeshObject(node, mesh, pose=matrix))
    return objects


def shrink_colliding_spheres(world: World, epsilon: float = 1e-1, debug: bool = True):
    distances = world.get_distances().squeeze(0)
    colliding = distances > 0.0
    distances[~colliding] = 0.0
    world.kinematics_config.reference_link_spheres[:, 3] -= distances
    world.kinematics_config.link_spheres[...] = world.kinematics_config.reference_link_spheres
    world.ik_solver.kinematics.kinematics_config.reference_link_spheres[
        ...
    ] = world.kinematics_config.reference_link_spheres
    world.ik_solver.kinematics.kinematics_config.link_spheres[
        ...
    ] = world.kinematics_config.reference_link_spheres

    if debug:
        world.add_robot_spheres(world.scene)
        world.show()
    raise NotImplementedError()


FRANKA_POSITIONS = np.pi * np.array([0, -1.0 / 4, 0, -2.0 / 4, 0, 2.0 / 4, 1.0 / 4])


def create_franka_world(
    num_robots: int = 1,
    num_objects: int = 1,
    robot_distance: float = 0.5,
    floor_width: Optional[float] = None,
    floor_depth: Optional[float] = None,
    platform_width: float = 0.2,
    object_width: float = 0.05,
    object_depth: Optional[float] = None,
    **kwargs: Any,
) -> World:
    if floor_width is None:
        floor_width = (robot_distance + 0.15) * 2

    base_poses = create_circle_poses(num_robots, robot_distance)
    robot_config = load_franka_config(base_poses)

    floor = CuboidObject(
        name="floor",
        width=floor_width,
        depth=floor_depth,
        height=0.1,
        color="grey",
        surface_config=SurfaceConfig(),
    )
    floor.place(bottom_pose=create_pose(z=-floor.height))

    platform = CuboidObject(name="platform", width=platform_width, height=0.01, color="black")
    platform.stack(floor)

    parent = platform
    objects: List[Object] = [platform, floor]
    objects.extend(
        create_primitive_objects(
            num_objects=num_objects,
            parent=parent,
            width=object_width,
            depth=object_depth,
            grasp_config=GraspConfig(),
        )
    )

    world = World(robot_config, objects, **kwargs)
    for arm in world.arms:
        world.set_joint_positions(world.get_arm_joints(arm), FRANKA_POSITIONS)

    if num_robots == 4:
        world.set_camera_pose(create_pose(roll=3 * np.pi / 8, yaw=np.pi / 4))
    return world


def create_franka_line_world(
    num_robots: int = 1,
    num_objects: int = 1,
    distance: float = 1.0,
    intermediate: bool = False,
    **kwargs: Any,
) -> World:
    base_poses = create_line_poses(num_robots, distance=(num_robots - 1) * distance)
    robot_config = load_franka_config(base_poses)

    platform_poses = [multiply_poses(base_pose, create_pose(x=0.5)) for base_pose in base_poses]
    platforms = create_primitive_objects(
        num_objects=len(platform_poses),
        name="platform",
        width=0.2,
        height=0.01,
        poses=platform_poses,
    )

    cubes = create_primitive_objects(
        num_objects=num_objects, width=0.05, grasp_config=GraspConfig()
    )
    for i, cube in enumerate(cubes):
        platform = platforms[i % len(platforms)]
        cube.stack(platform)
    objects = platforms + cubes

    if intermediate:
        platform_poses = [
            multiply_poses(platform_pose, create_pose(y=distance / 2))
            for platform_pose in platform_poses[:-1]
        ]
        objects.extend(
            create_primitive_objects(
                num_objects=len(platform_poses),
                name="platform",
                width=0.2,
                height=0.01,
                poses=platform_poses,
                colors=["grey"],
                start_index=len(platforms),
                surface_config=SurfaceConfig(),
            )
        )

    world = World(robot_config, objects, **kwargs)
    for arm in world.arms:
        world.set_joint_positions(world.get_arm_joints(arm), FRANKA_POSITIONS)
    world.set_camera_pose(CAMERA_POSE)

    return world
