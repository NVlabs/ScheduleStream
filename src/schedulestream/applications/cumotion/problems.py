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
from itertools import combinations
from typing import Any, Dict, Optional

# Third Party
from cumotion import Pose3

# NVIDIA
from schedulestream.applications.cumotion.generators import placement_sampler
from schedulestream.applications.cumotion.objects import Cuboid, Object
from schedulestream.applications.cumotion.utils import (
    BLACK,
    GREEN,
    GREY,
    RED,
    create_pose,
    hsv_colors,
)
from schedulestream.applications.cumotion.world import FrankaWorld, World


def create_franka_problem(
    num_cubes: int = 1, cube_width: float = 0.05, platform_width: Optional[float] = 0.2
) -> FrankaWorld:
    world = FrankaWorld()

    floor = Cuboid(name="floor", width=1.0, height=0.01, color=GREY)
    world.add_object(floor, pose=create_pose(floor.bottom_center) * create_pose([0.25, 0, 0]))

    if platform_width is not None:
        platform = Cuboid(name="platform", width=0.2, height=0.01, color=BLACK)
        world.add_object(
            platform, pose=world.get_pose(floor) * platform.stack(floor) * create_pose([0.25, 0, 0])
        )

    for i, color in enumerate(hsv_colors(num=num_cubes)):
        obj = Cuboid(name=f"cuboid{i}", width=cube_width, color=color, graspable=True)
        world.add_object(
            obj,
            pose=world.get_pose(platform) * obj.stack(platform),
        )
    return world


def sample_placements(
    world: World, support: Optional[Object] = None, distance: float = 0.05, **kwargs: Any
) -> Optional[Dict[Object, Pose3]]:
    if support is None:
        support = world.fixed_objects[-1]
    objects = world.movable_objects
    placement_generators = [placement_sampler(world, obj, support, **kwargs) for obj in objects]
    while True:
        obj_poses = {}
        for obj, placements in zip(objects, placement_generators):
            output = None
            while output is None:
                output = next(placements)
            (obj_pose,) = output
            obj_poses[obj] = obj_pose

        for (obj1, pose1), (obj2, pose2) in combinations(obj_poses.items(), 2):
            if world.object_object_collision(obj1, obj2, pose1, pose2, distance=distance):
                break
        else:
            for obj, pose in obj_poses.items():
                world.set_object_pose(obj, pose)
            return obj_poses
