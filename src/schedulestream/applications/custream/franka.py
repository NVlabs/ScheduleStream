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
from typing import Any, Dict, List, Optional, Union

# Third Party
import yourdfpy
from curobo.types.math import Pose

# NVIDIA
from schedulestream.applications.custream.config import create_robot_config, load_robot_yaml
from schedulestream.applications.custream.utils import unit_pose
from schedulestream.applications.custream.yourdf import (
    Yourdf,
    absolute_urdf_path,
    combine_yourdfs,
    load_robot_yourdf,
)


def repair_joint_limits(yourdf: Yourdf) -> None:
    for joint in yourdf.robot.joints:
        if joint.type == "fixed":
            continue
        if joint.limit is None:
            joint.limit = yourdfpy.Limit(velocity=1.0)


def create_multi_yourdf(
    urdf_path: Union[str, List[str]], base_poses: Optional[List[Pose]] = None, **kwargs: Any
) -> Yourdf:
    if base_poses is None:
        base_poses = [unit_pose()]
    if isinstance(urdf_path, str):
        urdf_path = len(base_poses) * [urdf_path]
    assert len(urdf_path) == len(base_poses)

    yourdfs = []
    for i, pose in enumerate(base_poses):
        yourdf = load_robot_yourdf(urdf_path[i], **kwargs)
        if len(base_poses) != 1:
            yourdf.apply_prefix(prefix=f"arm{i}_")
        yourdfs.append(yourdf)
    return combine_yourdfs(yourdfs, base_poses)


def create_multi_config(
    urdf_path: str,
    tool_link: str,
    base_poses: Optional[List[Pose]] = None,
    collision_spheres: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> dict:
    if collision_spheres is None:
        collision_spheres = {}
    composite_yourdf = create_multi_yourdf(urdf_path, base_poses)
    urdf_path = composite_yourdf.export()

    tool_links = [link for link in composite_yourdf.links if link.endswith(tool_link)]
    arms = [link.removesuffix(tool_link) for link in tool_links]
    robot_config = create_robot_config(urdf_path, tool_links, ignore_arms=False)

    link_map = {}
    for arm in arms:
        arm_links = [link for link in composite_yourdf.links if link.startswith(arm)]
        link_map.update({arm_link: arm_link.removeprefix(arm) for arm_link in arm_links})
    for arm_link in link_map:
        if link_map[arm_link] not in collision_spheres:
            continue
        spheres = collision_spheres.get(link_map[arm_link], [])
        robot_config["robot_cfg"]["kinematics"]["collision_spheres"][arm_link] = spheres

    return robot_config


def load_franka_config(base_poses: Optional[List[Pose]] = None) -> dict:
    robot_config = load_robot_yaml("franka.yml")
    kinematics = robot_config["robot_cfg"]["kinematics"]
    collision_spheres = kinematics["collision_spheres"]
    urdf_path = absolute_urdf_path(kinematics["urdf_path"])

    return create_multi_config(
        urdf_path, tool_link="ee_link", base_poses=base_poses, collision_spheres=collision_spheres
    )
