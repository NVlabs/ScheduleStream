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
from typing import List, Optional

# Third Party
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.content import get_assets_path
from curobo.types import Pose

# NVIDIA
from schedulestream.applications.custream2.utils import create_pose, to_matrix
from schedulestream.applications.custream.yourdf import combine_yourdfs, load_robot_yourdf


def _prefix_str_list(values: list, prefix: str) -> list:
    """Prefix each string in a list."""
    return [f"{prefix}{v}" if isinstance(v, str) else v for v in values]


def load_franka_config(base_poses: Optional[List[Pose]] = None) -> dict:
    """Build and return a curobov2 robot config dict for one or more Franka robots."""
    robot_config = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    kinematics = robot_config["robot_cfg"]["kinematics"]

    urdf_path = join_path(str(get_assets_path()), kinematics["urdf_path"])
    tool_link = "tool_link"
    tool_pose = to_matrix(create_pose(yaw=-math.pi / 2))

    if base_poses is None:
        yourdf = load_robot_yourdf(urdf_path, load_meshes=False)
        yourdf.add_frame(tool_link, parent="ee_link", pose=tool_pose)
        kinematics["urdf_path"] = yourdf.export()
        kinematics["tool_frames"] = [tool_link]
        return robot_config

    n = len(base_poses)
    orig_joints = list(kinematics["cspace"]["joint_names"])

    yourdfs = []
    for i in range(n):
        y = load_robot_yourdf(urdf_path, load_meshes=False)
        y.add_frame(tool_link, parent="ee_link", pose=tool_pose)
        if n > 1:
            y.apply_prefix(f"arm{i}_")
        yourdfs.append(y)

    composite = combine_yourdfs(yourdfs, base_poses)
    kinematics["urdf_path"] = composite.export()

    kinematics["base_link"] = "world_link"

    if n == 1:
        kinematics["tool_frames"] = [tool_link]
        return robot_config

    prefix_list = [f"arm{i}_" for i in range(n)]
    kinematics["tool_frames"] = [f"{p}{tool_link}" for p in prefix_list]

    orig_cspace = dict(kinematics["cspace"])
    kinematics["cspace"]["joint_names"] = [f"{p}{j}" for p in prefix_list for j in orig_joints]
    for key in ["cspace_distance_weight", "null_space_weight", "default_joint_position"]:
        val = orig_cspace.get(key)
        if isinstance(val, (list, tuple)):
            kinematics["cspace"][key] = list(val) * n

    list_fields = ["collision_link_names", "grasp_contact_link_names", "mesh_link_names"]
    for field in list_fields:
        orig = kinematics.get(field)
        if isinstance(orig, list):
            kinematics[field] = [f"{p}{name}" for p in prefix_list for name in orig]

    orig_spheres2 = kinematics.get("collision_spheres")
    if isinstance(orig_spheres2, dict):
        kinematics["collision_spheres"] = {
            f"{p}{link}": spheres for p in prefix_list for link, spheres in orig_spheres2.items()
        }

    orig_scb = kinematics.get("self_collision_buffer")
    if isinstance(orig_scb, dict):
        kinematics["self_collision_buffer"] = {
            f"{p}{k}": v for p in prefix_list for k, v in orig_scb.items()
        }

    orig_sci = kinematics.get("self_collision_ignore")
    if isinstance(orig_sci, dict):
        kinematics["self_collision_ignore"] = {
            f"{p}{k}": _prefix_str_list(vs, p) for p in prefix_list for k, vs in orig_sci.items()
        }

    orig_ecs = kinematics.get("extra_collision_spheres")
    if isinstance(orig_ecs, dict):
        kinematics["extra_collision_spheres"] = {
            f"{p}{k}": v for p in prefix_list for k, v in orig_ecs.items()
        }

    orig_extra = kinematics.get("extra_links")
    if isinstance(orig_extra, dict):
        new_extra: dict = {}
        for p in prefix_list:
            for key, spec in orig_extra.items():
                new_key = f"{p}{key}"
                new_spec = dict(spec)
                for name_field in ("link_name", "parent_link_name"):
                    if name_field in new_spec and isinstance(new_spec[name_field], str):
                        new_spec[name_field] = f"{p}{new_spec[name_field]}"
                new_extra[new_key] = new_spec
        kinematics["extra_links"] = new_extra

    orig_lock = kinematics.get("lock_joints")
    if isinstance(orig_lock, dict):
        kinematics["lock_joints"] = {
            f"{p}{j}": v for p in prefix_list for j, v in orig_lock.items()
        }

    return robot_config


if __name__ == "__main__":
    # Standard Library
    import argparse

    # NVIDIA
    from schedulestream.applications.custream2.scene import CAMERA_POSE
    from schedulestream.applications.custream2.world import World

    parser = argparse.ArgumentParser()
    parser.add_argument("--spheres", action="store_true", help="Visualize collision spheres.")
    parser.add_argument("--headless", action="store_true", help="Skip the viewer (load only).")
    parser.add_argument(
        "--num-confs",
        type=int,
        default=1,
        help="Number of collision-free configs to sample and show.",
    )
    args = parser.parse_args()

    robot_config = load_franka_config()
    world = World(robot_config, objects=[], debug=args.spheres)
    world.set_camera_pose(CAMERA_POSE)
    world.dump()

    joint_states = world.sample_joint_states(args.num_confs * 10, collisions=True)
    positions = joint_states.position.cpu().numpy()
    for pos in positions[: args.num_confs]:
        world.set_joint_positions(list(joint_states.joint_names), pos)
        if not args.headless:
            world.show()

    world.destroy()
