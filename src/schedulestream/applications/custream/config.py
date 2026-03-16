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


# Standard Library
import argparse
import copy
import os
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Third Party
import numpy as np
import yaml
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, load_yaml

# NVIDIA
from schedulestream.applications.custream.yourdf import Yourdf, load_robot_yourdf
from schedulestream.common.graph import dfs, undirected_from_edges
from schedulestream.common.utils import flatten, remove_duplicates


def load_robot_yaml(robot_yaml: Union[str, dict, RobotConfig]) -> dict:
    configs_dir = get_robot_configs_path()
    if not isinstance(robot_yaml, str):
        robot_config = robot_yaml
    else:
        if not os.path.exists(robot_yaml):
            config_names = sorted(
                config_name
                for config_name in os.listdir(configs_dir)
                if not os.path.isdir(os.path.join(configs_dir, config_name))
            )
            assert robot_yaml in config_names, config_names
            robot_yaml = os.path.join(configs_dir, robot_yaml)
        robot_config = load_yaml(robot_yaml)

    kinematics_config = robot_config["robot_cfg"]["kinematics"]
    if isinstance(kinematics_config["collision_spheres"], str):
        spheres_path = os.path.join(configs_dir, kinematics_config["collision_spheres"])
        kinematics_config["collision_spheres"] = load_yaml(spheres_path)["collision_spheres"]
    return robot_config


def get_active_links(yourdf: Yourdf, robot_config: dict, include_roots: bool = False) -> List[str]:
    kinematics = robot_config["robot_cfg"]["kinematics"]
    active_joints = kinematics["cspace"]["joint_names"]
    return yourdf.get_active_links(joints=active_joints, include_roots=include_roots)


def get_gripper_joints(robot_config: dict, yourdf: Optional[Yourdf] = None) -> List[str]:
    kinematics = robot_config["robot_cfg"]["kinematics"]
    if yourdf is None:
        yourdf = load_robot_yourdf(kinematics["urdf_path"], load_meshes=False)
    locked_joints = list(kinematics["lock_joints"])
    active_links = get_active_links(yourdf, robot_config)
    gripper_joints = list(set(map(yourdf.get_link_parent, active_links)) & set(locked_joints))
    return list(filter(gripper_joints.__contains__, locked_joints))


def get_gripper_links(robot_config: dict, yourdf: Optional[Yourdf] = None) -> List[str]:
    if yourdf is None:
        kinematics = robot_config["robot_cfg"]["kinematics"]
        yourdf = load_robot_yourdf(kinematics["urdf_path"], load_meshes=False)
    gripper_joints = get_gripper_joints(robot_config, yourdf)
    return yourdf.get_active_links(joints=gripper_joints, include_roots=True)


def create_ignore_pairs(
    yourdf: Yourdf, active_joints: List[str], depth: int = 2
) -> List[Tuple[str, str]]:
    inactive_joints = [joint for joint in yourdf.joints if joint not in active_joints]

    rigid_edges = list(yourdf.rigid_edges)
    rigid_edges.extend(map(yourdf.get_joint_edge, inactive_joints))
    clusters = yourdf.get_link_clusters(edges=rigid_edges)
    indices = list(range(len(clusters)))

    index_from_link = {}
    for index, cluster in enumerate(clusters):
        for link in cluster:
            index_from_link[link] = index

    active_edges = list(map(yourdf.get_joint_edge, active_joints))
    cluster_edges = undirected_from_edges(
        tuple(map(index_from_link.get, edge)) for edge in active_edges
    )
    ignore_pairs = []
    for index1 in indices:
        for index2 in dfs(cluster_edges, source_vertices=[index1], max_depth=depth):
            ignore_pairs.extend(product(clusters[index1], clusters[index2]))

    return ignore_pairs


def create_arm_ignore_pairs(yourdf: Yourdf, tool_links: List[str]) -> List[Tuple[str, str]]:
    arm_links = {}
    for tool_link in tool_links:
        active_joints = yourdf.get_active_joints(tool_link)
        active_links = yourdf.get_active_links(joints=active_joints, include_roots=True)
        arm_links[tool_link] = active_links

    arms_from_link = {}
    for arm, links in arm_links.items():
        for link in links:
            arms_from_link.setdefault(link, []).append(arm)

    arm_links = {
        arm: [link for link in links if len(arms_from_link[link]) == 1]
        for arm, links in arm_links.items()
    }
    ignore_pairs = []
    for arm1, arm2 in combinations(arm_links, r=2):
        ignore_pairs.extend(product(arm_links[arm1], arm_links[arm2]))
    return ignore_pairs


def create_self_collision_config(ignore_pairs: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
    self_collision_ignore = {}
    for link1, link2 in ignore_pairs:
        self_collision_ignore.setdefault(link1, set()).add(link2)
        self_collision_ignore.setdefault(link2, set()).add(link1)
    self_collision_ignore = {link: list(links) for link, links in self_collision_ignore.items()}
    return self_collision_ignore


def create_robot_config(
    urdf_path: str,
    tool_links: List[str],
    inactive_joints: Optional[List[str]] = None,
    ignore_arms: bool = True,
    tool_spheres: Optional[int] = 50,
    max_acceleration: float = 15.0,
    max_jerk: float = 500.0,
    verbose: bool = False,
    **kwargs: Any,
) -> dict:
    assert tool_links
    if inactive_joints is None:
        inactive_joints = []
    yourdf = load_robot_yourdf(urdf_path, load_meshes=False)

    links = yourdf.links
    for tool_link in tool_links:
        assert tool_link in links, (tool_link, links)

    active_joints = remove_duplicates(flatten(map(yourdf.get_active_joints, tool_links)))
    active_joints = list(filter(lambda j: not j in inactive_joints, active_joints))
    inactive_joints = [joint for joint in yourdf.joints if joint not in active_joints]

    if verbose:
        print(f"Yourdf: {yourdf}")
        print(f"Tool links ({len(tool_links)}): {tool_links}")
        print(f"Active joints ({len(active_joints)}): {active_joints}")
        print(f"Inactive joints ({len(inactive_joints)}): {inactive_joints}")
        print(f"Links ({len(links)}): {links}")

    collision_spheres = {link: [] for link in links}
    ignore_pairs = create_ignore_pairs(yourdf, active_joints, **kwargs)
    if ignore_arms:
        ignore_pairs.extend(create_arm_ignore_pairs(yourdf, tool_links))
    self_collision_ignore = create_self_collision_config(ignore_pairs)

    dofs = len(active_joints)
    retract_config = np.average(yourdf.get_joint_bounds(active_joints), axis=0)

    cspace = {
        "joint_names": active_joints,
        "null_space_weight": np.ones(dofs).tolist(),
        "retract_config": retract_config.tolist(),
        "cspace_distance_weight": np.ones(dofs).tolist(),
        "max_acceleration": max_acceleration,
        "max_jerk": max_jerk,
    }
    kinematics = {
        "use_usd_kinematics": False,
        "urdf_path": yourdf.urdf_path,
        "base_link": yourdf.base_link,
        "ee_link": tool_links[0],
        "link_names": tool_links,
        "collision_link_names": list(collision_spheres),
        "collision_spheres": collision_spheres,
        "collision_sphere_buffer": 0.0,
        "self_collision_ignore": self_collision_ignore,
        "self_collision_buffer": {link: 0.0 for link in links},
        "lock_joints": {joint: 0.0 for joint in inactive_joints},
        "extra_links": {},
        "cspace": cspace,
    }
    if tool_spheres is not None:
        kinematics["extra_collision_spheres"] = {
            tool_link: tool_spheres for tool_link in tool_links
        }

    robot_config = {"robot_cfg": {"kinematics": kinematics}}
    return robot_config


def extract_robot_config(
    robot_config: dict,
    tool_links: List[str],
    inactive_joints: Optional[List[str]] = None,
) -> dict:
    assert tool_links
    if inactive_joints is None:
        inactive_joints = []
    robot_config = copy.deepcopy(robot_config)
    kinematics = robot_config["kinematics"]
    cspace = kinematics["cspace"]
    assert set(tool_links) <= set(kinematics["link_names"])

    yourdf = load_robot_yourdf(kinematics["urdf_path"], load_meshes=False)
    active_joints = remove_duplicates(flatten(map(yourdf.get_active_joints, tool_links)))
    active_joints = list(
        filter(lambda j: (j in cspace["joint_names"]) and (not j in inactive_joints), active_joints)
    )
    inactive_joints = [joint for joint in yourdf.joints if joint not in active_joints]
    indices = list(map(cspace["joint_names"].index, active_joints))

    cspace.update(
        {
            "joint_names": active_joints,
            "null_space_weight": np.array(cspace["null_space_weight"])[indices].tolist(),
            "retract_config": np.array(cspace["retract_config"])[indices].tolist(),
            "cspace_distance_weight": np.array(cspace["cspace_distance_weight"])[indices].tolist(),
        }
    )

    kinematics.update(
        {
            "ee_link": tool_links[0],
            "link_names": list(tool_links),
            "lock_joints": {joint: 0.0 for joint in inactive_joints},
        }
    )

    return robot_config


def main():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf", type=str, help="The URDF path.")
    parser.add_argument("--links", type=str, nargs="+", help="The arm tool links.")
    args = parser.parse_args()
    print("Args:", args)

    yourdf = Yourdf(urdf_path=args.urdf)
    yourdf.dump()

    links = args.links
    if not links:
        links = yourdf.get_tool_links()
        print(f"Possible tool links: {links}")
    robot_config = create_robot_config(args.urdf, links)
    print(yaml.dump(robot_config))
    yourdf.show()


if __name__ == "__main__":
    main()
