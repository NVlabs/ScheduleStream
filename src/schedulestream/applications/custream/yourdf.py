#! /usr/bin/env python3
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
import math
import os
from typing import Annotated, Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

# Third Party
import numpy as np
import structlog
import trimesh
from yourdfpy import URDF, Joint, Limit, Link, Robot

# NVIDIA
from schedulestream.applications.custream.utils import PI
from schedulestream.common.graph import get_reachable
from schedulestream.common.utils import (
    compute_mapping,
    flatten,
    negate_test,
    remove_duplicates,
    safe_zip,
)

Pose = Annotated[np.dtype[np.floating[Any]], Literal["4", "4"]]
Conf = List[float]

WORLD_LINK = "world_link"


class Yourdf(URDF):
    def __init__(
        self,
        robot: Optional[Robot] = None,
        urdf_path: Optional[str] = None,
        mesh_dir: str = "",
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ):
        self.logger = structlog.get_logger()
        if robot is None:
            assert urdf_path is not None
            if not mesh_dir:
                mesh_dir = os.path.dirname(urdf_path)
            urdf = Yourdf.load(urdf_path, mesh_dir=mesh_dir, **kwargs)
            robot = urdf.robot

        super().__init__(robot, mesh_dir=mesh_dir, **kwargs)
        self.urdf_path = urdf_path
        self.mesh_dir = mesh_dir
        self.metadata = dict(metadata or {})
        self.prefix = ""

        for link_data in robot.links:
            for shape_data in link_data.visuals + link_data.collisions:
                mesh_data = shape_data.geometry.mesh
                if (mesh_data is not None) and (mesh_data.filename != "unknown_file"):
                    mesh_data.filename = self._filename_handler(mesh_data.filename)

    @property
    def name(self) -> str:
        return self.robot.name

    @property
    def graph(self) -> trimesh.scene.transforms.SceneGraph:
        return self.scene.graph

    @property
    def nodes(self) -> List[str]:
        return self.graph.nodes

    @property
    def root_link(self) -> str:
        return self.base_link

    @property
    def urdf_links(self) -> List[str]:
        if self.root_link != WORLD_LINK:
            return [self.root_link]
        return self.get_link_children(self.root_link)

    @property
    def leaf_links(self) -> List[str]:
        return [link for link in self.links if not self.get_link_children(link)]

    def subscene(self, nodes: List[str], scene: Optional[trimesh.Scene] = None) -> trimesh.Scene:
        if scene is None:
            scene = trimesh.Scene()
        for node in nodes:
            matrix, geometry = self.scene.graph.get(frame_to=node, frame_from=None)
            if geometry is not None:
                geometry = self.scene.geometry[geometry]
                scene.add_geometry(geometry, transform=matrix, parent_node_name=None)
        return scene

    @property
    def links(self) -> List[str]:
        return [l.name for l in self.robot.links]

    def get_link_pose(self, link: str, parent: Optional[str] = None) -> Pose:
        return self.get_transform(link, frame_from=parent)

    @property
    def joints(self) -> List[str]:
        return self.actuated_joint_names

    @property
    def all_joints(self) -> List[str]:
        return list(self.joint_map)

    @property
    def fixed_joints(self) -> List[str]:
        return list(filter(self.is_fixed, self.all_joints))

    @property
    def dofs(self) -> int:
        return len(self.joints)

    def convexify(self) -> None:
        for node, geometry in list(self.scene.geometry.items()):
            self.scene.geometry[node] = geometry.convex_hull

    def remove_visuals(self) -> None:
        for node, geometry in list(self.scene.geometry.items()):
            self.scene.geometry[node].visual = trimesh.visual.create_visual()

    def remove_mimic_joints(self):
        for joint in self.robot.joints:
            if joint.mimic is not None:
                joint.mimic = None

    def get_joint_index(self, joint: str) -> int:
        return self.actuated_joint_names.index(joint)

    def get_joint_indices(self, joints: Optional[List[str]] = None) -> List[int]:
        if joints is None:
            joints = self.joints
        return list(map(self.get_joint_index, joints))

    def get_joint_type(self, joint: str) -> str:
        assert joint in self.joint_map, (joint, list(self.joint_map))
        return self.joint_map[joint].type

    def get_joint_parent(self, joint: str) -> str:
        assert joint in self.joint_map, (joint, list(self.joint_map))
        return self.joint_map[joint].parent

    def get_joint_child(self, joint: str) -> str:
        assert joint in self.joint_map, (joint, list(self.joint_map))
        return self.joint_map[joint].child

    @property
    def parent_from_link(self) -> Dict[str, str]:
        return {joint.child: joint.name for joint in self.robot.joints}

    @property
    def children_from_link(self) -> Dict[str, List[str]]:
        children_from_link = {}
        for joint in self.robot.joints:
            children_from_link.setdefault(joint.parent, []).append(joint.name)
        return children_from_link

    def get_link_parent(self, link: str) -> Optional[str]:
        return self.parent_from_link.get(link, None)

    def get_link_children(self, link: str) -> List[str]:
        return self.children_from_link.get(link, [])

    def get_link_ancestors(self, link: str) -> List[str]:
        joint = self.get_link_parent(link)
        if joint is None:
            return []
        parent_link = self.get_joint_parent(joint)
        return self.get_link_ancestors(parent_link) + [parent_link]

    def get_link_descendants(self, link: str) -> List[str]:
        descendants = []
        for joint in self.get_link_children(link):
            child_link = self.get_joint_child(joint)
            descendants.append(child_link)
            descendants.extend(self.get_link_descendants(child_link))
        return descendants

    def get_link_subtree(self, link: str) -> List[str]:
        return [link] + self.get_link_descendants(link)

    def get_active_links(
        self, joints: Optional[List[str]] = None, include_roots: bool = True
    ) -> List[str]:
        if joints is None:
            joints = self.joints
        links = []
        for joint in joints:
            child_link = self.get_joint_child(joint)
            if include_roots:
                links.append(child_link)
            links.extend(self.get_link_descendants(child_link))
        return remove_duplicates(links)

    def get_joint_chain(self, link: Optional[str] = None) -> Iterable[str]:
        if link is None:
            link = self.root_link
        cluster_links = self.get_link_cluster(link)
        cluster_joints = remove_duplicates(flatten(map(self.get_link_children, cluster_links)))
        movable_joints = list(filter(negate_test(self.is_fixed), cluster_joints))
        if len(movable_joints) != 1:
            return
        [joint] = movable_joints
        child_link = self.get_joint_child(joint)
        yield joint
        yield from self.get_joint_chain(child_link)

    def get_arm_joints(self, **kwargs: Any) -> List[str]:
        return list(self.get_joint_chain(**kwargs))

    def get_gripper_joints(self, **kwargs: Any) -> List[str]:
        arm_joints = self.get_arm_joints(**kwargs)
        descendant_joints = remove_duplicates(
            map(
                self.get_link_parent,
                flatten(
                    map(self.get_link_descendants, map(self.get_joint_child, arm_joints)),
                ),
            )
        )
        active_joints = filter(negate_test(self.is_fixed), descendant_joints)
        return list(filter(negate_test(arm_joints.__contains__), active_joints))

    def get_tool_links(self, **kwargs: Any) -> List[str]:
        arm_joints = self.get_arm_joints(**kwargs)
        if not arm_joints:
            return []
        child_link = self.get_joint_child(arm_joints[-1])
        return [
            link for link in self.get_link_cluster(child_link) if not self.get_link_children(link)
        ]

    def get_active_joints(self, link: str) -> List[str]:
        ancestor_links = self.get_link_ancestors(link) + [link]
        ancestor_joints = set(map(self.get_link_parent, ancestor_links))
        active_joints = list(filter(ancestor_joints.__contains__, self.joints))
        return active_joints

    def get_joint_edge(self, joint: str) -> Tuple[str, str]:
        return self.get_joint_parent(joint), self.get_joint_child(joint)

    @property
    def rigid_edges(self) -> List[Tuple[str, str]]:
        return list(map(self.get_joint_edge, self.fixed_joints))

    def get_link_cluster(
        self, link: str, edges: Optional[List[Tuple[str, str]]] = None
    ) -> List[str]:
        if edges is None:
            edges = self.rigid_edges
        return get_reachable(edges, source_vertices=[link])

    def get_link_clusters(self, **kwargs: Any) -> List[List[str]]:
        clusters = []
        clustered_links = set()
        for link in self.links:
            if link not in clustered_links:
                cluster = self.get_link_cluster(link, **kwargs)
                clusters.append(cluster)
                clustered_links.update(cluster)
        return clusters

    @property
    def node_edges(self) -> List[Tuple[str, str]]:
        movable_joints = list(filter(negate_test(self.is_fixed), self.all_joints))
        movable_edges = {
            (self.get_joint_parent(joint), self.get_joint_child(joint)) for joint in movable_joints
        }
        return [edge for edge in self.graph.transforms.edge_data if edge not in movable_edges]

    def get_node_cluster(self, node: str) -> List[str]:
        return get_reachable(self.node_edges, source_vertices=[node])

    def get_joint_lower(self, joint: str) -> float:
        limit = self.joint_map[joint].limit
        if limit is None:
            return -math.inf
        if limit.lower is None:
            return -math.inf
        return limit.lower

    def get_joint_upper(self, joint: str) -> float:
        limit = self.joint_map[joint].limit
        if limit is None:
            return +math.inf
        if limit.upper is None:
            return +math.inf
        return limit.upper

    def get_joint_bound(self, joint: str) -> Tuple[float, float]:
        lower = self.get_joint_lower(joint)
        upper = self.get_joint_upper(joint)
        bound = (lower, upper)
        return bound

    def get_joint_bounds(self, joints: Optional[List[str]] = None) -> Tuple[Conf, Conf]:
        if joints is None:
            joints = self.joints
        return tuple(zip(*map(self.get_joint_bound, joints)))

    @property
    def joint_bounds(self) -> Tuple[Conf, Conf]:
        return self.get_joint_bounds()

    @property
    def joint_lower(self) -> Conf:
        lower, _ = self.joint_bounds
        return lower

    @property
    def joint_upper(self) -> Conf:
        _, upper = self.joint_bounds
        return upper

    def is_fixed(self, joint: str) -> bool:
        return self.get_joint_type(joint) == "fixed"

    def is_circular(self, joint: str) -> bool:
        return self.get_joint_type(joint) == "continuous"

    def get_circular_upper(self, joints: Optional[List[str]] = None) -> Conf:
        if joints is None:
            joints = self.joints
        return np.array([math.pi if self.is_circular(joint) else math.inf for joint in joints])

    @property
    def circular_upper(self) -> Conf:
        return self.get_circular_upper()

    def get_circular_bounds(self, joints: Optional[List[str]] = None) -> Tuple[Conf, Conf]:
        if joints is None:
            joints = self.joints
        lower, upper = self.get_joint_bounds(joints)
        lower = np.maximum(-np.array(self.circular_upper), lower)
        upper = np.minimum(upper, +np.array(self.circular_upper))
        return lower, upper

    @property
    def circular_bounds(self) -> Tuple[Conf, Conf]:
        return self.get_circular_bounds()

    def get_joint_position(self, joint: str) -> float:
        return self.get_joint_positions([joint])[0]

    def get_joint_positions(self, joints: Optional[List[str]] = None) -> Conf:
        if joints is None:
            joints = self.joints
        indices = self.get_joint_indices(joints)
        return [self.cfg[index] for index in indices]

    def get_joint_state(self, joints: Optional[List[str]] = None) -> Dict[str, float]:
        if joints is None:
            joints = self.joints
        positions = self.get_joint_positions(joints)
        return compute_mapping(joints, positions)

    def get_conf(self) -> Conf:
        return self.get_joint_positions()

    def set_joint_position(self, joint: str, position: float) -> None:
        self.set_joint_positions(joints=[joint], positions=[position])

    def set_joint_positions(
        self, joints: Optional[List[str]] = None, positions: Optional[Conf] = None
    ) -> Dict[str, float]:
        if positions is None:
            return {}
        if joints is None:
            joints = self.joints
        assert len(joints) == len(positions), (len(joints), len(positions))
        joint_positions = compute_mapping(joints, positions)
        self.update_cfg(joint_positions)
        return joint_positions

    def set_joint_state(self, joint_state: Dict[str, float]) -> None:
        if not joint_state:
            return
        joints, positions = zip(*joint_state.items())
        self.set_joint_positions(joints, positions)

    def set_conf(self, conf: Conf) -> Dict[str, float]:
        return self.set_joint_positions(positions=conf)

    def sample_joint_positions(self, joints: Optional[List[str]] = None) -> Conf:
        return np.random.uniform(*self.get_circular_bounds(joints))

    def sample_conf(self) -> Conf:
        return self.sample_joint_positions()

    def randomize_joint_positions(self, joints: Optional[List[str]] = None) -> Conf:
        if joints is None:
            joints = self.joints
        positions = self.sample_joint_positions(joints)
        self.set_joint_positions(joints, positions)
        return positions

    def randomize_conf(self) -> Conf:
        return self.randomize_joint_positions()

    def update(self) -> None:
        self._create_maps()
        self._update_actuated_joints()

    def clone(self) -> "Yourdf":
        return Yourdf(
            self.robot, urdf_path=self.urdf_path, mesh_dir=self.mesh_dir, metadata=self.metadata
        )

    def show(self, collision: bool = False, flags: Optional[Dict[str, Any]] = None, **kwargs: Any):
        if flags is None:
            flags = {"axis": "world", "grid": True}
        scene = self._scene_collision if collision else self.scene
        if scene is not None:
            scene.show(flags=flags, **kwargs)

    def dump(self, joints: Optional[List[str]] = None) -> None:
        if joints is None:
            joints = self.joints
        print(f"Robot: {self.name}")
        print(f"Base link: {self.base_link}")
        print(f"Links ({len(self.links)}): {self.links}")
        print(f"Tool links ({len(self.get_tool_links())}): {self.get_tool_links()}")
        print(f"Arm joints ({len(self.get_arm_joints())}): {self.get_arm_joints()}")
        for joint in joints:
            lower, upper = self.get_joint_bound(joint)
            print(
                f"Joint: {self.get_joint_index(joint)}/{self.num_dofs} | Name: {joint} | Type:"
                f" {self.get_joint_type(joint)} | Parent: {self.get_joint_parent(joint)} | Child:"
                f" {self.get_joint_child(joint)} | Position: {self.get_joint_position(joint):.3f} |"
                f" Lower: {lower:.3f} | Upper: {upper:.3f}"
            )

    def apply_prefix(self, prefix: str) -> None:
        self.prefix = f"{prefix}{self.prefix}"
        self.robot.name = f"{prefix}{self.robot.name}"
        self._base_link = f"{prefix}{self._base_link}"
        for link in self.robot.links:
            link.name = f"{prefix}{link.name}"
        for joint in self.robot.joints:
            joint.name = f"{prefix}{joint.name}"
            joint.parent = f"{prefix}{joint.parent}"
            joint.child = f"{prefix}{joint.child}"
        self.update()

    def add_frame(self, frame: str, parent: str, pose: Optional[Pose] = None):
        if pose is None:
            pose = np.identity(4)
        self.robot.links.append(Link(name=frame))
        self.robot.joints.append(
            Joint(
                name=f"{frame}_fixed",
                type="fixed",
                parent=parent,
                child=frame,
                origin=pose,
            )
        )
        self.update()

    def export(self, urdf_path: Optional[str] = None) -> str:
        if urdf_path is None:
            dir_name, _ = os.path.splitext(os.path.basename(__file__))
            temp_dir = f"/tmp/{dir_name}"
            os.makedirs(temp_dir, exist_ok=True)
            file_name = f"{self.name}.urdf"
            urdf_path = os.path.abspath(os.path.join(temp_dir, file_name))

        self.write_xml_file(urdf_path)
        self.logger.info(f"Exported: {urdf_path}")
        return urdf_path

    def state(self, joints: Optional[List[str]] = None) -> "YourdfState":
        if joints is None:
            joints = self.joints
        return YourdfState(self, self.get_joint_state(joints))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    __repr__ = __str__


class YourdfState:
    def __init__(self, yourdf: Yourdf, joint_state: Dict[str, float]) -> None:
        self.yourdf = yourdf
        self.joint_state = dict(joint_state)

    @property
    def joints(self) -> List[str]:
        return list(self.joint_state)

    def set(self) -> None:
        self.yourdf.set_joint_state(self.joint_state)

    def show(self, **kwargs: Any) -> None:
        self.set()
        self.yourdf.show(**kwargs)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.set()

    def __str__(self):
        return f"{self.__class__.__name__}({self.joints})"

    __repr__ = __str__


def absolute_urdf_path(urdf_path: str) -> str:
    if os.path.exists(urdf_path):
        return os.path.abspath(urdf_path)
    # Third Party
    from curobo.util_file import get_assets_path

    assets_dir = get_assets_path()
    return os.path.join(assets_dir, urdf_path)


def load_robot_yourdf(urdf_path: str, **kwargs: Any) -> Yourdf:
    urdf_path = absolute_urdf_path(urdf_path)
    return Yourdf(urdf_path=urdf_path, **kwargs)


def combine_yourdfs(yourdfs: List[Yourdf], poses: List[Pose]) -> Yourdf:
    assert yourdfs
    name = "-".join(yourdf.name for yourdf in yourdfs)
    new_robot = Robot(name=name)
    world_link = Link(name=WORLD_LINK)
    new_robot.links.append(world_link)

    for i, (yourdf, pose) in enumerate(safe_zip(yourdfs, poses)):
        robot_name = yourdf.robot.name
        base_link = yourdf.link_map[yourdf.base_link]
        [matrix] = pose.get_numpy_matrix()

        for link in yourdf.robot.links:
            link = copy.copy(link)
            new_robot.links.append(link)
            if link == base_link:
                joint = Joint(
                    name=f"world_{robot_name}_fixed",
                    type="fixed",
                    parent=world_link.name,
                    child=link.name,
                    origin=matrix,
                )
                new_robot.joints.append(joint)

        for joint in yourdf.robot.joints:
            joint = copy.copy(joint)
            new_robot.joints.append(joint)

    new_robot.materials.extend(yourdfs[0].robot.materials)
    composite_yourdf = Yourdf(new_robot, mesh_dir=yourdfs[0].mesh_dir)
    return composite_yourdf


LINEAR_DOFS = ["x", "y", "z"]
ANGULAR_DOFS = ["roll", "pitch", "yaw"]
BASE_DOFS = LINEAR_DOFS + ANGULAR_DOFS


def add_base_dofs(
    yourdf: Yourdf,
    dofs: Optional[Tuple[str]] = ("x", "y", "yaw"),
    dof_limits: Optional[Dict[str, float]] = None,
    linear_velocity: float = 1.0,
    angular_velocity: float = PI / 2,
    effort: float = 1e3,
) -> List[str]:
    dof_limits = dof_limits or {}
    new_joints = []
    for name in dofs:
        axis = np.zeros(3)
        if name in LINEAR_DOFS:
            k = LINEAR_DOFS.index(name)
            axis[k] = 1
            lower, upper = dof_limits.get(name, (-1.0, +1.0))
            new_joints.append(
                Joint(
                    name=name,
                    type="prismatic",
                    axis=axis,
                    limit=Limit(
                        lower=lower,
                        upper=upper,
                        velocity=linear_velocity,
                        effort=effort,
                    ),
                )
            )
        elif name in ANGULAR_DOFS:
            k = ANGULAR_DOFS.index(name)
            axis[k] = 1
            new_joints.append(
                Joint(
                    name="yaw",
                    type="continuous",
                    axis=axis,
                    limit=Limit(velocity=angular_velocity, effort=effort),
                )
            )
        else:
            raise ValueError(name)

    if not new_joints:
        return new_joints

    new_links = [Link(name=f"_base_link_{joint.name}") for joint in new_joints]
    base_link = yourdf.link_map[yourdf.base_link]
    child_links = new_links[1:] + [base_link]
    for parent, joint, child in zip(new_links, new_joints, child_links):
        joint.parent = parent.name
        joint.child = child.name
    yourdf.robot.links = new_links + yourdf.robot.links
    yourdf.robot.joints = new_joints + yourdf.robot.joints
    yourdf.update()

    return new_joints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf", type=str, help="The URDF path.")
    parser.add_argument("-c", "--convex", action="store_true", help="Convexifies the robot.")
    args = parser.parse_args()
    print("Args:", args)

    yourdf = Yourdf(urdf_path=args.urdf)
    if args.convex:
        yourdf.convexify()
    yourdf.dump()
    yourdf.set_conf(np.average(yourdf.joint_bounds, axis=0))
    print("Joint state:", yourdf.get_joint_state())
    yourdf.show()


if __name__ == "__main__":
    main()
