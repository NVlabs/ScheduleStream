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
from functools import cache
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

# Third Party
import curobo.wrap.reacher.ik_solver
import numpy as np
import structlog
import torch
import trimesh
from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelState
from curobo.cuda_robot_model.kinematics_parser import KinematicsParser
from curobo.cuda_robot_model.types import (
    CSpaceConfig,
    JointLimits,
    KinematicsTensorConfig,
    SelfCollisionKinematicsConfig,
)
from curobo.geom.sdf.world import CollisionQueryBuffer, WorldCollision
from curobo.geom.types import Obstacle, SphereFitType, WorldConfig
from curobo.rollout.arm_reacher import ArmReacher
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.sample_lib import HaltonGenerator
from curobo.util.torch_utils import get_torch_jit_decorator
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# NVIDIA
from schedulestream.applications.custream.command import Trajectory
from schedulestream.applications.custream.config import extract_robot_config, load_robot_yaml
from schedulestream.applications.custream.ik_wrapper import IKWrapper
from schedulestream.applications.custream.object import Object
from schedulestream.applications.custream.spheres import add_spheres
from schedulestream.applications.custream.state import Attachment, Configuration, State
from schedulestream.applications.custream.utils import (
    INF,
    PI,
    Vector,
    create_frame,
    create_pose,
    extract_joint_state,
    matrix_from_pose,
    multiply_poses,
    to_cpu,
    to_pose,
    transform_spheres,
)
from schedulestream.applications.custream.yourdf import load_robot_yourdf
from schedulestream.common.utils import (
    Context,
    batched,
    current_time,
    elapsed_time,
    fill_batch,
    flatten,
    get_length,
    remove_duplicates,
    safe_zip,
)

Conf = Vector
# Standard Library
from dataclasses import dataclass


@dataclass
class TAMPConfig:
    approach_link_distance: float = 0.03
    approach_object_distance: float = 0.02
    approach_position_step: float = 0.10
    approach_orientation_step: float = PI / 16
    approach_velocity_scale: float = 0.5


def create_world_config(objects: Optional[List[Object]] = None) -> Optional[WorldConfig]:
    if not objects:
        return None
    world_config = WorldConfig()
    for obj in objects:
        world_config.add_obstacle(obj)
    return world_config


class ActiveContext(Context):
    def __init__(
        self, world: "World", objects: Optional[Tuple[str]] = (), links: Optional[List[str]] = ()
    ):
        if objects is None:
            objects = world.object_names
        if links is None:
            links = world.links
        self.world = world
        self.active_objects = {obj: world.is_object_active(obj) for obj in objects}
        self.active_links = {link: world.is_link_active(link) for link in links}
        torch.cuda.synchronize()

    @property
    def objects(self) -> List[str]:
        return list(self.active_objects)

    @property
    def links(self) -> List[str]:
        return list(self.active_links)

    def set(self) -> None:
        for obj, active in self.active_objects.items():
            self.world.set_object_active(obj, active)
        for link, active in self.active_links.items():
            self.world.set_link_active(link, active)
        torch.cuda.synchronize()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(objects={self.objects}, links={self.links})"

    __repr__ = __str__


class World(MotionGen):
    def __init__(
        self,
        robot_config: Union[str, dict, RobotConfig],
        objects: Optional[List[Object]] = None,
        *args: Any,
        tamp_config: Optional[TAMPConfig] = None,
        num_attached_spheres: Optional[int] = None,
        batch_size: Optional[int] = None,
        convexify: bool = False,
        visualize_spheres: bool = False,
        debug: bool = True,
        **kwargs: Any,
    ):
        self.logger = structlog.get_logger()

        if tamp_config is None:
            tamp_config = TAMPConfig()
        self.tamp_config = tamp_config
        world_config = create_world_config(objects)
        if batch_size is not None:
            batch_size = int(batch_size)
        self.batch_size = batch_size

        self.robot_yaml = load_robot_yaml(robot_config)
        self.robot_config = self.robot_yaml["robot_cfg"]
        self.yourdf = load_robot_yourdf(self.kinematics_yaml["urdf_path"])
        if convexify:
            self.yourdf.convexify()

        self.kinematics_yaml["base_link"] = self.yourdf.base_link
        if num_attached_spheres is not None:
            self.kinematics_yaml["extra_collision_spheres"] = {
                attached_link: num_attached_spheres for attached_link in self.attached_links
            }

        self.gripper_limits = self.kinematics_yaml.pop("gripper_limits", {})
        self.robot_world_config = RobotWorldConfig.load_from_config(
            robot_config=self.robot_config,
            world_model=world_config,
            collision_activation_distance=0.0,
        )
        self.robot_world = RobotWorld(self.robot_world_config)

        ik_solver_config = dict(
            self_collision_check=True,
            self_collision_opt=False,
        )
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_config, *args, world_model=world_config, **kwargs, **ik_solver_config
        )
        super().__init__(self.motion_gen_config)
        self.ik_wrapper = IKWrapper(self.ik_solver)
        for obj in self.objects:
            obj.world = self

        self.add_objects(self.scene, visualize_spheres=visualize_spheres)
        if visualize_spheres:
            self.add_robot_spheres(self.scene)
        if debug:
            for link in self.tool_links:
                self.scene.add_geometry(create_frame(), parent_node_name=link)
        camera_pose = multiply_poses(create_pose(roll=3 * np.pi / 8))
        self.set_camera_pose(camera_pose)

        self.attached_objects = {}
        self.dirty_poses = set()
        self.active_links = {}
        self.active_objects = {}
        if self.world_collision is not None:
            self._enable_collision = self.world_collision.enable_obstacle
            self.world_collision.enable_obstacle = self.wrapped_enable_collision

        for obj in self.objects:
            if not obj.collider:
                self.set_object_active(obj.name, active=False)

    def set(self) -> None:
        for attachment in self.attached_objects.values():
            attachment.set()

    @cache
    def _get_ik_wrapper(self, tool_links: Tuple[str]) -> IKWrapper:
        ik_solver_config = dict()
        robot_config = extract_robot_config(
            self.robot_config, list(tool_links), inactive_joints=None
        )
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_config,
            world_model=self.world_config,
            tensor_args=self.tensor_args,
            world_coll_checker=self.world_collision,
            **ik_solver_config,
        )
        ik_solver = IKSolver(ik_config)
        return IKWrapper(ik_solver)

    def get_ik_wrapper(self, tool_links: List[str]) -> IKWrapper:
        if tool_links == self.tool_links:
            return self.ik_wrapper
        return self._get_ik_wrapper(tuple(tool_links))

    def wrapped_enable_collision(self, name: str, enable: bool = True) -> None:
        self.active_objects[name] = enable
        return self._enable_collision(name, enable=enable)

    def initialize(self, batch_size: int = 1, iterations: int = 3) -> None:
        link_poses = self.get_link_poses()
        batch_poses = {link: pose.repeat(batch_size).clone() for link, pose in link_poses.items()}
        self.logger.info(
            f"Initializing IK for links {list(batch_poses)} with batch size {batch_size} for"
            f" {iterations} iterations..."
        )
        for _ in range(iterations):
            self.inverse_kinematics(batch_poses, seed=None)

    def add_robot_spheres(self, scene: Optional[trimesh.Scene] = None) -> trimesh.Scene:
        if scene is None:
            scene = trimesh.Scene()
        for link in self.links:
            add_spheres(self.get_link_spheres(link), scene=scene, parent_node_name=link)
        return scene

    def objects_scene(
        self, names: Optional[List[str]] = None, scene: Optional[trimesh.Scene] = None
    ) -> trimesh.Scene:
        if names is None:
            names = self.object_names
        if scene is None:
            scene = trimesh.Scene()
        for name in names:
            scene = self.get_object(name).scene(scene=scene)
        return scene

    def active_scene(self, **kwargs: Any) -> trimesh.Scene:
        return self.objects_scene(names=self.get_active_objects(), **kwargs)

    def add_objects(
        self, scene: Optional[trimesh.Scene] = None, visualize_spheres: bool = False
    ) -> trimesh.Scene:
        if scene is None:
            scene = trimesh.Scene()
        world_scene = self.get_world_scene()
        for node_name in world_scene.graph.nodes_geometry:
            matrix, _ = world_scene.graph.get(node_name, frame_from=None)
            if node_name not in self.scene.geometry:
                mesh = world_scene.geometry[node_name]
                scene.add_geometry(
                    mesh,
                    node_name=node_name,
                    geom_name=node_name,
                    transform=matrix,
                )
            scene.graph.update(frame_to=node_name, frame_from=None, matrix=matrix)

            if not visualize_spheres:
                continue

            obj = self.get_object(node_name)
            if obj.movable:
                for sphere in obj.get_bounding_spheres():
                    mesh = sphere.get_trimesh_mesh()
                    matrix = matrix_from_pose(to_pose(sphere.pose))
                    scene.add_geometry(mesh, parent_node_name=node_name, transform=matrix)
        return scene

    def set_camera_pose(self, pose: Pose) -> None:
        camera_matrix = matrix_from_pose(pose)
        camera_pose = self.scene.camera.look_at(
            points=trimesh.bounds.corners(self.scene.bounds),
            rotation=camera_matrix,
            center=np.average(self.scene.bounds, axis=0) / 2,
            pad=None,
        )
        self.scene.camera_transform = camera_pose

    @property
    def device(self) -> torch.device:
        return self.tensor_args.device

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor_args.dtype

    def to_device(self, array: np.ndarray) -> torch.Tensor:
        return self.tensor_args.to_device(array)

    def to_cpu(self, tensor: torch.Tensor) -> np.ndarray:
        return to_cpu(tensor)

    @property
    def time_step(self) -> float:
        return self.interpolation_dt

    @property
    def kinematics_yaml(self) -> dict:
        return self.robot_yaml["robot_cfg"]["kinematics"]

    @property
    def cspace_yaml(self) -> dict:
        return self.kinematics_yaml["cspace"]

    @property
    def scene(self) -> trimesh.Scene:
        return self.yourdf.scene

    @property
    def robot_model(self) -> CudaRobotModel:
        return self.kinematics

    @property
    def kinematics_config(self) -> KinematicsTensorConfig:
        return self.robot_model.kinematics_config

    @property
    def kinematics_parser(self) -> KinematicsParser:
        return self.robot_model.kinematics_parser

    @property
    def robot_generator_config(self) -> CudaRobotGeneratorConfig:
        return self.robot_model.generator_config

    @property
    def cspace_config(self) -> CSpaceConfig:
        return self.robot_model.cspace

    @property
    def self_collision_config(self) -> SelfCollisionKinematicsConfig:
        return self.robot_model.self_collision_config

    @property
    def arm_reacher(self) -> ArmReacher:
        return self.ik_solver.rollout_fn

    @property
    def sampler(self) -> HaltonGenerator:
        return self.ik_solver.q_sample_gen

    @property
    def world_collision(self) -> Optional[WorldCollision]:
        return self.world_coll_checker

    @property
    def world_config(self) -> Optional[WorldConfig]:
        if self.world_collision is None:
            return None
        return self.world_collision.world_model

    @property
    def nodes(self) -> List[str]:
        return self.yourdf.nodes

    def get_node_pose(self, node: str, parent_node: Optional[str] = None) -> Pose:
        return Pose.from_matrix(self.yourdf.get_transform(node, frame_from=parent_node))

    def get_link_arm(self, tool_link: str) -> str:
        assert tool_link in self.tool_links
        if len(self.tool_links) == 1:
            return "arm"
        return tool_link.split("_")[0]

    @property
    def arms(self) -> List[str]:
        return list(map(self.get_link_arm, self.tool_links))

    @property
    def arm_names(self) -> List[str]:
        return self.arms

    @property
    def default_arm(self) -> str:
        return self.arms[0]

    def get_arm_link(self, arm: str) -> Optional[str]:
        link_from_arm = {self.get_link_arm(link): link for link in self.tool_links}
        return link_from_arm[arm]

    def get_arm_base_link(self, arm: str) -> Optional[str]:
        root_joint = self.get_arm_joints(arm)[0]
        return self.yourdf.get_joint_parent(root_joint)

    def get_tool_joints(self, tool_link: str) -> List[str]:
        return self.yourdf.get_active_joints(tool_link)

    def get_arm_joints(self, arm: str) -> List[str]:
        tool_link = self.get_arm_link(arm)
        if tool_link is None:
            return []
        return list(filter(self.active_joints.__contains__, self.get_tool_joints(tool_link)))

    def get_gripper_joints(self, arm: str) -> List[str]:
        arm_joints = self.get_arm_joints(arm)[-1:]
        descendant_links = set()
        for joint in arm_joints:
            link = self.yourdf.get_joint_parent(joint)
            descendant_links.update(self.yourdf.get_link_descendants(link))
        descendant_joints = set(map(self.yourdf.get_link_parent, descendant_links))
        gripper_joints = descendant_joints - set(arm_joints)
        return list(filter(gripper_joints.__contains__, self.all_joints))

    def get_finger_joints(self, arm: str) -> List[str]:
        joints_from_parent = {}
        for joint in self.yourdf.robot.joints:
            if joint.name in self.locked_joints:
                joints_from_parent.setdefault(joint.parent, []).append(joint)

        tool_link = self.get_arm_link(arm)
        finger_joints = []
        for link in self.yourdf.get_link_cluster(tool_link):
            for joint in joints_from_parent.get(link, []):
                finger_joints.append(joint.name)
        return finger_joints

    @property
    def objects(self) -> List[Object]:
        if self.world_config is None:
            return []
        return self.world_config.objects

    @property
    def movable_objects(self) -> List[Object]:
        return [obj for obj in self.objects if obj.movable]

    @property
    def fixed_objects(self) -> List[Object]:
        return [obj for obj in self.objects if obj not in self.movable_objects]

    @property
    def obstacle_objects(self) -> List[Object]:
        return [obj for obj in self.fixed_objects if obj.collider]

    @property
    def object_names(self) -> List[str]:
        return [obj.name for obj in self.objects]

    @property
    def movable_names(self) -> List[str]:
        return [obj.name for obj in self.movable_objects]

    @property
    def fixed_names(self) -> List[str]:
        return [obj.name for obj in self.fixed_objects]

    @property
    def obstacle_names(self) -> List[str]:
        return [obj.name for obj in self.obstacle_objects]

    def is_object(self, name: str) -> bool:
        return str(name) in self.object_names

    def get_object(self, name: str) -> Object:
        obj = self.world_config.get_obstacle(str(name))
        assert obj is not None, (name, self.object_names)
        return obj

    def get_object_pose(self, name: str) -> Pose:
        obj = self.get_object(name)
        return Pose.from_list(obj.pose, tensor_args=self.tensor_args)

    def set_object_pose(self, name: str, pose: Pose, collision: bool = True) -> None:
        assert len(pose) == 1
        if collision:
            self.world_collision.update_obstacle_pose(name, pose, update_cpu_reference=True)
        self.scene.graph.update(frame_to=name, frame_from=None, matrix=matrix_from_pose(pose))
        self.dirty_poses.add(name)

    def is_object_active(self, name: str) -> bool:
        return self.active_objects.get(str(name), True)

    def get_active_objects(self) -> List[str]:
        return list(filter(self.is_object_active, self.object_names))

    def set_object_active(self, name: str, active: bool, **kwargs: Any) -> None:
        name = str(name)
        if self.is_object_active(name) != active:
            self.world_collision.enable_obstacle(name, enable=active, **kwargs)

    def set_objects_active(
        self, active: bool, names: Optional[List[str]] = None, **kwargs: Any
    ) -> None:
        if names is None:
            names = self.object_names
        for name in names:
            self.set_object_active(name, active, **kwargs)

    def enable_objects_active(self, active_names: Iterable[str]) -> None:
        active_names = list(active_names)
        inactive_names = [name for name in self.object_names if name not in active_names]
        self.set_objects_active(names=inactive_names, active=False)
        self.set_objects_active(names=active_names, active=True)

    def get_world_scene(self, **kwargs: Any) -> trimesh.Scene:
        if self.world_config is None:
            return trimesh.Scene()
        return WorldConfig.get_scene_graph(self.world_config, **kwargs)

    def active_context(self, *args: Any, **kwargs: Any) -> ActiveContext:
        return ActiveContext(self, *args, **kwargs)

    def trajectory(self, *args: Any, **kwargs: Any) -> Trajectory:
        return Trajectory(self, *args, **kwargs)

    def configuration(
        self, joint_state: Optional[JointState] = None, *args: Any, **kwargs: Any
    ) -> Configuration:
        if joint_state is None:
            joint_state = self.joint_state
        return Configuration(self, joint_state, *args, **kwargs)

    def arm_configuration(self, arm: Optional[str] = None) -> Configuration:
        if arm is None:
            arm = self.arms[0]
        joints = self.get_arm_joints(arm)
        joint_state = self.get_joint_state(joints)
        return self.configuration(joint_state)

    def attachment(
        self,
        obj: str,
        pose: Optional[Pose] = None,
        parent: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Attachment:
        if pose is None:
            pose = self.get_object_pose(obj)
            assert parent is None
        return Attachment(self, obj, pose, parent, *args, **kwargs)

    def state(
        self, joints: Optional[List[str]] = None, objects: Optional[List[str]] = None, **kwargs: Any
    ) -> State:
        assert not self.attached_objects
        if joints is None:
            joints = self.all_joints
        conf = self.configuration(self.get_joint_state(joints=joints))
        if objects is None:
            objects = self.object_names
        attachments = [self.attachment(obj) for obj in objects]
        return State(self, conf=conf, attachments=attachments, **kwargs)

    @property
    def urdf_path(self) -> str:
        return self.robot_generator_config.urdf_path

    @property
    def usd_path(self) -> str:
        return self.robot_generator_config.usd_path

    @property
    def robot_name(self) -> str:
        return self.yourdf.name

    @property
    def link_from_name(self) -> Dict[str, int]:
        return self.kinematics_config.link_name_to_idx_map

    @property
    def links(self) -> List[str]:
        name_from_index = {index: name for name, index in self.link_from_name.items()}
        return list(map(name_from_index.get, range(len(name_from_index))))

    @property
    def base_link(self) -> str:
        return self.robot_model.base_link

    @property
    def ee_link(self) -> str:
        return self.robot_model.ee_link

    @property
    def tool_links(self) -> List[str]:
        return self.robot_model.link_names

    @property
    def attached_links(self):
        return list(self.kinematics_yaml["extra_links"])

    def get_sphere_indices(self, link: str) -> torch.Tensor:
        return self.kinematics_config.get_sphere_index_from_link_name(link)

    def get_sphere_link(self, sphere_index: int) -> str:
        link_index = self.kinematics_config.link_sphere_idx_map[sphere_index]
        return self.links[link_index]

    @cache
    def get_link_spheres(self, link: str) -> torch.Tensor:
        return self.kinematics_config.get_link_spheres(link)

    def get_rigid_links(self, link: str, inactive: bool = False) -> List[str]:
        edges = list(self.yourdf.rigid_edges)
        if inactive:
            edges.extend(map(self.yourdf.get_joint_edge, self.locked_joints))
        return self.yourdf.get_link_cluster(link, edges=edges)

    def get_rigid_spheres(self, link: str, valid: bool = True, **kwargs) -> torch.Tensor:
        links = self.get_rigid_links(link, **kwargs)
        spheres = self.to_device(torch.empty(size=(1, 0, 4)))
        for _link in links:
            pose = self.get_node_pose(_link, parent_node=link)
            link_spheres = transform_spheres(pose, self.get_link_spheres(_link))
            spheres = torch.cat([spheres, link_spheres], dim=1)
        if valid:
            spheres = spheres[spheres[..., -1] > 0, ...]
        return spheres

    def dump_spheres(self):
        for i, link in enumerate(self.links):
            spheres = self.get_link_spheres(link)
            active_spheres = [sphere for sphere in spheres if sphere[-1] > 0.0]
            print(
                f"{i}/{len(self.links)} Name: {link} | Spheres: {len(spheres)} | Active:"
                f" {len(active_spheres)}"
            )

    def is_link_active(self, link: str) -> bool:
        return self.active_links.get(link, True)

    def set_link_active(self, link: str, active: bool) -> None:
        if self.is_link_active(link) == active:
            return
        if active:
            self.kinematics_config.enable_link_spheres(link)
        else:
            self.kinematics_config.disable_link_spheres(link)
        self.active_links[link] = active

    def set_links_active(self, active: bool, links: Optional[list] = None) -> None:
        if links is None:
            links = self.links
        for link in links:
            self.set_link_active(link, active)

    @property
    def all_joints(self) -> List[str]:
        return self.robot_model.all_articulated_joint_names

    @property
    def active_joints(self) -> List[str]:
        return self.joint_names

    @property
    def locked_joints(self) -> List[str]:
        if self.locked_joint_state is None:
            return []
        return self.locked_joint_state.joint_names

    @property
    def arm_joints(self) -> List[str]:
        return self.active_joints

    @property
    def dofs(self) -> int:
        return len(self.active_joints)

    @property
    def gripper_joints(self) -> List[str]:
        gripper_joints = set(flatten(map(self.get_gripper_joints, self.arms)))
        return list(filter(gripper_joints.__contains__, self.all_joints))

    def get_joint_positions(self, joints: Optional[List[str]] = None) -> Conf:
        if joints is None:
            joints = self.active_joints
        return self.yourdf.get_joint_positions(joints)

    @property
    def conf(self) -> Conf:
        return self.get_joint_positions()

    def to_joint_state(
        self, confs: Optional[List[Conf]] = None, joints: Optional[List[str]] = None
    ) -> JointState:
        if joints is None:
            joints = self.active_joints
        if confs is None:
            confs = [self.get_joint_positions(joints)]
        joint_state = JointState.from_position(self.to_device(confs).clone(), joint_names=joints)
        return joint_state

    def get_joint_state(self, joints: Optional[List[str]] = None) -> JointState:
        return self.to_joint_state(joints=joints)

    @property
    def all_joint_state(self) -> JointState:
        return self.to_joint_state(joints=self.all_joints)

    @property
    def joint_state(self) -> JointState:
        return self.to_joint_state(joints=self.active_joints)

    @property
    def locked_joint_state(self) -> JointState:
        return self.kinematics_config.lock_jointstate

    def _set_locked_joint_positions(
        self, joints: Optional[List[str]] = None, positions: Optional[Conf] = None
    ) -> None:
        if positions is None:
            return
        if joints is None:
            joints = self.locked_joints
        assert len(joints) == len(positions)
        if not joints:
            return
        joint_state = self.locked_joint_state
        joint_positions = dict(safe_zip(joint_state.joint_names, to_cpu(joint_state.position)))
        joint_positions.update(dict(safe_zip(joints, positions)))
        new_positions = self.to_device(np.array(list(map(joint_positions.get, self.locked_joints))))
        if torch.allclose(self.locked_joint_state.position, new_positions):
            return
        self.update_locked_joints(joint_positions, self.robot_config)
        self.locked_joint_state.position.copy_(new_positions)

    def set_joint_positions(
        self,
        joints: Optional[List[str]] = None,
        positions: Optional[Conf] = None,
        locked: bool = True,
    ) -> None:
        if positions is None:
            return
        if joints is None:
            joints = self.active_joints
        joint_positions = self.yourdf.set_joint_positions(joints, to_cpu(positions))
        if not locked:
            return
        locked_joints = [joint for joint in joints if joint in self.locked_joints]
        locked_positions = list(map(joint_positions.get, locked_joints))
        self._set_locked_joint_positions(locked_joints, locked_positions)

    def set_conf(self, conf, **kwargs: Any) -> None:
        self.set_joint_positions(positions=to_cpu(conf), **kwargs)

    def set_joint_state(self, joint_state: JointState, **kwargs: Any) -> None:
        position = joint_state.position
        if len(position.shape) != 1:
            position = position.squeeze(0)
        self.set_joint_positions(joint_state.joint_names, position, **kwargs)

    @property
    def retract_conf(self) -> torch.Tensor:
        return self.get_retract_config()

    def set_retract_conf(self, conf: Optional[Conf] = None) -> None:
        if conf is None:
            conf = self.conf
        self.cspace_config.retract_config[:] = self.to_device(conf)

    def get_joint_bound(self, joint: str) -> Tuple[float, float]:
        return self.yourdf.get_joint_bound(joint)

    def get_joint_bounds(self, joints: Optional[List[str]] = None) -> Tuple[Conf, Conf]:
        if joints is None:
            joints = self.active_joints
        return self.yourdf.get_joint_bounds(joints)

    def get_gripper_limit(self, joint: str) -> Tuple[float, float]:
        if joint in self.gripper_limits:
            return self.gripper_limits[joint]
        return self.get_joint_bound(joint)

    def get_gripper_limits(self, joints: Optional[List[str]] = None) -> Tuple[Conf, Conf]:
        if joints is None:
            joints = self.gripper_joints
        if not joints:
            return [], []
        return zip(*map(self.get_gripper_limit, joints))

    @property
    def joint_limits(self) -> Tuple[Conf, Conf]:
        joint_limits = self.robot_model.get_joint_limits()
        return to_cpu(joint_limits.position)

    @property
    def lower_limit(self) -> Conf:
        lower_limit, _ = self.joint_limits
        return lower_limit

    @property
    def upper_limit(self) -> Conf:
        _, upper_limit = self.joint_limits
        return upper_limit

    @property
    def velocity_limit(self) -> torch.Tensor:
        joint_limits = self.robot_model.get_joint_limits()
        _, upper_limit = joint_limits.velocity
        return torch.absolute(upper_limit)

    @property
    def joint_weight(self) -> torch.Tensor:
        return self.cspace_config.null_space_weight

    def get_joint_index(self, joint: str) -> int:
        [index] = self.get_joint_indices(joints=[joint])
        return index

    def get_joint_indices(self, joints: List[str]) -> List[int]:
        return list(map(self.active_joints.index, joints))

    def extract_conf(self, conf: Conf, joints: List[str]) -> Conf:
        assert conf.shape[-1] == self.dofs
        indices = self.get_joint_indices(joints)
        return conf[..., indices]

    def get_local_limits(self, joints: List[str], epsilon: float = 1e-3) -> JointLimits:
        indices = self.get_joint_indices(joints)
        conf = np.array(self.conf)
        epsilon_range = epsilon * np.ones(len(indices))
        joint_limits = self.kinematics_config.joint_limits
        joint_limits = joint_limits.clone()
        joint_limits.position[0][indices] = self.to_device(conf[indices] - epsilon_range)
        joint_limits.position[1][indices] = self.to_device(conf[indices] + epsilon_range)
        return joint_limits

    def set_joint_limits(self, joint_limits: JointLimits) -> None:
        self.sampler.low_bounds = joint_limits.position[0]
        self.sampler.range_b = joint_limits.position[1] - joint_limits.position[0]
        for arm_reacher in self.get_all_rollout_instances():
            arm_reacher.bound_cost.set_bounds(joint_limits)
            arm_reacher.bound_constraint.set_bounds(joint_limits)

    def clip_confs(self, confs: np.ndarray) -> np.ndarray:
        return np.clip(
            confs,
            np.tile(self.lower_limit, confs.shape[:-1] + (1,)),
            np.tile(self.upper_limit, confs.shape[:-1] + (1,)),
        )

    def sample_valid(self, num: int = 1, valid: bool = True, **kwargs: Any) -> torch.Tensor:
        return self.robot_world.sample(n=num, mask_valid=valid, **kwargs)

    def sample_confs(self, num: int = 1, valid: bool = False, **kwargs: Any) -> np.ndarray:
        return np.random.uniform(self.lower_limit, self.upper_limit)

    def sample_truncated(self, center: Optional[Conf] = None, std: float = INF) -> np.ndarray:
        if std == INF:
            return self.sample_confs()
        if center is None:
            center = self.conf
        return np.random.normal(center, std)

    def get_link_state(self, confs: Optional[List[Conf]] = None) -> CudaRobotModelState:
        if confs is None:
            confs = [self.conf]
        position = self.to_device(confs)
        assert position.shape[-1] == self.dofs, (position.shape[-1], self.dofs)
        return self.robot_model.get_state(position, link_name=None, calculate_jacobian=False)

    def get_ee_pose(self, **kwargs: Any) -> Pose:
        link_state = self.get_link_state(**kwargs)
        return link_state.ee_pose

    def get_link_poses(
        self, link_state: Optional[CudaRobotModelState] = None, **kwargs: Any
    ) -> Dict[str, Pose]:
        if link_state is None:
            link_state = self.get_link_state(**kwargs)
        return {
            link: Pose(
                position=link_state.links_position[..., i, :],
                quaternion=link_state.links_quaternion[..., i, :],
                name=link,
            )
            for i, link in enumerate(link_state.link_names)
        }

    def get_link_pose(self, link: str, **kwargs: Any) -> Pose:
        return self.get_link_poses(**kwargs)[link]

    def get_spheres(
        self, link_state: Optional[CudaRobotModelState] = None, **kwargs: Any
    ) -> torch.Tensor:
        if link_state is None:
            link_state = self.get_link_state(**kwargs)
        return link_state.link_spheres_tensor

    def _get_sphere_distances(
        self, spheres: torch.Tensor, collision_distance: float = 0.0, weight: float = 1.0
    ) -> torch.Tensor:
        if self.world_collision is None or (spheres.nelement() == 0):
            return torch.zeros(spheres.shape[:2])
        radii = spheres[..., 3]
        spheres = spheres.unsqueeze(1)
        query_buffer = CollisionQueryBuffer.initialize_from_shape(
            spheres.shape, self.tensor_args, self.world_collision.collision_types
        )
        activation_distance = self.tensor_args.to_device([collision_distance])
        weight = self.tensor_args.to_device([weight])
        distances = self.world_collision.get_sphere_distance(
            spheres,
            query_buffer,
            weight,
            activation_distance,
            sum_collisions=False,
            compute_esdf=True,
        ).squeeze(1)
        distances += radii
        return distances

    def get_sphere_distances(self, spheres: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.batch_size is None:
            distances = self._get_sphere_distances(spheres, **kwargs)
        else:
            distances = torch.cat(
                [
                    self._get_sphere_distances(spheres[indices, ...], **kwargs)
                    for indices in batched(range(spheres.shape[0]), self.batch_size)
                ]
            )
        return distances

    def get_object_distances(
        self, spheres: torch.Tensor, obj: str, poses: Optional[Pose] = None
    ) -> torch.Tensor:
        initial_pose = self.get_object_pose(obj)
        if poses is None:
            poses = initial_pose
        collision_from_world = multiply_poses(initial_pose, poses.inverse())
        spheres_collision = transform_spheres(collision_from_world, spheres)
        with self.active_context(objects=None):
            self.enable_objects_active(active_names=[obj])
            return self.get_sphere_distances(spheres_collision)

    def get_distances(
        self,
        collision_distance: float = 0.0,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        link_spheres = self.get_spheres(**kwargs)
        return self.get_sphere_distances(
            link_spheres, collision_distance=collision_distance, weight=weight
        )

    def get_collisions(self, **kwargs: Any) -> torch.Tensor:
        distances = self.get_distances(**kwargs)
        return distances > 0.0

    def get_colliding_links(self, **kwargs: Any) -> List[str]:
        collisions = self.get_collisions(**kwargs)
        collisions = torch.any(collisions, dim=0)
        indices = to_cpu(torch.nonzero(collisions)[:, 0])
        return remove_duplicates(map(self.get_sphere_link, indices))

    def _get_sphere_self_distances(self, spheres: torch.Tensor) -> torch.Tensor:
        return self.robot_world.get_self_collision_distance(spheres.unsqueeze(1)).squeeze(1)

    def get_sphere_self_distances(self, spheres: torch.Tensor) -> torch.Tensor:
        if self.batch_size is None:
            distances = self._get_sphere_self_distances(spheres)
        else:
            distances = torch.cat(
                [
                    self._get_sphere_self_distances(spheres[indices, ...])
                    for indices in batched(range(spheres.shape[0]), self.batch_size)
                ]
            )
        return distances

    def get_self_distances(self, **kwargs: Any) -> torch.Tensor:
        return self.get_sphere_self_distances(self.get_spheres(**kwargs))

    def get_self_collisions(self, **kwargs: Any) -> torch.Tensor:
        distances = self.get_self_distances(**kwargs)
        return distances > 0.0

    def get_limit_distances(self, joint_state: JointState) -> torch.Tensor:
        lower_limit, upper_limit = self.robot_model.get_joint_limits().position
        lower_difference = self.joint_state_difference(lower_limit, joint_state)
        upper_difference = -self.joint_state_difference(upper_limit, joint_state)
        distances = torch.maximum(lower_difference, upper_difference)
        return distances

    def attach_object(
        self,
        obj: str,
        link: str,
        grasp_pose: Optional[Pose] = None,
        fit_type: Optional[SphereFitType] = None,
        attach: bool = False,
        **kwargs: Any,
    ) -> Pose:
        link_pose = self.get_link_pose(link)
        if grasp_pose is not None:
            obj_pose = link_pose.multiply(grasp_pose)
        else:
            obj_pose = self.get_object_pose(obj)
        attachment_pose = link_pose.inverse().multiply(obj_pose)

        assert obj not in self.attached_objects, (obj, list(self.attached_objects))
        attachment = Attachment(self, obj, attachment_pose, parent=link)
        self.attached_objects[obj] = attachment
        if not attach:
            return attachment_pose

        if len(self.get_link_spheres(link)):
            ee_pose = self.compute_kinematics(self.joint_state).ee_pose
            self.attach_objects_to_robot(
                self.joint_state,
                object_names=[obj],
                link_name=link,
                sphere_fit_type=fit_type,
                world_objects_pose_offset=multiply_poses(
                    attachment_pose, ee_pose.inverse()
                ).inverse(),
                **kwargs,
            )
        self.set_object_active(obj, active=False)

        return attachment_pose

    def attachment_links(self) -> List[str]:
        return remove_duplicates(attachment.link for attachment in self.attached_objects.values())

    def detach_objects(
        self,
        links: Optional[List[str]] = None,
        detach: bool = False,
    ) -> Set[str]:
        if links is None:
            links = self.attachment_links
        link_poses = self.get_link_poses()
        attached_names = set()
        for name, attachment in list(self.attached_objects.items()):
            link = attachment.parent
            if link not in links:
                continue
            link_pose = link_poses[link]
            obj_pose = link_pose.multiply(attachment.pose)
            self.set_object_pose(name, obj_pose)
            self.attached_objects.pop(name)
            attached_names.add(name)
        if not detach:
            return attached_names
        for name in attached_names:
            self.set_object_active(name, active=True)
        for link in links:
            if len(self.get_link_spheres(link)):
                self.detach_object_from_robot(link_name=link)
        return attached_names

    @property
    def num_seeds(self) -> int:
        return self.ik_solver.num_seeds

    @property
    def num_problems(self) -> int:
        num_problems = self.ik_solver.solver.n_problems
        return num_problems

    @property
    def ik_batch(self) -> int:
        if self.num_problems == 1:
            return self.num_problems
        return self.num_problems // self.num_seeds

    @property
    def position_threshold(self) -> float:
        return self.ik_solver.position_threshold

    @property
    def rotation_threshold(self) -> float:
        return self.ik_solver.rotation_threshold

    def fill_poses(self, poses: Pose) -> Pose:
        if self.batch_size == 1:
            return poses
        return Pose.cat(list(fill_batch(poses, batch_size=self.ik_batch)))

    def inverse_kinematics(
        self,
        link_poses: Dict[str, Pose],
        seed: Optional[JointState] = None,
        debug: bool = False,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tuple[JointState, torch.Tensor]:
        start_time = current_time()
        assert link_poses
        links = list(link_poses)
        [batch] = set(map(len, link_poses.values()))

        ik_wrapper = self.ik_wrapper

        retract_config = self.retract_conf.repeat(batch, 1)
        _ee_pose = self.get_ee_pose(confs=retract_config)
        _link_poses = self.get_link_poses(confs=retract_config)
        for link, pose in link_poses.items():
            if link == ik_wrapper.ee_link:
                _ee_pose = pose
                _link_poses[link] = pose
            else:
                _link_poses[link] = pose
        _ee_pose = _ee_pose.clone()
        _link_poses = {
            link: pose.clone()
            for link, pose in _link_poses.items()
            if link in ik_wrapper.tool_links
        }
        if ik_wrapper.ik_batch != 1:
            _ee_pose = self.fill_poses(_ee_pose)
            _link_poses = {link: self.fill_poses(pose) for link, pose in _link_poses.items()}

        if seed is not None:
            num_repeats = int(math.ceil(ik_wrapper.ik_batch / seed.shape[0]))
            seed = seed.repeat(num_repeats, 1, 1)[: ik_wrapper.ik_batch, ...]
            seed = self.extract_conf(seed, ik_wrapper.active_joints)
        retract_config = self.extract_conf(retract_config, ik_wrapper.active_joints)

        result = ik_wrapper.solve_batch(
            _ee_pose,
            retract_config=retract_config[0],
            seed_config=seed,
            return_seeds=ik_wrapper.num_seeds,
            link_poses=_link_poses,
            **kwargs,
        )

        joint_state = self.to_joint_state(result.solution, ik_wrapper.active_joints)
        active_joints = set(flatten(map(self.yourdf.get_active_joints, links)))
        active_joints = list(filter(active_joints.__contains__, ik_wrapper.active_joints))
        joint_state = extract_joint_state(joint_state, active_joints)

        success = result.success
        joint_state = joint_state[:batch, ...]
        success = success[:batch, ...]
        if verbose:
            batch_solutions = to_cpu(success.sum(axis=-1))
            solutions = sum(batch_solutions)
            (self.logger.info if solutions else self.logger.warning)(
                f"{self.inverse_kinematics.__name__}) Shape: {to_cpu(joint_state.shape)} |"
                f" Solutions: {batch_solutions} | Links ({len(links)}): {links} | Position Error:"
                f" {torch.min(result.position_error):.3f} m | Rotation Error:"
                f" {torch.min(result.rotation_error):.3f} rad | Elapsed:"
                f" {elapsed_time(start_time):.3f} sec"
            )
        if debug and not success.count_nonzero():
            print(seed)
            for i in range(joint_state.shape[0]):
                for j in range(joint_state.shape[1]):
                    print(result.position_error[i, j])
                    print(result.rotation_error[i, j])
                    self.set_joint_state(joint_state[i, j])
                    self.show()
        return joint_state, success

    def iterative_inverse_kinematics(
        self, link_poses_list: Dict[str, List[Pose]], verbose: bool = True, **kwargs: Any
    ) -> Tuple[JointState, torch.Tensor]:
        start_time = current_time()
        assert link_poses_list
        links = list(link_poses_list)
        length = max(map(len, link_poses_list.values()))
        batch = max(map(len, flatten(link_poses_list.values())))
        joints = None
        joint_states = []
        all_success = None
        max_distance = None
        for i in range(length):
            link_poses = {link: link_poses_list[link][i] for link in links}
            assert link_poses
            seed = None
            if joint_states:
                seed = self.complete_joint_state(joint_states[-1]).position
            joint_state, success = self.inverse_kinematics(
                link_poses, seed=seed, verbose=True, **kwargs
            )
            if joint_states:
                difference = joint_state.position - joint_states[-1].position[-1]
                distance = torch.norm(difference, p=torch.inf, dim=-1)
            else:
                distance = torch.zeros(success.shape).to(self.device)
            if joints is None:
                joints = joint_state.joint_names
                all_success = success.clone()
                max_distance = distance.clone()
            assert joint_state.joint_names == joints
            joint_states.append(joint_state)
            all_success &= success
            max_distance = torch.max(max_distance, distance)
            if not all_success.any():
                break

        positions = torch.stack([joint_state.position for joint_state in joint_states])
        positions = torch.movedim(positions, 0, 2)
        max_distance = torch.where(
            all_success, max_distance, torch.full(max_distance.shape, torch.inf, device=self.device)
        )
        _, indices = torch.sort(max_distance, dim=-1)
        for i in range(len(indices)):
            positions[i] = positions[i, indices[i]]
            max_distance[i] = max_distance[i, indices[i]]
        joint_state = self.to_joint_state(positions, joints=joints)

        if verbose:
            batch_solutions = to_cpu(all_success.sum(axis=-1))
            solutions = sum(batch_solutions)
            (self.logger.info if solutions else self.logger.warning)(
                f"{self.iterative_inverse_kinematics.__name__}) Solutions: {batch_solutions} | Min"
                f" Distance: {math.degrees(torch.min(max_distance)):.3f} deg | Links: {links} |"
                f" Batch: {batch} | Seeds: {self.num_seeds} | Length: {length} | Dim:"
                f" {joint_state.shape[-1]} |  {elapsed_time(start_time):.3f} sec"
            )
        return joint_state, max_distance

    def link_inverse_kinematics(
        self, pose: Pose, link: Optional[str] = None, **kwargs: Any
    ) -> Tuple[JointState, torch.Tensor]:
        if link is None:
            link = self.ee_link
        link_poses = {link: pose}
        return self.inverse_kinematics(link_poses, **kwargs)

    def link_iterative_inverse_kinematics(
        self, poses_list: Pose, link: Optional[str] = None, **kwargs: Any
    ) -> Tuple[JointState, torch.Tensor]:
        if link is None:
            link = self.ee_link
        link_poses_list = {link: poses_list}
        return self.iterative_inverse_kinematics(link_poses_list, **kwargs)

    def complete_joint_state(self, joint_state: JointState) -> JointState:
        active_joints = list(joint_state.joint_names)
        inactive_joints = [joint for joint in self.active_joints if joint not in active_joints]
        if not inactive_joints:
            return joint_state.clone()
        augmented_state = extract_joint_state(self.joint_state, inactive_joints)
        return joint_state.get_augmented_joint_state(self.active_joints, augmented_state)

    def joint_state_difference(self, conf: torch.Tensor, joint_state: JointState) -> torch.Tensor:
        conf = self.extract_conf(conf, joint_state.joint_names)
        conf = conf.repeat(*joint_state.shape[:-1], 1)
        difference = conf - joint_state.position
        return difference

    def joint_state_distances(
        self, conf: torch.Tensor, joint_state: JointState, norm: int = 1
    ) -> torch.Tensor:
        joints = joint_state.joint_names
        difference = self.joint_state_difference(conf, joint_state)
        delta = torch.absolute(difference)
        weight = self.extract_conf(self.joint_weight, joints)
        distances = torch.norm(delta * weight, p=norm, dim=-1)
        return distances

    def check_state(self, joint_state: JointState) -> bool:
        joint_state = self.complete_joint_state(joint_state)
        valid, status = self.check_start_state(joint_state)
        return valid

    def plan_motion(
        self, goal_state: JointState, start_state: Optional[JointState] = None, **kwargs: Any
    ) -> Optional[JointState]:
        start_time = current_time()
        active_joints = list(goal_state.joint_names)
        if start_state is None:
            start_state = self.joint_state
        start_limit = -torch.max(self.get_limit_distances(start_state))
        start_state = self.complete_joint_state(start_state)
        goal_limit = -torch.max(self.get_limit_distances(goal_state))
        goal_state = self.complete_joint_state(goal_state)

        result = self.plan_single_js(
            start_state.clone(),
            goal_state.clone(),
            plan_config=MotionGenPlanConfig(**kwargs).clone(),
        )
        joint_state = result.get_interpolated_plan()
        joint_state = extract_joint_state(joint_state, active_joints)
        # NVIDIA
        from schedulestream.applications.custream.metrics import compute_joint_distance

        distance = compute_joint_distance(joint_state)
        success = result.success.item()
        status = result.status
        if status is not None:
            status = status.value
        attempts = (result.attempts + 1) * result.trajopt_attempts
        (self.logger.info if success else self.logger.warning)(
            f"{self.plan_motion.__name__}) Success: {success} | Length: {get_length(joint_state)} |"
            f" Duration: {get_length(joint_state)*self.time_step:.3f} sec | Distance:"
            f" {distance:.3f} | Valid: {result.valid_query} | Status: {status} | Start Limit:"
            f" {start_limit:.3f} | Goal Limit: {goal_limit:.3f} | Graph: {result.used_graph} |"
            f" Attempts: {attempts} | Joints ({len(active_joints)}): {active_joints} | Elapsed:"
            f" {elapsed_time(start_time):.3f} sec"
        )
        if not success:
            return None
        return joint_state

    def dump(self) -> None:
        for i, arm in enumerate(self.arms):
            self.logger.info(
                f"{i}/{len(self.arms)}) Arm: {arm} | Tool link: {self.get_arm_link(arm)} | Joints:"
                f" {self.get_arm_joints(arm)}"
            )
        self.logger.info(
            f"Arms: {self.arms} | Base link: {self.base_link} | EE link: {self.ee_link} | Tool"
            f" links: {self.tool_links} | Attached links: {self.attached_links} | Joints:"
            f" {self.active_joints} | Objects: {self.object_names}"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(active={self.active_joints}, movable={self.movable_names},"
            f" locked={self.locked_joints}, fixed={self.fixed_names})"
        )

    def show(self, **kwargs: Any) -> None:
        self.scene.show(**kwargs)

    def __hash__(self) -> int:
        return hash((self.__class__, id(self)))

    def __eq__(self, other: Any) -> bool:
        return self is other


@get_torch_jit_decorator()
def get_result(
    pose_error,
    position_error,
    rotation_error,
    goalset_index: Union[torch.Tensor, None],
    success,
    sol_position,
    col,
    batch_size: int,
    return_seeds: int,
    num_seeds: int,
    sort: bool = False,
):
    """JIT compatible function to get the best IK solutions."""
    error = pose_error.view(-1, num_seeds)
    error[~success] += 1000.0
    _, idx = torch.topk(error, k=return_seeds, largest=False, dim=-1)
    if sort:
        _, idx = torch.topk(error, k=return_seeds, largest=False, dim=-1)
    else:
        idx = torch.arange(return_seeds, dtype=torch.long, device=error.device)
    idx = idx + num_seeds * col.unsqueeze(-1)
    q_sol = sol_position[idx].view(batch_size, return_seeds, -1)

    success = success.view(-1)[idx].view(batch_size, return_seeds)
    position_error = position_error[idx].view(batch_size, return_seeds)
    rotation_error = rotation_error[idx].view(batch_size, return_seeds)
    total_error = position_error + rotation_error
    if goalset_index is not None:
        goalset_index = goalset_index[idx].view(batch_size, return_seeds)
    return q_sol, success, position_error, rotation_error, total_error, goalset_index


curobo.wrap.reacher.ik_solver.get_result = get_result
