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
import copy
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

# Third Party
import numpy as np
import torch
from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelState
from curobo.cuda_robot_model.kinematics_parser import KinematicsParser
from curobo.cuda_robot_model.types import (
    CSpaceConfig,
    KinematicsTensorConfig,
    SelfCollisionKinematicsConfig,
)
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig

# NVIDIA
from schedulestream.applications.custream.utils import to_cpu
from schedulestream.common.utils import fill_batch


class CuRoboWrapper:
    def __init__(self, robot_config_dict: Optional[dict] = None):
        self.robot_config_dict = robot_config_dict

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
    def urdf_path(self) -> str:
        return self.robot_generator_config.urdf_path

    @property
    def usd_path(self) -> str:
        return self.robot_generator_config.usd_path

    @property
    def link_indices(self) -> Dict[str, int]:
        return self.kinematics_config.link_name_to_idx_map

    @property
    def links(self) -> List[str]:
        name_from_index = {index: name for name, index in self.link_indices.items()}
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
    def locked_joint_state(self) -> JointState:
        return self.kinematics_config.lock_jointstate

    def set_locked_joint_state(self, joint_state: JointState) -> None:
        new_positions = self.locked_joint_state.position.clone()
        joint_positions = dict(zip(joint_state.joint_names, joint_state.position))
        for index, joint in enumerate(self.locked_joints):
            if joint in joint_positions:
                new_positions[index] = joint_positions[joint]
        if torch.allclose(self.locked_joint_state.position, new_positions):
            return

        assert self.robot_config_dict is not None
        robot_config_dict = copy.deepcopy(self.robot_config_dict)
        locked_dict = dict(
            zip(self.locked_joint_state.joint_names, self.to_cpu(self.locked_joint_state.position))
        )
        input_dict = dict(zip(joint_state.joint_names, self.to_cpu(joint_state.position)))
        for joint, position in input_dict.items():
            if joint in locked_dict:
                locked_dict[joint] = position
        robot_config_dict["kinematics"]["lock_joints"] = locked_dict
        robot_cfg = RobotConfig.from_dict(robot_config_dict, self.tensor_args)
        kinematics_config = robot_cfg.kinematics.kinematics_config
        self.robot_model.update_kinematics_config(kinematics_config)
        assert kinematics_config.lock_jointstate.joint_names == self.locked_joint_state.joint_names
        self.locked_joint_state.position.copy_(kinematics_config.lock_jointstate.position)

    @property
    def retract_conf(self) -> torch.Tensor:
        return self.ik_solver.get_retract_config()

    def set_retract_conf(self, conf: torch.Tensor) -> None:
        self.retract_conf[:] = self.to_device(conf)

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

    def fill_poses(self, poses: Pose) -> Pose:
        return Pose.cat(list(fill_batch(poses, batch_size=self.ik_batch)))

    @property
    def num_spheres(self):
        return self.kinematics_config.total_spheres

    def get_link_spheres(self, link: str) -> torch.Tensor:
        try:
            return self.kinematics_config.get_link_spheres(link)
        except KeyError:
            return self.to_device(np.empty([0, 4]))

    def get_reference_link_spheres(self, link: str) -> torch.Tensor:
        try:
            return self.kinematics_config.get_reference_link_spheres(link)
        except KeyError:
            return self.to_device(np.empty([0, 4]))

    def set_link_active(self, active: bool, link: str) -> None:
        if active:
            self.kinematics_config.enable_link_spheres(link)
        else:
            self.kinematics_config.disable_link_spheres(link)

    def set_links_active(self, active: bool, links: Optional[List[str]] = None) -> None:
        if links is None:
            links = self.links
        for link in links:
            self.set_link_active(active, link)

    @contextmanager
    def link_spheres_saver(self, links: Optional[List[str]] = None):
        if links is None:
            links = self.links
        link_spheres = {}
        for link in links:
            if link not in link_spheres:
                spheres = self.get_link_spheres(link)
                if len(spheres) != 0:
                    link_spheres[link] = spheres.clone()
        yield
        for link, spheres in link_spheres.items():
            self.kinematics_config.update_link_spheres(link, spheres)

    def get_link_state(self, confs: Optional[torch.Tensor] = None) -> CudaRobotModelState:
        if confs is None:
            confs = self.retract_conf.clone().unsqueeze(0)
        return self.robot_model.get_state(
            self.to_device(confs), link_name=None, calculate_jacobian=False
        )

    def get_ee_pose(self, **kwargs: Any) -> Pose:
        link_state = self.get_link_state(**kwargs)
        return link_state.ee_pose

    def get_link_poses(self, **kwargs: Any) -> Dict[str, Pose]:
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
        link_poses = self.get_link_poses(**kwargs)
        return link_poses[link]

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(ee_link={self.ee_link}, tool_links={self.tool_links},"
            f" joints={self.active_joints})"
        )

    __repr__ = __str__
