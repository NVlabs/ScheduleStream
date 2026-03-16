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
from typing import Dict, List, Optional, Sequence

# Third Party
import gymnasium as gym
import isaaclab.utils.math as PoseUtils
import numpy as np
import torch
from curobo.types.math import Pose
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTerm

# NVIDIA
from schedulestream.applications.custream.command import Attach, Commands, Composite, Detach
from schedulestream.applications.custream.grasp import normalized_gripper_positions
from schedulestream.applications.custream.state import State
from schedulestream.applications.custream.world import World

OPEN_ACTION = +1
CLOSE_ACTION = -OPEN_ACTION


def get_robot_eef_pose(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    if env_ids is None:
        env_ids = slice(None)

    eef_pos = env.obs_buf["policy"]["eef_pos"][env_ids]
    eef_quat = env.obs_buf["policy"]["eef_quat"][env_ids]
    return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))


def target_eef_pose_to_action(
    env: ManagerBasedEnv,
    target_eef_pose: torch.Tensor,
    env_id: int = 0,
) -> torch.Tensor:
    target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

    curr_pose = get_robot_eef_pose(env, env_ids=[env_id])[0]
    curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

    delta_position = target_pos - curr_pos

    delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
    delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
    delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

    pose_action = torch.cat([delta_position, delta_rotation], dim=0)
    return pose_action


class PathController:
    def __init__(
        self,
        joints: str,
        joint_positions: torch.Tensor,
        link_poses: Dict[str, torch.Tensor],
        gripper_actions: torch.Tensor,
    ):
        self.joints = joints
        self.joint_positions = joint_positions
        self.link_poses = link_poses
        self.gripper_actions = gripper_actions
        assert self.steps == len(self.gripper_actions), (self.steps, len(self.gripper_actions))
        for link, poses in link_poses.items():
            assert self.steps == len(poses), (link, self.steps, len(poses))
        self.reset()

    @property
    def steps(self) -> int:
        return len(self.joint_positions)

    @property
    def completed(self) -> bool:
        return self.step >= self.steps

    def reset(self) -> None:
        self.step = 0

    def ik_action(self, env: gym.Env, term: ActionTerm, env_id: int) -> torch.Tensor:
        assert isinstance(term, DifferentialInverseKinematicsAction)
        assert term.cfg.clip is None
        base_env = env
        link = term.cfg.body_name
        assert link in self.link_poses, (link, list(self.link_poses))
        poses = self.link_poses[link]
        link_pose = Pose(position=poses[self.step, :3], quaternion=poses[self.step, 3:7])
        if term.cfg.body_offset is not None:
            offset_pose = Pose(
                position=torch.Tensor(term.cfg.body_offset.pos).to(base_env.device),
                quaternion=torch.Tensor(term.cfg.body_offset.rot).to(base_env.device),
            )
            link_pose = link_pose.multiply(offset_pose)

        link_matrix = link_pose.get_matrix().squeeze(0)
        pose_action = target_eef_pose_to_action(
            base_env,
            target_eef_pose=link_matrix,
            env_id=env_id,
        ).squeeze(0)
        if term.cfg.controller.use_relative_mode:
            pose_action /= term.cfg.scale
        return pose_action

    def next_action(self, env: gym.Env, obs: dict, env_id: int) -> Optional[torch.Tensor]:
        if self.completed:
            return None
        action_parts = []
        for term in get_action_terms(env):
            if isinstance(term, BinaryJointPositionAction):
                action_parts.append(self.gripper_actions[self.step].unsqueeze(0))
            elif isinstance(term, DifferentialInverseKinematicsAction):
                action_parts.append(self.ik_action(env, term, env_id))
            else:
                raise NotImplementedError(term)
        self.step += 1
        return torch.cat(action_parts, dim=-1)

    def __str__(self):
        return f"{self.__class__.__name__}(steps={self.steps})"

    __repr__ = __str__


def get_action_terms(env: ManagerBasedEnv) -> Optional[List[ActionTerm]]:
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return []
    return list(map(action_manager.get_term, action_manager.active_terms))


def discretize_gripper_action(world: World) -> float:
    normalized_gripper_conf = normalized_gripper_positions(world)
    if np.average(normalized_gripper_conf) <= 0.25:
        gripper_action = CLOSE_ACTION
    elif np.average(normalized_gripper_conf) >= 0.75:
        gripper_action = OPEN_ACTION
    else:
        gripper_action = 0.0
    return gripper_action


def get_action_links(env: ManagerBasedEnv) -> List[str]:
    links = []
    for term in get_action_terms(env):
        if isinstance(term, DifferentialInverseKinematicsAction):
            links.append(term.cfg.body_name)
    return links


def create_controller(
    planner: "Planner", state: State, commands: Optional[Commands]
) -> Optional[PathController]:
    env = planner.base_env
    world = state.world
    state.set()
    if commands is None:
        return None
    joints = world.all_joints
    joint_positions = []
    link_poses = {link: [] for link in get_action_links(env)}
    gripper_actions = []
    for command, step in commands.iterator(locked=False):
        if isinstance(command, Composite) and (len(command.commands) == 1):
            [command] = command.commands

        positions = world.to_device(world.get_joint_positions(joints))
        _link_poses = {}
        for link in link_poses:
            _link_poses[link] = world.get_node_pose(link)
        if gripper_actions:
            gripper_action = gripper_actions[-1]
        else:
            gripper_action = discretize_gripper_action(world)
        for _command in Composite.flatten([command]):
            if isinstance(_command, Attach):
                gripper_action = CLOSE_ACTION
            elif isinstance(_command, Detach):
                gripper_action = OPEN_ACTION

        joint_positions.append(positions)
        for link, pose in _link_poses.items():
            pose = planner.from_reference(pose)
            link_poses[link].append(torch.cat([pose.position, pose.quaternion], dim=-1).squeeze(0))
        gripper_actions.append(gripper_action)

    return PathController(
        joints=joints,
        joint_positions=torch.stack(joint_positions),
        link_poses={link: torch.stack(poses) for link, poses in link_poses.items()},
        gripper_actions=world.to_device(gripper_actions),
    )
