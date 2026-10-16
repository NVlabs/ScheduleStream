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
"""Per-step env-action extraction for solved cuStream2 plans (IsaacLab TAMP agent).

:meth:`Planner.create_rollout_controller` rolls solved plan commands out in the cuStream2 world model
(open loop) and precomputes, for every step, the world joint configuration, the frame-corrected
IK link pose targets, and the binary gripper action. :class:`PathController` then emits one env
action per step, supporting JointPosition / RelativeJointPosition /
DifferentialInverseKinematics / BinaryJointPosition action terms.

Import only after the Isaac app is launched.
"""

from __future__ import annotations

# Standard Library
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence

# Third Party
import gymnasium as gym
import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, RelativeJointPositionAction
from isaaclab.envs.mdp.actions.rmpflow_task_space_actions import RMPFlowAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTerm

# NVIDIA
from schedulestream.applications.custream2.animate import flatten_commands, process_task_commands
from schedulestream.applications.custream2.command import (
    Attach,
    Detach,
    HandCommand,
    LinkPath,
    Trajectory,
)
from schedulestream.applications.custream2.state import Configuration
from schedulestream.applications.custream2.utils import to_cpu
from schedulestream.applications.isaaclab.utils import get_action_links, get_action_terms


def binary_gripper_actions(env: ManagerBasedEnv) -> tuple[float, float]:
    """The ``(open, close)`` action values for the env's binary gripper term.

    Derived by running candidates through the term's own ``process_actions`` and seeing which
    land on ``_close_command``, because the conventions are mutually exclusive and there is no
    pair that satisfies both. ``BinaryJointAction`` closes on ``actions < 0``; Arena's
    ``BinaryJointPositionZeroToOneAction`` closes on ``actions > 0.5``; a third variant in the
    same module thresholds on ``cfg.threshold``/``cfg.positive_threshold``. Hard-coding one
    pair silently *inverts* the gripper under another, which reads as a grasp that opens on
    contact rather than as an error.

    Probing keeps this file free of any Arena import, and reproduces both hand-written pairs it
    replaces: ``(1.0, -1.0)`` for the base term, ``(0.0, 1.0)`` for the ZeroToOne subclass.
    """
    term = next(
        (t for t in get_action_terms(env) if isinstance(t, BinaryJointPositionAction)), None
    )
    if term is None:
        raise LookupError("no BinaryJointPositionAction in the env's action terms")

    def closes(value: float) -> bool:
        term.process_actions(torch.full_like(term._raw_actions, value))
        return bool(torch.allclose(term._processed_actions[0], term._close_command))

    raw = term._raw_actions.clone()
    try:
        verdicts = {value: closes(value) for value in (1.0, 0.0, -1.0)}
    finally:
        term._raw_actions[:] = raw

    opened = next((v for v, shuts in verdicts.items() if not shuts), None)
    closed = next((v for v, shuts in verdicts.items() if shuts), None)
    if opened is None or closed is None:
        raise ValueError(f"{type(term).__name__}: probes {verdicts} map to only one of open/close")
    return opened, closed


def rollout_path_controller(
    state: Any,
    commands: Optional[Sequence[Any]],
    gripper: tuple[float, float],
    joint_names: Optional[List[str]] = None,
    links: Sequence[str] = (),
    link_pose: Optional[Callable[[str], torch.Tensor]] = None,
    joint_space: bool = True,
    **kwargs: Any,
) -> Optional["PathController"]:
    """Roll ``commands`` out in the world model and precompute one target per step.

    The cuStream2 execution model: each leaf command's ``execute()`` advances the world a
    single step, so the configuration after each is that step's target. The gripper is read off
    the world's arm attachments (closed while holding) rather than the command stream, so it
    stays consistent with what the plan actually did.

    ``joint_names`` labels the PathController's columns, which are matched against the action
    term's ``_joint_names``; it defaults to the world's own spelling. ``link_pose`` supplies the
    frame-corrected target for each entry of ``links`` (IK envs); joint-space envs pass neither.
    """
    if commands is None:
        return None
    state.set()
    world = state.world
    # NVIDIA
    from schedulestream.applications.custream2.command import Commands

    open_action, close_action = gripper
    [arm] = world.arms
    joints = world.all_joints
    joint_positions: List[torch.Tensor] = []
    link_poses: Dict[str, List[torch.Tensor]] = {link: [] for link in links}
    gripper_actions: List[float] = []
    for command in Commands.flatten(commands):
        for _ in command.execute():
            if joint_space:
                joint_positions.append(world.to_device(world.get_joint_positions(joints)))
            for link in links:
                link_poses[link].append(link_pose(link))
            gripper_actions.append(close_action if world.attached_arms[arm] else open_action)
    if not gripper_actions:
        return None
    return PathController(
        joints=joint_names if joint_names is not None else joints,
        joint_positions=torch.stack(joint_positions) if joint_space else None,
        link_poses={link: torch.stack(poses) for link, poses in link_poses.items()},
        gripper_actions=world.to_device(gripper_actions),
        **kwargs,
    )


def task_path_controller(
    state: Any,
    base_env: ManagerBasedEnv,
    env_id: int,
    raw_control_pose: Callable[[str, Optional[torch.Tensor]], torch.Tensor],
    commands: Optional[Sequence[Any]],
) -> Optional["PathController"]:
    """Precompute the controller from task-space commands alone (no world rollout).

    Flattens the plan (simultaneous commands are unsupported) and converts it with
    :func:`process_task_commands` so every command is task-space, then reads link pose targets
    per command type instead of stepping the world with ``execute()``: LinkPaths supply
    per-waypoint poses; hand/attach/detach commands hold the last pose for their single step.
    The gripper is inferred from Attach/Detach, and no joint-space targets are produced.

    ``raw_control_pose`` is the planner's own ``raw_control_pose`` (e.g.
    ``Planner.raw_control_pose``): it decides which frame the control pose is lifted into, and
    must match the frame the ``eef_pos``/``eef_quat`` observation reports in.
    """
    if commands is None:
        return None
    state.set()
    world = state.world

    links = get_action_links(base_env)
    [arm] = world.arms
    open_action, close_action = binary_gripper_actions(base_env)
    tool_link = world.get_tool_link(arm)
    tool_matrix = raw_control_pose(tool_link, link_pose=world.get_tool_pose(arm))
    eef = eef_frame_corrections(base_env, env_id, raw_control_pose, links)
    corrections = {
        link: torch.linalg.inv(tool_matrix) @ raw_control_pose(link) @ eef[link] for link in links
    }
    last_matrix = {link: tool_matrix @ corrections[link] for link in links}
    holding = bool(world.attached_arms[arm])

    link_poses: Dict[str, List[torch.Tensor]] = {link: [] for link in links}
    gripper_actions = []
    for command in process_task_commands(flatten_commands(commands), task_space=True):
        path: Optional[LinkPath] = None
        steps = 1
        if isinstance(command, Attach):
            holding = True
        elif isinstance(command, Detach):
            holding = False
        elif isinstance(command, HandCommand):
            pass
        elif isinstance(command, LinkPath):
            assert command.link == tool_link, (command.link, tool_link)
            path = command
            steps = command.length
        else:
            raise NotImplementedError(command)

        for index in range(steps):
            for link in links:
                if path is not None:
                    last_matrix[link] = (
                        raw_control_pose(tool_link, link_pose=path.pose(index)) @ corrections[link]
                    )
                link_poses[link].append(last_matrix[link])
            gripper_actions.append(close_action if holding else open_action)

    return PathController(
        joints=world.all_joints,
        joint_positions=None,
        link_poses={link: torch.stack(poses) for link, poses in link_poses.items()},
        gripper_actions=world.to_device(gripper_actions),
    )


def joint_path_controller(
    state: Any,
    base_env: ManagerBasedEnv,
    commands: Optional[Sequence[Any]],
) -> Optional["PathController"]:
    """Precompute the controller from joint-space commands alone (no world rollout).

    Flattens the plan (simultaneous commands are unsupported) and reads joint targets per
    command type instead of stepping the world with ``execute()``: Trajectories supply
    per-waypoint positions over their joints; Configurations and hand commands one step each;
    joints a command doesn't cover hold their last value. The gripper is inferred from
    Attach/Detach, and no link pose targets are produced.
    """
    if commands is None:
        return None
    state.set()
    world = state.world

    joints = world.all_joints
    positions = dict(zip(joints, to_cpu(world.get_joint_positions(joints))))
    [arm] = world.arms
    open_action, close_action = binary_gripper_actions(base_env)
    holding = bool(world.attached_arms[arm])

    joint_positions: List[List[float]] = []
    gripper_actions = []

    for command in flatten_commands(commands):
        if isinstance(command, Attach):
            holding = True
            steps = [{}]
        elif isinstance(command, Detach):
            holding = False
            steps = [{}]
        elif isinstance(command, HandCommand):
            steps = [dict(zip(command.finger_joints, to_cpu(command.conf)))]
        elif isinstance(command, (Configuration, Trajectory)):
            joint_state = command.joint_state
            steps = [
                dict(zip(joint_state.joint_names, waypoint))
                for waypoint in to_cpu(joint_state.position)
            ]
        else:
            raise NotImplementedError(command)

        for step in steps:
            positions.update(step)
            joint_positions.append([positions[joint] for joint in joints])
            gripper_actions.append(close_action if holding else open_action)

    return PathController(
        joints=joints,
        joint_positions=world.to_device(joint_positions),
        link_poses={},
        gripper_actions=world.to_device(gripper_actions),
    )


def get_robot_eef_pose(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    """Read the EEF pose observation (``eef_pos``/``eef_quat``, quat w,x,y,z) as 4x4 matrices."""
    if env_ids is None:
        env_ids = slice(None)
    eef_pos = env.obs_buf["policy"]["eef_pos"][env_ids]
    eef_quat = env.obs_buf["policy"]["eef_quat"][env_ids]
    return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))


def unscale_action(term: ActionTerm, env_id: int, target: torch.Tensor) -> torch.Tensor:
    """Invert a joint term's scale/offset, so commanding the result reproduces ``target``.

    Both are per-env tensors when the cfg specifies them per joint, and plain scalars
    otherwise, which is the only reason this needs a helper rather than a subtraction.
    """
    offset = term._offset[env_id] if isinstance(term._offset, torch.Tensor) else term._offset
    scale = term._scale[env_id] if isinstance(term._scale, torch.Tensor) else term._scale
    return (target - offset) / scale


def noise_actions(
    env: ManagerBasedEnv, actions: torch.Tensor, std: Optional[float]
) -> torch.Tensor:
    """Add Gaussian noise of ``std`` to the arm action terms, in place.

    A ``std`` of None is a no-op, so callers need not guard.

    The binary gripper is left alone: it thresholds rather than scales, so any perturbation
    large enough to matter flips open/close outright and desynchronises the grasp from the
    plan. IK terms are clamped back into the unit cube afterwards, which is the range only
    their deltas are normalized to -- joint targets are not.

    Terms are selected by exclusion rather than an allow-list, so a term type nobody
    anticipated gets noised rather than silently skipped. ``BinaryJointAction`` and
    ``JointAction`` are unrelated ActionTerm subclasses, so this matches the allow-list for
    every term that exists today.

    The pre-noise actions are stashed on ``env.unnoised_actions`` for UnnoisedActionsRecorder,
    which records what the planner intended rather than what was executed.
    """
    if std is None:
        return actions
    env.unnoised_actions = actions.clone()
    start = 0
    for term in get_action_terms(env):
        end = start + term.action_dim
        if not isinstance(term, BinaryJointPositionAction):
            actions[:, start:end] += std * torch.randn_like(actions[:, start:end])
            if isinstance(term, DifferentialInverseKinematicsAction):
                actions[:, start:end] = torch.clamp(actions[:, start:end], -1.0, 1.0)
        start = end
    return actions


def hold_action(env: ManagerBasedEnv, env_id: int) -> torch.Tensor:
    """The action that keeps ``env_id`` where it is, with the gripper open.

    Stationary is not the same value for every term, which is why this cannot just be zeros:
    relative joint and task-space terms are deltas, so zero holds, but an absolute
    JointPositionAction reads zero as "go to the zero configuration" and would fling the arm.
    Absolute terms therefore command the measured joint positions, put back through the term's
    own scale/offset so the round trip lands on the same targets.

    Capture this once per reset and re-emit it, rather than recomputing per step: re-reading
    the measured positions each step would track the arm's settling drift instead of holding
    the pose it started from.
    """
    open_action, _ = binary_gripper_actions(env)
    parts = []
    for term in get_action_terms(env):
        if isinstance(term, BinaryJointPositionAction):
            parts.append(torch.full((term.action_dim,), open_action, device=env.device))
        elif isinstance(term, JointPositionAction):
            current = term._asset.data.joint_pos[env_id, term._joint_ids]
            parts.append(unscale_action(term, env_id, current))
        else:
            parts.append(torch.zeros(term.action_dim, device=env.device))
    return torch.cat(parts, dim=-1)


def eef_frame_corrections(
    env: ManagerBasedEnv,
    env_id: int,
    control_pose: Callable[[str], torch.Tensor],
    links: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Per link, the body-frame transform mapping cuRobo's control frame onto the eef observation.

    cuRobo's URDF and the env's USD need not agree on a control link's frame (this IsaacLab's
    Franka defines ``panda_hand`` ~180 deg about z from cuStream2's), and the IK action's body
    offset shifts it again. ``inv(control(q)) @ obs(q)`` measures whatever that fixed difference
    is, as a body-frame (right-multiplied) transform, so it stays valid at every configuration
    rather than only the one it was measured at.

    ``control_pose`` is the planner's own ``raw_control_pose``: it decides which frame the
    control pose is lifted into, which has to be the frame the eef observation reports in.
    """
    if not links:
        return {}
    obs_eef = get_robot_eef_pose(env, env_ids=[env_id])[0].to(
        device=env.device, dtype=torch.float32
    )
    return {link: torch.linalg.inv(control_pose(link)) @ obs_eef for link in links}


def _orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
    """Project ``matrix`` onto the nearest proper rotation matrix (SVD orthogonalization).

    ``quat_from_matrix`` NaNs out any input whose recovered quaternion norm is off by more than
    2e-5, which ordinary float32 pose-composition chains routinely exceed; this restores a tight
    orthonormal matrix so that check passes.
    """
    u, _, vh = torch.linalg.svd(matrix)
    r = u @ vh
    det = torch.linalg.det(r)
    u[..., -1] = u[..., -1] * det.unsqueeze(-1)
    return u @ vh


def target_eef_pose_to_action(
    env: ManagerBasedEnv,
    target_eef_pose: torch.Tensor,
    env_id: int = 0,
) -> torch.Tensor:
    """Convert a world-frame target EEF pose into a delta-pose action against the current obs."""
    target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

    curr_pose = get_robot_eef_pose(env, env_ids=[env_id])[0]
    curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

    delta_position = target_pos - curr_pos

    delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
    delta_quat = PoseUtils.quat_from_matrix(_orthonormalize(delta_rot_mat))
    delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

    return torch.cat([delta_position, delta_rotation], dim=0)


class Controller(ABC):
    """Common interface for objects that emit one precomputed env action per step."""

    def __init__(self) -> None:
        self.reset()

    @property
    @abstractmethod
    def steps(self) -> int:
        """Total number of precomputed steps."""

    @property
    def completed(self) -> bool:
        return self.step >= self.steps

    def reset(self) -> None:
        self.step = 0

    @abstractmethod
    def next_action(self, env: gym.Env, env_id: int) -> Optional[torch.Tensor]:
        """Return this step's env action and advance, or None once :attr:`completed`."""

    def __str__(self):
        return f"{self.__class__.__name__}(steps={self.steps})"

    __repr__ = __str__


class ActionController(Controller):
    """Emit one precomputed env action per step, stored directly rather than derived per term."""

    def __init__(self, actions: List[torch.Tensor]):
        self.actions = actions
        super().__init__()

    @property
    def steps(self) -> int:
        return len(self.actions)

    def next_action(self, env: gym.Env, env_id: int) -> Optional[torch.Tensor]:
        if self.completed:
            return None
        action = self.actions[self.step]
        self.step += 1
        return action


class PathController(Controller):
    """Emit one env action per precomputed plan step."""

    def __init__(
        self,
        joints: List[str],
        joint_positions: Optional[torch.Tensor],
        link_poses: Dict[str, torch.Tensor],
        gripper_actions: torch.Tensor,
        task_scale: float = 1.0,
        joint_scale: float = 1.0,
    ):
        """
        Args:
            joints: Joint names indexing ``joint_positions`` columns.
            joint_positions: ``[T, dof]`` world joint configuration per step, or None when the
                env has no joint-space action terms.
            link_poses: Per IK link, ``[T, 4, 4]`` world-frame control-frame targets. The IK
                action's body offset AND the cuRobo->env frame correction are already baked in,
                so each pose diffs directly against the EEF pose observation.
            gripper_actions: ``[T]`` binary gripper action per step.
            task_scale: Extra multiplier applied to relative-mode IK/RMPFlow pose actions.
            joint_scale: Extra multiplier applied to relative joint-position actions.
        """
        self.joints = joints
        self.joint_positions = joint_positions
        self.link_poses = link_poses
        self.gripper_actions = gripper_actions
        self.task_scale = task_scale
        self.joint_scale = joint_scale
        if joint_positions is not None:
            assert self.steps == len(joint_positions), (self.steps, len(joint_positions))
        for link, poses in link_poses.items():
            assert self.steps == len(poses), (link, self.steps, len(poses))
        super().__init__()

    @property
    def steps(self) -> int:
        return len(self.gripper_actions)

    def joint_targets(self, term: ActionTerm) -> torch.Tensor:
        """This step's precomputed joint targets, in the term's own joint order."""
        assert self.joint_positions is not None
        indices = torch.tensor(
            [self.joints.index(name) for name in term._joint_names], dtype=torch.long
        )
        return self.joint_positions[self.step][indices]

    def joint_position_action(self, env: gym.Env, term: ActionTerm, env_id: int) -> torch.Tensor:
        assert isinstance(term, JointPositionAction)
        return unscale_action(term, env_id, self.joint_targets(term))

    def relative_joint_position_action(
        self, env: gym.Env, term: ActionTerm, env_id: int
    ) -> torch.Tensor:
        assert isinstance(term, RelativeJointPositionAction)
        current_pos = term._asset.data.joint_pos[env_id, term._joint_ids]
        return self.joint_scale * unscale_action(
            term, env_id, self.joint_targets(term) - current_pos
        )

    def ik_action(self, env: gym.Env, term: ActionTerm, env_id: int) -> torch.Tensor:
        assert isinstance(term, DifferentialInverseKinematicsAction)
        assert term.cfg.clip is None
        link = term.cfg.body_name
        assert link in self.link_poses, (link, list(self.link_poses))
        assert term.cfg.controller.use_relative_mode
        target_matrix = self.link_poses[link][self.step]
        pose_action = target_eef_pose_to_action(env, target_eef_pose=target_matrix, env_id=env_id)
        pose_action /= term.cfg.scale
        return self.task_scale * pose_action

    def rmpflow_action(self, env: gym.Env, term: ActionTerm, env_id: int) -> torch.Tensor:
        assert isinstance(term, RMPFlowAction)
        assert term.cfg.clip is None
        link = term.cfg.body_name
        assert link in self.link_poses, (link, list(self.link_poses))
        target_matrix = self.link_poses[link][self.step]
        _CORR_QUAT = [0.0, 0.0, 1.0, 0.0]
        tpos, trot = PoseUtils.unmake_pose(target_matrix)
        corr = PoseUtils.matrix_from_quat(
            torch.tensor(_CORR_QUAT, device=trot.device, dtype=trot.dtype)
        )
        target_matrix = PoseUtils.make_pose(tpos, trot @ corr)
        if term.cfg.use_relative_mode:
            pose_action = target_eef_pose_to_action(
                env, target_eef_pose=target_matrix, env_id=env_id
            )
            return self.task_scale * pose_action / term.cfg.scale
        target_pos, target_rot = PoseUtils.unmake_pose(target_matrix)
        target_quat = PoseUtils.quat_from_matrix(target_rot)
        return torch.cat([target_pos, target_quat], dim=-1) / term.cfg.scale

    def next_action(self, env: gym.Env, env_id: int) -> Optional[torch.Tensor]:
        if self.completed:
            return None
        action_parts = []
        for term in get_action_terms(env):
            if isinstance(term, BinaryJointPositionAction):
                action_parts.append(self.gripper_actions[self.step].unsqueeze(0))
            elif isinstance(term, JointPositionAction):
                action_parts.append(self.joint_position_action(env, term, env_id))
            elif isinstance(term, RelativeJointPositionAction):
                action_parts.append(self.relative_joint_position_action(env, term, env_id))
            elif isinstance(term, DifferentialInverseKinematicsAction):
                action_parts.append(self.ik_action(env, term, env_id))
            elif isinstance(term, RMPFlowAction):
                action_parts.append(self.rmpflow_action(env, term, env_id))
            else:
                raise NotImplementedError(term)
        self.step += 1
        return torch.cat(action_parts, dim=-1)
