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
"""cuStream2 TAMP planner for IsaacLab envs (tamp_agent.py).

Extends the env-agnostic :class:`custream2.policy.Planner` with IsaacLab observation hooks:
builds a cuStream2 :class:`World` from the live USD scene, derives the symbolic goal from the
env's success termination, solves TAMP, and precomputes a :class:`controller.Controller`
(:class:`~controller.PathController` or, for multi-robot envs, :class:`~controller.ActionController`)
for per-step env-action extraction (open-loop replay).

Ported from cuStream (cuRoboV1) to cuStream2 (cuRoboV2) and to IsaacLab 3.0 (Isaac Sim 6.0).
The version-sensitive pieces:

* ``scene.state`` root poses are warp xyzw quaternions (converted to wxyz for cuRobo);
* the World must be warmed up after construction (captures cuRobo's CUDA graphs) under
  ``autograd_enabled`` (cuRobo IK/trajopt call backward());
* this IsaacLab's Franka USD defines ``panda_hand`` ~180 deg about z from cuStream2's Franka URDF,
  so IK targets need a constant body-frame correction calibrated against the EEF observation
  (:func:`controller.eef_frame_corrections`);
* USD obstacle extraction fan-triangulates n-gon meshes (usd_utils patch) and skips
  degenerate ones; only env-registered, non-kinematic assets are movable.

Import only after the Isaac app is launched.
"""

from __future__ import annotations

# Standard Library
from typing import Any, List, Optional

# Third Party
import torch
from curobo.types import JointState, Pose
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs.mdp import BinaryJointPositionAction
from isaaclab.managers import TerminationTermCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import convert_quat
from isaaclab_tasks.manager_based.manipulation.stack.config.droid.stack_joint_pos_env_cfg import (
    cubes_stacked as droid_bimanual_cubes_stacked,
)
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.blocked_ik_rel_visuomotor_env_cfg import (
    red_block_on_platform as franka_red_block_on_platform,
)
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.either_ik_rel_visuomotor_env_cfg import (
    either_cube_packed as franka_either_cube_packed,
)
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.pack_ik_rel_visuomotor_env_cfg import (
    cubes_packed as franka_cubes_packed,
)
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_env_cfg import (
    cubes_stacked_four as franka_cubes_stacked_four,
)
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_abs_visuomotor_dual_env_cfg import (
    cubes_stacked_dual_arm as franka_cubes_stacked_dual_arm,
)
from isaaclab_tasks.manager_based.manipulation.stack.mdp import cubes_stacked

# NVIDIA
from schedulestream.applications.custream2.command import Command, Commands
from schedulestream.applications.custream2.policy import Planner as BasePlanner
from schedulestream.applications.custream2.tamp import At, Attached, Holding, reset_globals
from schedulestream.applications.custream2.utils import autograd_enabled, multiply_poses
from schedulestream.applications.custream.utils import unit_pose
from schedulestream.applications.isaaclab.controller import (
    ActionController,
    Controller,
    PathController,
    binary_gripper_actions,
    eef_frame_corrections,
    joint_path_controller,
    rollout_path_controller,
    task_path_controller,
    unscale_action,
)
from schedulestream.applications.isaaclab.isaac_utils import (
    get_action_terms,
    get_robots_in_action_order,
)
from schedulestream.applications.isaaclab.robot import ARM_PREFIX
from schedulestream.applications.isaaclab.utils import (
    create_world,
    get_action_joints,
    get_action_links,
    resolve_ik_offsets,
)
from schedulestream.common.utils import apply_mapping
from schedulestream.language.connective import Conjunction, Disjunction


class Planner(BasePlanner):
    """cuStream2 Planner observing an IsaacLab env; one instance serves all env copies
    (``solve(env_id)`` selects which copy the observation hooks read)."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        success_term: TerminationTermCfg,
        batch_size: int = 128,
        collisions: bool = True,
        max_time: float = 60.0,
        hold: Optional[int] = None,
        grasp: Optional[str] = None,
        speed: float = 1.0,
        use_cuda_graph: bool = True,
        animate: bool = False,
        record: bool = False,
        verbose: bool = False,
        **solve_kwargs: Any,
    ) -> None:
        self.env = env
        self.env_id = 0
        self.success_term = success_term
        self.hold_steps = hold
        self.verbose = verbose
        world = create_world(
            self.base_env,
            batch_size=batch_size,
            grasp=grasp,
            verbose=verbose,
            speed=speed,
            use_cuda_graph=use_cuda_graph,
        )
        super().__init__(
            world,
            goal=None,
            max_time=max_time,
            animate=animate,
            record=record,
            collisions=collisions,
            **solve_kwargs,
        )
        self._ik_offsets = resolve_ik_offsets(self.base_env, self.world.device)
        self.update_state()
        motion = self.solve_kwargs.get("motion", "joint")
        world.warmup(motion_planner=(motion.split(":", 1)[0] == "curobo"))

    @property
    def base_env(self) -> ManagerBasedEnv:
        if isinstance(self.env, ManagerBasedEnv):
            return self.env
        return self.env.unwrapped

    @property
    def cfg(self) -> ManagerBasedEnvCfg:
        return self.base_env.cfg

    @property
    def scene(self) -> InteractiveScene:
        return self.base_env.scene

    @property
    def sim(self) -> SimulationContext:
        return self.base_env.sim

    @property
    def robots(self) -> List[str]:
        """Return the env's articulation names, in the env action vector's own order.

        See :func:`get_robots_in_action_order`: env.scene.articulations dict order
        need not match how the action vector concatenates each robot's terms.
        """
        return get_robots_in_action_order(self.base_env)

    @property
    def robot(self) -> str:
        """Return the env's single articulation name."""
        [robot] = self.robots
        return robot

    @property
    def articulation(self) -> Articulation:
        return self.scene.articulations[self.robot]

    def _pose(self, name: str) -> Pose:
        """Env-world pose of a scene body. ``scene.state`` quaternions are warp xyzw."""
        for bodies in self.scene.state.values():
            if name not in bodies:
                continue
            body = bodies[name]
            if "root_pose" not in body:
                continue
            root_pose = self.world.to_device(body["root_pose"][self.env_id : self.env_id + 1])
            root_pose[..., 3:7] = convert_quat(root_pose[..., 3:7], to="wxyz")
            return Pose(position=root_pose[:, :3], quaternion=root_pose[:, 3:])
        raise KeyError(name)

    def get_env_robot_pose(self) -> Pose:
        """Return the robot's world root pose.

        For multiple robots (a composite multi-robot world), the shared root the arms were each
        offset from is this env copy's origin, not any one arm's own pose. It must be the env
        origin rather than the identity: :func:`isaaclab.robot.build_multi_robot_config` places the
        arms env-locally, while :meth:`_pose` reports objects in world frame, so returning identity
        leaves objects offset by ``env_origins[env_id]`` and puts every grasp out of reach.
        """
        if len(self.robots) == 1:
            return self._pose(self.robot)
        origin = self.world.to_device(self.base_env.scene.env_origins[self.env_id]).reshape(1, 3)
        return Pose(position=origin, quaternion=unit_pose().quaternion.to(origin.device))

    def get_env_object_pose(self, obj: str) -> Pose:
        return self._pose(obj)

    def get_env_joint_state(self) -> JointState:
        """Return the robot's current joint state in world joint order.

        Multi-robot: each articulation's joints are read under the same ``ARM_PREFIX{i}_``
        naming :func:`isaaclab.robot.build_multi_robot_config` gave the combined world.
        """
        if len(self.robots) == 1:
            positions = self.scene.state["articulation"][self.robot]["joint_position"][self.env_id]
            return JointState.from_position(
                self.world.to_device(positions).unsqueeze(0),
                joint_names=list(self.articulation.joint_names),
            )
        joint_positions = {}
        for i, robot in enumerate(self.robots):
            articulation = self.scene.articulations[robot]
            robot_joints = [f"{ARM_PREFIX}{i}_{joint}" for joint in articulation.joint_names]
            robot_positions = self.scene.state["articulation"][robot]["joint_position"][self.env_id]
            joint_positions.update(zip(robot_joints, robot_positions))
        positions = torch.tensor(
            apply_mapping(joint_positions, self.world.all_joints),
            dtype=torch.float32,
            device=self.world.device,
        )
        return JointState.from_position(positions.unsqueeze(0), joint_names=self.world.all_joints)

    def create_goal(self, success_term: Optional[TerminationTermCfg]) -> Any:
        """Derive the symbolic goal from the env success term.

        With no success term, fall back on the movable-object count: one movable -> hold it;
        several -> stack the two highest-indexed.

        When ``reset``, every arm gets an additional ``At(arm) <= conf`` goal pinning it to its
        current configuration -- a soft preference (not a hard constraint), so arms not otherwise
        driven by the goal settle back rather than drift.
        """
        if success_term is None:
            self.world.dump()
            movable = self.world.movable_names
            if len(movable) == 1:
                [arm] = self.world.arms
                return Holding(arm) <= movable[0]
            obj1, obj2 = sorted(movable, reverse=True)[:2]
            return Attached(obj1) == obj2

        reset = False
        platform_cubes = None
        if success_term.func == cubes_stacked:
            cubes = [f"cube_{i}" for i in range(1, 3 + 1)]
            for i, cube in enumerate(cubes):
                cube_cfg = f"{cube}_cfg"
                if cube_cfg not in success_term.params:
                    continue
                if success_term.params[cube_cfg] is None:
                    cubes[i] = None
                else:
                    cubes[i] = success_term.params[cube_cfg].name
        elif success_term.func == franka_cubes_stacked_four:
            cubes = ["cube_1", "cube_2", "cube_3", "cube_4"]
        elif success_term.func == droid_bimanual_cubes_stacked:
            cubes = ["cube_0", "cube_1", "cube_2"]
            cubes = cubes[:2]
            reset = True
        elif success_term.func == franka_cubes_stacked_dual_arm:
            cubes = ["cube_1", "cube_2", "cube_3"]
            reset = True
        elif success_term.func == franka_cubes_packed:
            platform_cubes = ["cube_1", "cube_2"]
        elif success_term.func == franka_red_block_on_platform:
            platform_cubes = ["cube_2"]
        elif success_term.func == franka_either_cube_packed:
            return Disjunction(
                Attached("cube_1") == "platform",
                Attached("cube_2") == "platform",
            )
        else:
            raise NotImplementedError(success_term.func)

        if platform_cubes is not None:
            goal = None
            for cube in platform_cubes:
                term = Attached(cube) == "platform"
                goal = term if goal is None else goal & term
        else:
            goal = None
            for lower, upper in zip(cubes, cubes[1:]):
                if lower is None or upper is None:
                    break
                term = Attached(upper) == lower
                goal = term if goal is None else goal & term
        goals = [goal]
        if reset:
            for arm in self.world.arms:
                conf = self.state.arm_configuration(arm)
                goals.append(At(arm) <= conf)
        return Conjunction(*goals)

    @property
    def eef_obs_in_base_frame(self) -> bool:
        """True when the env reports ``eef_pos``/``eef_quat`` in the robot base frame.

        Galbot uses ``ee_frame_pose_in_base_frame`` (root-relative); Franka uses world-frame
        ``ee_frame_pos``. This selects the frame :meth:`raw_control_pose` builds targets in.
        """
        term = getattr(self.base_env.cfg.observations.policy, "eef_pos", None)
        return getattr(getattr(term, "func", None), "__name__", "") == "ee_frame_pose_in_base_frame"

    def raw_control_pose(self, link: str, link_pose: Optional[Pose] = None) -> torch.Tensor:
        """IK control-frame pose as cuRobo sees it, in the same frame as the ``eef_pos`` observation.

        Built so :func:`eef_frame_corrections` (a constant body-frame transform) cancels cleanly at
        any config. When the env's eef observation is in the robot base frame
        (:attr:`eef_obs_in_base_frame`), cuRobo's model is already in that frame, so return
        ``link_pose * offset`` directly; a world-frame lift (``root * .. - env_origin``) would leave a
        ``root`` rotation on the left that only cancels at the calibration config and drives the arm
        away otherwise. Otherwise (world-frame obs) lift into the env-relative world frame.

        ``link_pose`` defaults to the world model's current pose for ``link``.
        """
        if link_pose is None:
            link_pose = self.world.get_node_pose(link)
        offset = self._ik_offsets.get(link)
        if offset is not None:
            link_pose = link_pose.multiply(offset)
        if self.eef_obs_in_base_frame:
            return (
                link_pose.get_matrix()
                .squeeze(0)
                .to(device=self.base_env.device, dtype=torch.float32)
            )
        link_pose = multiply_poses(self.get_env_robot_pose(), link_pose)
        mat = link_pose.get_matrix().squeeze(0).to(device=self.base_env.device, dtype=torch.float32)
        env_origin = self.base_env.scene.env_origins[self.env_id].to(
            dtype=mat.dtype, device=mat.device
        )
        mat[:3, 3] -= env_origin
        return mat

    def plan(self) -> Optional[List[Command]]:
        """Plan toward the goal; cuRobo IK/trajopt call backward(), so run under autograd."""
        with autograd_enabled():
            return super().plan()

    def hold(self, steps: int = 100) -> List[Command]:
        """Null plan: hold the current configuration for ``steps`` steps."""
        return steps * [self.world.configuration()]

    def create_rollout_controller(
        self, commands: Optional[List[Command]]
    ) -> Optional[PathController]:
        """Roll ``commands`` out from the planning state and precompute per-step controller targets.

        Uses the cuStream2 execution model: each leaf command's ``execute()`` mutates the world
        one step at a time. The gripper is inferred from the world's arm attachments (closed
        while holding), not from the command stream. IK link targets are frame-corrected via
        :func:`eef_frame_corrections` so cuRobo's link frame maps onto the env's EEF observation
        convention.
        """
        self.state.set()
        if commands is None:
            return None

        links = get_action_links(self.base_env)
        corrections = eef_frame_corrections(
            self.base_env, self.env_id, self.raw_control_pose, links
        )
        return rollout_path_controller(
            self.state,
            commands,
            gripper=binary_gripper_actions(self.base_env),
            links=links,
            link_pose=lambda link: self.raw_control_pose(link) @ corrections[link],
            joint_space=bool(get_action_joints(self.base_env)),
        )

    def create_task_controller(self, commands: Optional[List[Command]]) -> Optional[PathController]:
        """Precompute the controller from task-space commands alone (no world rollout).

        Thin wrapper around :func:`task_path_controller` (shared with Arena) supplying this
        planner's state/env/frame hooks.
        """
        return task_path_controller(
            self.state, self.base_env, self.env_id, self.raw_control_pose, commands
        )

    def create_joint_controller(
        self, commands: Optional[List[Command]]
    ) -> Optional[PathController]:
        """Precompute the controller from joint-space commands alone (no world rollout).

        Thin wrapper around :func:`joint_path_controller` (shared with Arena) supplying this
        planner's state/env hooks.
        """
        return joint_path_controller(self.state, self.base_env, commands)

    def create_multi_robot_controller(
        self, commands: Optional[List[Command]]
    ) -> Optional[ActionController]:
        """Roll ``commands`` out and precompute one joint-space action per step for multi-robot envs.

        Uses the cuStream2 execution model: each leaf command's ``execute()`` mutates the world
        one step at a time. Each step's action concatenates the target position for every action
        term (arm AND gripper joints alike), in the env's own action-term order -- reading each
        term's own (per-robot, un-prefixed) joint names against the combined world's
        ``arm{i}_``-prefixed naming from :func:`isaaclab.robot.build_multi_robot_config`.

        A ``BinaryJointPositionAction`` term contributes exactly one value to the action vector
        regardless of how many joints its ``joint_names`` regex matches (e.g. a parallel gripper's
        two finger joints, both driven by the same open/close scalar) -- unlike a continuous
        ``JointPositionAction`` term, whose action dim equals its matched joint count. Those get
        that one value from :func:`binary_gripper_actions`'s ``attached_arms``-derived open/close
        pair, like its other callers. Continuous gripper joints (e.g. the droid task's single
        ``finger_joint``) aren't read from the world's kinematic joint value (locked/inactive
        joints don't reliably track ``Open``/``Close``'s target there); instead, the arm's
        ``attached_arms`` flag picks between :meth:`World.get_finger_limit`'s (closed, opened)
        bounds directly.

        Continuous terms' literal joint-angle targets are run through :func:`unscale_action`
        before being appended -- a ``JointPositionActionCfg`` with ``scale``/``use_default_offset``
        (e.g. this stack family's ``scale=0.5, use_default_offset=True``) interprets the raw action
        as ``target = raw * scale + offset``, so sending the literal target as the raw action
        (as this used to) silently commands the wrong joint position under any non-identity
        scale/offset. It was previously only exercised by the droid task's ``scale=1.0,
        use_default_offset=False`` gripper, where the missing unscale is a no-op, which is why the
        bug (visible as e.g. arms drifting off the held configuration during ``--hold``) went
        unnoticed until a task with a real scale/offset (like the dual-Franka stack task) used
        this multi-robot path.
        """
        self.state.set()
        if commands is None:
            return None

        binary_open_action = binary_close_action = None
        if any(
            isinstance(term, BinaryJointPositionAction) for term in get_action_terms(self.base_env)
        ):
            binary_open_action, binary_close_action = binary_gripper_actions(self.base_env)

        term_specs = []
        for term in get_action_terms(self.base_env):
            arm = f"{ARM_PREFIX}{self.robots.index(term.cfg.asset_name)}"
            if isinstance(term, BinaryJointPositionAction):
                term_specs.append((term, arm, None))
                continue
            names = getattr(term, "joint_names", None) or getattr(term, "_joint_names", [])
            finger_joints = set(self.world.get_finger_joints(arm))
            term_specs.append(
                (term, arm, [(f"{arm}_{name}", f"{arm}_{name}" in finger_joints) for name in names])
            )

        actions = []
        for command in Commands.flatten(commands):
            for _ in command.execute():
                positions = []
                for term, arm, joints in term_specs:
                    if joints is None:
                        positions.append(
                            binary_close_action
                            if self.world.attached_arms[arm]
                            else binary_open_action
                        )
                        continue
                    joint_names = [joint for joint, _ in joints]
                    targets = []
                    for (joint, is_gripper), position in zip(
                        joints, self.world.get_joint_positions(joint_names)
                    ):
                        if is_gripper:
                            closed, opened = self.world.get_finger_limit(joint)
                            position = closed if self.world.attached_arms[arm] else opened
                        targets.append(position)
                    target = unscale_action(term, self.env_id, self.world.to_device(targets))
                    positions.extend(target.tolist())
                actions.append(self.world.to_device(positions))
        if not actions:
            return None
        return ActionController(actions)

    def create_controller(self, commands: Optional[List[Command]]) -> Optional[Controller]:
        """Dispatch to the rollout, task, or joint controller depending on the env's action space.

        Multi-robot envs use the multi-robot controller (no per-arm frame-corrected task-space
        form yet). Otherwise: task-space (IK-only) envs use the task controller; joint-space envs
        (and hold plans, whose Configuration commands have no task-space form) use the world
        rollout.
        """
        if len(self.robots) != 1:
            return self.create_multi_robot_controller(commands)
        if get_action_joints(self.base_env):
            return self.create_rollout_controller(commands)
        return self.create_task_controller(commands)

    def solve(self, env_id: int = 0) -> Optional[Controller]:
        """Sync the world from env copy ``env_id``, plan (or hold), and precompute the per-step
        controller. Returns None on planning failure; the agent loop resets the env and replans.
        """
        self.env_id = env_id
        self.update_state()
        if self.hold_steps is not None:
            return self.create_controller(self.hold(self.hold_steps))
        reset_globals()
        self.set_goal(self.create_goal(self.success_term))
        commands = self.plan()
        return self.create_controller(commands)

    def show(self, **kwargs: Any) -> None:
        self.world.show(**kwargs)
