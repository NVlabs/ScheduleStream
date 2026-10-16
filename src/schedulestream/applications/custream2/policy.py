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
"""Env-agnostic TAMP Planner/Policy: plan toward a goal from observation
hooks, then replay the plan one extracted action per step."""

from __future__ import annotations

# Standard Library
import traceback
from collections import Counter
from typing import Any, Iterator, List, Optional

# Third Party
import torch
from curobo.types import JointState, Pose

# NVIDIA
from schedulestream.applications.custream2.animate import animate_commands
from schedulestream.applications.custream2.command import Command, Commands
from schedulestream.applications.custream2.state import Attachment, State
from schedulestream.applications.custream2.tamp import solve_tamp
from schedulestream.applications.custream2.utils import multiply_poses
from schedulestream.applications.custream2.world import World
from schedulestream.common.utils import timeout_context
from schedulestream.language.expression import Formula


class Planner:
    """Plans toward ``goal`` from the observation methods; holds no execution state."""

    def __init__(
        self,
        world: World,
        goal: Optional[Formula] = None,
        max_time: float = 60.0,
        animate: bool = False,
        record: bool = False,
        **solve_kwargs: Any,
    ) -> None:
        self.world = world
        self.state = world.state()
        self.goal = goal
        self.max_time = max_time
        self.animate = animate
        self.record = record
        self.solve_kwargs = dict(solve_kwargs)
        self.frames: List[bytes] = []
        self.errors = Counter()
        self.num_attempts = 0
        self.num_successes = 0
        self.errors_in_row = 0
        self.failures_in_row = 0

    def get_env_robot_pose(self) -> Pose:
        """Return the robot's world root pose."""
        raise NotImplementedError()

    def get_env_object_pose(self, obj: str) -> Pose:
        """Return the named object's world pose."""
        raise NotImplementedError()

    def get_env_joint_state(self) -> JointState:
        """Return the robot's current joint state."""
        raise NotImplementedError()

    def update_joint_state(self, locked: bool = True) -> None:
        """Apply the observed joint state to the world.

        ``locked`` forwards to :meth:`World.set_joint_positions` (False skips
        syncing locked joints to cuRobo).
        """
        joint_state = self.get_env_joint_state()
        self.world.set_joint_positions(
            list(joint_state.joint_names), joint_state.position.flatten().tolist(), locked=locked
        )

    def update_object_pose(self, obj: str, collision: bool = True) -> bool:
        """Write ``obj``'s observed pose to the world, re-expressed in the
        robot's root frame (the World origin).

        Returns False (leaving the current pose) when the observation source
        doesn't track ``obj``. ``collision`` forwards to
        :meth:`World.set_object_pose` (False skips the checker sync).
        """
        try:
            obj_pose = self.get_env_object_pose(obj)
        except KeyError:
            return False
        reference_inv = self.get_env_robot_pose().inverse()
        self.world.set_object_pose(
            obj, multiply_poses(reference_inv, obj_pose), collision=collision
        )
        return True

    def update_state(self, collision: bool = True) -> State:
        """Sync the world from the observation methods and snapshot the state."""
        self.update_joint_state()
        for name in self.world.object_names:
            self.update_object_pose(name, collision=collision)
        self.state = self.world.state()
        return self.state

    def get_world_object_pose(self, obj: str) -> Pose:
        """Return the world model's pose for an object."""
        return self.world.get_object_pose(obj)

    def get_world_link_pose(self, link: str) -> Pose:
        """Return the world model's pose for a robot link."""
        return self.world.get_link_pose(link)

    def get_world_joint_state(self, joints: Optional[List[str]] = None) -> JointState:
        """Return the world model's current joint state."""
        return self.world.get_joint_state(joints=joints)

    def get_world_arm_attachments(self, arm: Optional[str] = None) -> List[Attachment]:
        """Return the Attachments held by ``arm`` (default: the single arm) in the world model."""
        if arm is None:
            [arm] = self.world.arms
        return self.world.attached_arms[arm]

    def extract_action(self) -> Optional[torch.Tensor]:
        """Extract the env action implied by the current world state."""
        raise NotImplementedError()

    def get_controller(
        self, commands: List[Command], update: bool = False
    ) -> Iterator[torch.Tensor]:
        """Step the world through ``commands``, yielding one extracted action per frame.

        When ``update``, re-sync the world from the env before each command.
        A command yielding ``False`` aborts the controller (e.g. an IK-tracked
        step that failed to converge).
        """
        for command in commands:
            if update:
                self.update_state()
            for output in command.execute():
                if output is False:
                    return
                action = self.extract_action()
                if action is None:
                    continue
                yield action
                if update:
                    self.update_joint_state()

    def set_goal(self, goal: Optional[Formula]) -> None:
        """Replace the goal."""
        self.goal = goal

    def reset(self) -> State:
        """Reset the world and sync it via the observation methods."""
        self.world.reset()
        return self.update_state()

    def plan(self) -> Optional[List[Command]]:
        """Plan toward the goal and return the flattened leaf commands.

        Returns None on failure (including timeout or solver errors, which are
        counted in ``self.errors``); with no goal the plan is trivially empty.
        ``self.num_attempts`` counts every call and ``self.num_successes``
        every success; ``self.errors_in_row`` and ``self.failures_in_row``
        count consecutive solver errors and consecutive planning failures
        respectively (a solver error also counts as a planning failure),
        each resetting to 0 on the next success.
        When ``animate``/``record``, the solved plan is animated from the
        planning state and any recorded frames accumulate in ``self.frames``.
        """
        if self.goal is None:
            return []
        self.num_attempts += 1
        self.world.set_default_joint_positions()
        try:
            with timeout_context(timeout=2 * self.max_time):
                commands = solve_tamp(
                    self.state, self.goal, max_time=self.max_time, **self.solve_kwargs
                )
        except Exception as e:
            traceback.print_exc()
            self.errors[repr(e)] += 1
            self.errors_in_row += 1
            commands = None
        else:
            self.errors_in_row = 0
        if self.animate or self.record:
            self.frames.extend(animate_commands(self.state, commands, record=self.record))
        self.state.set()
        if commands is None:
            self.failures_in_row += 1
            return None
        self.failures_in_row = 0
        self.num_successes += 1
        return list(Commands.flatten([commands]))

    def __str__(self) -> str:
        return f"{type(self).__name__}(goal={self.goal})"

    __repr__ = __str__


class Policy(Planner):
    """Planner that also replays the plan: one env action per ``get_action`` call."""

    def __init__(
        self,
        world: World,
        goal: Optional[Formula] = None,
        max_attempts: int = 1,
        **solve_kwargs: Any,
    ) -> None:
        super().__init__(world, goal=goal, **solve_kwargs)
        self.max_attempts = max_attempts
        self.controller: Any = None
        self.step = 0
        self.attempt = 0

    def set_goal(self, goal: Optional[Formula]) -> None:
        """Replace the goal and invalidate the current plan (replans on the next get_action)."""
        super().set_goal(goal)
        self.controller = None
        self.step = 0
        self.attempt = 0

    @property
    def planned(self) -> bool:
        """Return True when planning has run (successfully or not)."""
        return self.controller is not None

    @property
    def failed(self) -> bool:
        """Return True when planning ran and did not find a plan."""
        return self.controller is False

    @property
    def solved(self) -> bool:
        """Return True when planning has run and produced a plan (possibly empty)."""
        return self.planned and not self.failed

    def reset(self) -> State:
        """Sync the world via the observation methods and invalidate the plan.

        Planning happens lazily on the first ``get_action`` call.
        """
        state = super().reset()
        self.controller = None
        self.step = 0
        self.attempt = 0
        return state

    def get_action(self) -> Optional[torch.Tensor]:
        """Return the next action from the controller, or None when exhausted.

        Plans lazily on the first call after a reset or set_goal; caches
        ``False`` on planning failure so planning isn't retried. Each exhausted
        controller counts as one execution attempt: until ``max_attempts`` are
        spent, exhaustion invalidates the controller so the next call replans
        from the current state.
        """
        if self.controller is None:
            self.update_state()
            commands = self.plan()
            if commands is None:
                self.controller = False
            else:
                self.controller = self.get_controller(commands, update=True)
        if isinstance(self.controller, bool):
            return None

        action = next(self.controller, None)
        if action is not None:
            self.step += 1
            return action

        self.attempt += 1
        if self.attempt >= self.max_attempts:
            self.controller = True
            return action

        self.controller = None
        return self.get_action()

    __call__ = get_action

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield actions from get_action until the plan is exhausted (or fails)."""
        while (action := self.get_action()) is not None:
            yield action

    def __str__(self) -> str:
        return f"{type(self).__name__}(goal={self.goal}, steps={self.step})"

    __repr__ = __str__
