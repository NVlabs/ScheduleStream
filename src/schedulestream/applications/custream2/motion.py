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
import argparse
import traceback
from typing import Any, Dict, List, Optional, Union

# Third Party
import torch
from curobo.types import JointState, Pose

# NVIDIA
from schedulestream.applications.custream2.command import (
    Command,
    DeferredCommand,
    LinkPath,
    Trajectory,
)
from schedulestream.applications.custream2.grasp import Grasp
from schedulestream.applications.custream2.placement import Placement
from schedulestream.applications.custream2.retime import (
    resample_waypoints,
    retime_poses,
    retime_waypoints,
)
from schedulestream.applications.custream2.state import Configuration
from schedulestream.applications.custream2.utils import (
    add_spheres,
    extract_joint_state,
    get_optimized_plan,
    zero_joint_state_derivatives,
)
from schedulestream.common.utils import current_time, elapsed_time
from schedulestream.language.stream import Context


def parse_motion_planner(value: str) -> str:
    """argparse type for --motion: "<planner>" or "<planner>:<cost>"."""
    if value.split(":", 1)[0] not in ("curobo", "joint", "task", "linear"):
        raise argparse.ArgumentTypeError(f"invalid motion planner: {value!r}")
    return value


def plan_motion(
    world: Any,
    goal_js: JointState,
    start_js: Optional[JointState] = None,
    motion: str = "curobo",
    shortcut_iterations: int = 5,
    retime: bool = False,
    debug: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Optional[Union[JointState, Pose]]:
    """Plan a trajectory to ``goal_js`` over its joints, with sphere clearance
    applied for the plan's duration.

    ``start_js`` defaults to the world's current joint state. Dispatches on
    ``motion``: linear (straight-line interpolation, no collision checking),
    curobo (plan_cspace), the cspace PRM ("joint[:<cost>]"), or the
    SE(3) tool-pose PRM ("task"). Returns a fixed-dt joint-state trajectory
    over ``goal_js``'s joints — except for "task", which returns the planned
    tool-pose waypoints as a batched Pose — or None on failure.

    ``shortcut_iterations`` bounds the shortcutting rounds in the joint and task planners; 0
    returns the raw roadmap path.

    ``retime`` re-times a curobo plan from the joint velocity limits instead of keeping the
    optimizer's own profile, trading its acceleration limits for a shorter trajectory.
    """
    joints = list(goal_js.joint_names)
    if start_js is None:
        start_js = world.get_joint_state(joints)
    pos1 = extract_joint_state(start_js, joints).position
    pos2 = extract_joint_state(goal_js, joints).position
    start_js = JointState.from_position(pos1, joint_names=joints)
    goal_js = JointState.from_position(pos2, joint_names=joints)

    if motion == "linear":
        wp = JointState.from_position(torch.cat([pos1, pos2], dim=0), joint_names=joints)
        return retime_waypoints(world, wp)

    start_full = world.reorder_joint_state(start_js)
    goal_full = world.reorder_joint_state(goal_js)
    with world.sphere_context():
        clearance_js = JointState.from_position(
            torch.cat([start_full.position, goal_full.position], dim=0),
            joint_names=list(start_full.joint_names),
        )
        world.set_sphere_clearance(
            clearance_js, clearance=world.tamp_config.motion_sphere_clearance
        )
        motion, _, cost = motion.partition(":")
        if motion == "joint":
            # NVIDIA
            from schedulestream.applications.custream2.joint_prm import plan_joint_path

            plan_kwargs = {"cost": cost} if cost else {}
            traj_js = plan_joint_path(
                world,
                goal_js,
                start_js=start_js,
                shortcut_iterations=shortcut_iterations,
                verbose=verbose,
                **plan_kwargs,
            )
            traj_js = retime_waypoints(world, traj_js)
        elif motion == "task":
            assert not cost, f"cost suffix unsupported for {motion!r}"
            # NVIDIA
            from schedulestream.applications.custream2.task_prm import plan_task_path

            [arm] = [a for a in world.arms if set(joints) <= set(world.get_arm_joints(a))]
            traj_js = plan_task_path(
                world,
                arm,
                goal_js,
                start_js=start_js,
                shortcut_iterations=shortcut_iterations,
                verbose=verbose,
            )
            traj_js = retime_poses(world, traj_js)
        elif motion == "curobo":
            assert not cost, f"cost suffix unsupported for {motion!r}"
            for js in (start_full, goal_full):
                zero_joint_state_derivatives(js)
            world.set_joint_cspace_weights(joints)
            result = world.plan_cspace(
                goal_full, start_full, max_attempts=5, enable_graph_attempt=1, **kwargs
            )
            traj_js = get_optimized_plan(result)
            if traj_js is not None:
                traj_js = extract_joint_state(traj_js, joints)
                traj_js = (retime_waypoints if retime else resample_waypoints)(world, traj_js)
        else:
            raise ValueError(f"Unrecognized motion planner: {motion!r}")
        if debug and (traj_js is None):
            scene = world.active_scene()
            add_spheres(world.get_spheres(start_full), color=(0, 0, 255, 100), scene=scene)
            add_spheres(world.get_spheres(goal_full), color=(0, 255, 0, 100), scene=scene)
            scene.show()
    if (traj_js is None) or isinstance(traj_js, Pose):
        return traj_js
    traj_js.position = traj_js.position.view(-1, traj_js.position.shape[-1])
    return extract_joint_state(traj_js, joints)


class DeferredTrajectory(DeferredCommand):
    """Trajectory to a target joint state, planned at execution time from the
    world's then-current state via :func:`plan_motion`."""

    def __init__(
        self,
        world: Any,
        goal_js: JointState,
        grasped: Optional[str] = None,
        collider: Optional[Command] = None,
        motion: str = "curobo",
        **kwargs: Any,
    ) -> None:
        super().__init__(world, collider=collider)
        self.goal_js = goal_js
        self.grasped = grasped
        self.motion = motion
        self.kwargs = dict(kwargs)

    @property
    def joints(self) -> List[str]:
        """All joint names in the target joint state."""
        return list(self.goal_js.joint_names)

    @property
    def arms(self) -> List[str]:
        """Arms whose joints contain all the target joints."""
        joints = set(self.joints)
        return [a for a in self.world.arms if joints <= set(self.world.get_arm_joints(a))]

    @property
    def arm(self) -> str:
        """The single arm this trajectory controls; raises if not exactly one."""
        arms = self.arms
        if len(arms) != 1:
            raise ValueError(f"Expected exactly 1 arm, got {arms}")
        return arms[0]

    @property
    def tool_link(self) -> str:
        """The tool link of the single arm this trajectory controls."""
        return self.world.get_tool_link(self.arm)

    @property
    def is_task_space(self) -> bool:
        """True when the deferred plan is over the tool pose (materializes a LinkPath)."""
        return self.motion == "task"

    def get_command(self) -> Optional[Union[Trajectory, LinkPath]]:
        """Plan to the target joint state; kwargs forward to plan_motion.

        When ``grasped`` is set, the named object is grasped online at its
        current pose relative to the arm's tool link and carried by the plan.
        """
        grasps = []
        if self.grasped is not None:
            grasps.append(Grasp(self.world, self.grasped, self.tool_link))
        for grasp in grasps:
            grasp.attach()
        traj_js = plan_motion(self.world, self.goal_js, motion=self.motion, **self.kwargs)
        for grasp in grasps:
            grasp.detach()
        if traj_js is None:
            return None
        if isinstance(traj_js, Pose):
            return LinkPath(self.world, self.tool_link, traj_js, grasps=grasps)
        return Trajectory(self.world, traj_js, grasps=grasps)

    def __str__(self) -> str:
        return f"dt{id(self) % 1000}"

    __repr__ = __str__


def motion_stream(
    world: Any,
    arm: str,
    conf1: Configuration,
    conf2: Configuration,
    grasp: Optional[Grasp] = None,
    context: Optional[Context] = None,
    collisions: bool = True,
    defer: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Optional[Union[Trajectory, LinkPath, DeferredTrajectory]]:
    """Plan a trajectory between conf1 and conf2 for arm.

    When ``motion="linear"``, skip cspace planning and return a straight-line
    interpolation of the two joint configurations (mirrors custream v1's
    motion_stream linear branch).
    """
    if not collisions:
        context = None
    start_time = current_time()
    joints = world.get_arm_joints(arm)
    start_js = JointState.from_position(conf1.joint_state.position.clone(), joint_names=joints)
    goal_js = JointState.from_position(conf2.joint_state.position.clone(), joint_names=joints)

    placements: List[Placement] = []
    if context is not None:
        for constraint in context.constraints:
            _, _, _, attachment = constraint.expression.term.unwrap_arguments()
            if attachment is None:
                pass
            elif isinstance(attachment, Placement):
                placements.append(attachment)
            elif isinstance(attachment, Grasp) and attachment.arm == arm:
                grasp = attachment

    if defer:
        return DeferredTrajectory(
            world,
            goal_js,
            grasped=None if grasp is None else grasp.obj,
            verbose=verbose,
            **kwargs,
        )

    placed = [placement.obj for placement in placements]
    state = world.state(joints=[], objects=placed)
    for placement in placements:
        placement.set()
    if grasp is not None:
        grasp.attach()
    grasps = [] if grasp is None else [grasp]
    grasped = [grasp.obj for grasp in grasps]

    inactive_names = (
        set(world.movable_names) - set(placed) if collisions else set(world.object_names)
    )
    with world.active_context(objects=inactive_names):
        world.set_objects_active(active=False, names=inactive_names)
        if verbose:
            active = world.get_active_objects()
            print(
                f"motion_stream) Arm: {arm} | Grasped ({len(grasped)}): {grasped} | "
                f"Placed ({len(placed)}): {placed} | Obstacles ({len(active)}): {active}"
            )
        traj_js = plan_motion(
            world,
            goal_js,
            start_js=start_js,
            verbose=verbose,
            **kwargs,
        )

    if grasp is not None:
        grasp.detach()
    state.set()
    if traj_js is None:
        if verbose:
            print(
                f"motion_stream) Arm: {arm} | Failed | Elapsed: {elapsed_time(start_time):.3f} sec"
            )
        return None
    if isinstance(traj_js, Pose):
        traj = LinkPath(world, world.get_tool_link(arm), traj_js, grasps=grasps)
    else:
        traj = Trajectory(world, traj_js, grasps=grasps)
    if verbose:
        print(
            f"motion_stream) Arm: {arm} | Steps: {traj.length} | Duration: {traj.duration:.3f} sec"
            f" | Elapsed: {elapsed_time(start_time):.3f} sec"
        )
    return traj
