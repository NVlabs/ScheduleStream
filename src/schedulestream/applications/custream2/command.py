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
import time
from functools import cached_property
from itertools import count, zip_longest
from typing import TYPE_CHECKING, Any, Iterator, List, Optional

# Third Party
import torch
from curobo._src.util.trajectory import get_batch_interpolated_trajectory
from curobo.types import JointState, Pose

# NVIDIA
from schedulestream.applications.custream2.utils import (
    WORLD,
    add_spheres,
    concatenate_joint_states,
    extract_joint_state,
    multiply_poses,
    transform_spheres,
    unit_pose,
    zero_joint_state_derivatives,
)
from schedulestream.common.utils import INF

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.applications.custream2.placement import Placement
    from schedulestream.applications.custream2.state import Configuration
    from schedulestream.applications.custream2.world import World


def get_placement_parent(placement: Optional[Placement]) -> Optional[str]:
    """The body of a reference ``placement``, or WORLD."""
    return WORLD if placement is WORLD else placement.obj


class Command:
    """Base class for all executable TAMP commands."""

    def __init__(self, world: World, **kwargs: Any) -> None:
        self.world = world
        self.metadata: dict = {}

    @property
    def is_task_space(self) -> bool:
        """Return True when this command is defined in task (pose) space rather than joint space."""
        return False

    @property
    def time_step(self) -> float:
        """Delegate to world.time_step; overridden by subclasses with their own dt."""
        return self.world.time_step

    @property
    def duration(self) -> float:
        """Default duration is one time step; subclasses override."""
        return self.time_step

    def decompose(self) -> Iterator[Command]:
        """Yield leaf command tokens for scheduling (default: self)."""
        yield self

    def get_task_command(self, **kwargs: Any) -> Command:
        """Return the task-space equivalent of this command (default: itself)."""
        return self

    def solve_joint_command(self, **kwargs: Any) -> Optional[Command]:
        """Return the joint-space equivalent of this command (default: execute then return itself).
        """
        for _ in self.execute(**kwargs):
            pass
        return self

    def execute(self, **kwargs: Any) -> Iterator[None]:
        """Advance the world by one or more steps, yielding once per frame."""
        raise NotImplementedError


class DeferredCommand(Command):
    """Command materialized lazily: :meth:`get_command` runs on first execute."""

    def __init__(self, world: World, collider: Optional[Command] = None, **kwargs: Any) -> None:
        super().__init__(world, **kwargs)
        self.collider = collider

    def get_command(self) -> Optional[Command]:
        """Create the underlying command (e.g. by planning); None on failure."""
        raise NotImplementedError

    def execute(self, **kwargs: Any) -> Iterator[None]:
        """Materialize the command and execute it (no-op when get_command fails)."""
        command = self.get_command()
        if command is not None:
            yield from command.execute(**kwargs)

    def __str__(self) -> str:
        return f"{type(self).__name__}()"

    __repr__ = __str__


class Commands(Command):
    """Ordered sequence of commands produced by the TAMP solver."""

    def __init__(
        self,
        world: World,
        commands: Optional[List[Any]] = None,
        collider: Optional[Command] = None,
    ) -> None:
        super().__init__(world)
        self.commands = list(Commands.flatten(commands or []))
        self.collider = collider

    @property
    def duration(self) -> float:
        """Sum of durations of all contained commands."""
        total = 0.0
        for cmd in self.commands:
            total += cmd.duration
        return total

    @property
    def is_task_space(self) -> bool:
        """True if any contained command is task-space."""
        return any(cmd.is_task_space for cmd in self.commands)

    @staticmethod
    def flatten(commands) -> Iterator[Command]:
        """Recursively unwrap nested Commands, yielding only leaf commands."""
        for command in commands:
            if isinstance(command, Commands):
                yield from Commands.flatten(command.commands)
            else:
                yield command

    def decompose(self) -> Iterator[Command]:
        """Yield decomposed leaf tokens from all contained commands."""
        for cmd in self.commands:
            yield from cmd.decompose()

    def execute(self, **kwargs: Any) -> Iterator[None]:
        """Execute contained commands in sequence."""
        for cmd in self.commands:
            yield from cmd.execute(**kwargs)

    def __str__(self) -> str:
        return f"c{id(self) % 1000}"

    __repr__ = __str__


class Composite(Command):
    """Parallel execution of multiple commands (they advance together)."""

    def __init__(self, world: World, commands: List[Any], **kwargs: Any) -> None:
        super().__init__(world, **kwargs)
        self.commands = list(Composite.flatten(commands))

    @property
    def duration(self) -> float:
        """Max duration across parallel commands (the bottleneck)."""
        return max((cmd.duration for cmd in self.commands), default=0.0)

    @property
    def is_task_space(self) -> bool:
        """True if any contained command is task-space."""
        return any(cmd.is_task_space for cmd in self.commands)

    @staticmethod
    def flatten(commands: List[Command]) -> Iterator[Command]:
        """Recursively unwrap nested Composites, yielding leaf commands."""
        for cmd in commands:
            if isinstance(cmd, Composite):
                yield from Composite.flatten(cmd.commands)
            else:
                yield cmd

    def execute(self, **kwargs: Any) -> Iterator[None]:
        """Advance all parallel commands together, padding shorter ones."""
        iters = [cmd.execute(**kwargs) for cmd in self.commands]
        for _ in zip_longest(*iters):
            yield

    def __str__(self) -> str:
        return f"Composite({self.commands})"

    __repr__ = __str__


class Trajectory(Command):
    """Opaque token representing an arm trajectory, optionally relative to the reference frame of
    a ``placement``, such as the surface a motion places onto."""

    def __init__(
        self,
        world: World,
        joint_state: JointState,
        grasps: Optional[List[Any]] = None,
        placement: Optional[Placement] = WORLD,
    ) -> None:
        super().__init__(world)
        self.joint_state = joint_state
        self.grasps = list(grasps or [])
        self.placement = placement
        self.collider = self
        pos = joint_state.position
        if pos.ndim != 2:
            raise ValueError(
                "Trajectory expects a 2D joint state [n_steps, n_joints], got shape"
                f" {tuple(pos.shape)}"
            )
        assert joint_state.dt is not None, "Trajectory requires joint_state.dt to be set"

    @property
    def parent(self) -> Optional[str]:
        """The body of the reference placement, or WORLD."""
        return get_placement_parent(self.placement)

    @property
    def grasped(self) -> List[str]:
        """Return the names of the objects held by this trajectory's grasps."""
        return [grasp.obj for grasp in self.grasps]

    @property
    def joints(self) -> List[str]:
        """All joint names in this trajectory's joint state."""
        return list(self.joint_state.joint_names)

    @property
    def arms(self) -> List[str]:
        """Arm identifiers whose joints appear in this trajectory."""
        js = set(self.joints)
        return [a for a in self.world.arms if set(self.world.get_arm_joints(a)) & js]

    @property
    def arm(self) -> str:
        """The single arm this trajectory controls; raises if not exactly one."""
        arms = self.arms
        if len(arms) != 1:
            raise ValueError(f"Expected exactly 1 arm, got {arms}")
        return arms[0]

    @cached_property
    def time_step(self) -> float:
        """Per-step dt from joint_state; overrides Command.time_step."""
        dt = self.joint_state.dt
        if isinstance(dt, torch.Tensor) and dt.numel() > 1:
            dt = dt.flatten()[0]
        return float(dt)

    @property
    def device(self) -> torch.device:
        """Torch device of the position tensor."""
        return self.positions.device

    @property
    def positions(self) -> torch.Tensor:
        """Raw [n_steps, n_joints] position tensor."""
        return self.joint_state.position

    @property
    def length(self) -> int:
        """Number of waypoints (time steps) in this trajectory."""
        return self.positions.shape[-2]

    @property
    def duration(self) -> float:
        """Total time spanned by the trajectory."""
        return (self.length - 1) * self.time_step

    def configuration(self, index: int, joints: Optional[List[str]] = None) -> Configuration:
        """Extract a single-step Configuration at the given waypoint index."""
        # NVIDIA
        from schedulestream.applications.custream2.state import Configuration

        if joints is None:
            joints = self.joints
        js = extract_joint_state(self.joint_state, joints)
        joint_state = JointState.from_position(
            js.position[index].unsqueeze(0), joint_names=js.joint_names
        )
        label = "endpoint" if index % self.length in (0, self.length - 1) else None
        return Configuration(
            self.world, joint_state, grasps=self.grasps, label=label, placement=self.placement
        )

    def decompose(self) -> Iterator[Configuration]:
        """Yield one Configuration per waypoint."""
        for index in range(self.length):
            yield self.configuration(index)

    def execute(self, **kwargs: Any) -> Iterator[None]:
        """Set joint positions for each waypoint and yield once per step."""
        for waypoint in self.positions.cpu().numpy():
            self.world.set_joint_positions(self.joints, waypoint, locked=False)
            yield

    def show(self, **kwargs: Any) -> None:
        """Step through each waypoint, apply it to the world, and call world.show()."""
        for _ in self.execute():
            self.world.show(**kwargs)

    def reverse(self, grasps: Optional[List[Any]] = None, placement: Any = False) -> Trajectory:
        """Return a new Trajectory with waypoints in reversed order, with ``grasps`` and
        ``placement`` defaulting to this trajectory's; ``placement=WORLD`` makes it the world's."""
        if grasps is None:
            grasps = self.grasps
        if placement is False:
            placement = self.placement
        reversed_js = JointState.from_position(self.positions.flip(0), joint_names=self.joints)
        reversed_js.dt = self.joint_state.dt
        return Trajectory(self.world, reversed_js, grasps=grasps, placement=placement)

    def resample(self, dt: float) -> Trajectory:
        """Return a new Trajectory resampled to the given time step dt."""
        interp_dt = torch.tensor([dt], dtype=torch.float32, device=self.device)
        interp_js, _ = get_batch_interpolated_trajectory(self.joint_state, interp_dt)
        new_js = JointState.from_position(interp_js.position[0], joint_names=self.joints)
        new_js.dt = interp_dt
        return Trajectory(self.world, new_js, grasps=self.grasps, placement=self.placement)

    def link_path(self, arm: Optional[str] = None) -> LinkPath:
        """Return the tool-link LinkPath traced by this trajectory (FK per waypoint).

        ``arm`` defaults to the single arm this trajectory controls.
        """
        if arm is None:
            arm = self.arm
        poses = self.world.get_tool_pose(arm, self.joint_state)
        tool_link = self.world.get_tool_link(arm)
        grasps = [grasp for grasp in self.grasps if grasp.link == tool_link]
        return LinkPath(self.world, tool_link, poses, grasps=grasps, placement=self.placement)

    def object_path(self, obj: Optional[str] = None) -> ObjectPath:
        """Return the world-frame ObjectPath traced by grasped ``obj`` (FK per waypoint composed
        with its grasp).

        ``obj`` defaults to the single object this trajectory grasps.
        """
        if obj is None:
            [obj] = self.grasped
        grasp = self.grasps[self.grasped.index(obj)]
        link_poses = self.world.get_tool_poses(self.joint_state)[grasp.link]
        grasp_pose = grasp.pose.to(device=link_poses.position.device)
        poses = multiply_poses(link_poses, grasp_pose)
        return ObjectPath(self.world, obj, poses, placement=self.placement)

    def pose_path(self) -> PosePath:
        """Return the ObjectPath of the single grasped object if any, else the LinkPath of the
        single arm."""
        return self.object_path() if self.grasps else self.link_path()

    def get_task_command(self, **kwargs: Any) -> LinkPath:
        """Return the task-space equivalent of this trajectory: its tool-link LinkPath."""
        return self.link_path()

    def __str__(self) -> str:
        return f"t{id(self) % 1000}"

    __repr__ = __str__
    __hash__ = object.__hash__
    __eq__ = object.__eq__


class PosePath(Command):
    """Timed path of world-frame poses (one batched Pose per waypoint), optionally relative to the
    reference frame of a ``placement``, like a Trajectory's."""

    def __init__(self, world: World, poses: Pose, placement: Optional[Placement] = WORLD) -> None:
        super().__init__(world)
        self.poses = poses
        self.placement = placement

    @property
    def parent(self) -> Optional[str]:
        """The body of the reference placement, or WORLD."""
        return get_placement_parent(self.placement)

    def get_parent_pose(self) -> Pose:
        """The world pose of the reference placement, or the identity for the WORLD."""
        return unit_pose() if self.placement is WORLD else self.placement.get_pose()

    @property
    def is_task_space(self) -> bool:
        return True

    @property
    def length(self) -> int:
        """Number of poses (time steps) in this path."""
        return self.poses.batch_size

    @property
    def duration(self) -> float:
        """Total time spanned by the path."""
        return max(0, self.length - 1) * self.time_step

    def pose(self, index: int) -> Pose:
        """Return the single pose at the given waypoint index."""
        return self.poses[index]

    def spheres(self, **kwargs: Any) -> torch.Tensor:
        """Return ``[N, S, 4]`` world-frame collision spheres along the path."""
        raise NotImplementedError

    def show(self, **kwargs: Any) -> None:
        """Show the world's active scene with this path's spheres overlaid."""
        scene = self.world.active_scene()
        add_spheres(self.spheres(), scene=scene)
        scene.show(**kwargs)

    __hash__ = object.__hash__
    __eq__ = object.__eq__


def execute_until_converged(
    world: World,
    joints: List[str],
    positions: Any,
    tolerance: float = math.radians(INF),
    max_distance: Optional[float] = math.radians(30),
    p: float = INF,
    max_steps: Optional[int] = None,
    verbose: bool = False,
) -> Iterator[None]:
    """Command ``positions`` and yield at least once, then until the world's
    ``joints`` reach them.

    The target is set on the world before every yield, so closed-loop
    controllers (which re-sync world joints from the env between yields) keep
    commanding it while dwelling; in open-loop execution the world holds the
    command, so this yields exactly once. ``max_distance`` truncates each
    commanded step to that joint-space distance from the current position.
    """
    target = torch.as_tensor(positions, dtype=torch.float32).flatten().cpu()
    for step in count() if max_steps is None else range(max_steps):
        current = torch.as_tensor(world.get_joint_positions(joints), dtype=torch.float32)
        command = target
        if max_distance is not None:
            distance = (target - current).norm(p=p)
            if distance > max_distance:
                command = current + (target - current) * (max_distance / distance)
        world.set_joint_state(
            JointState.from_position(command.unsqueeze(0), joint_names=list(joints))
        )
        yield
        current = torch.as_tensor(world.get_joint_positions(joints), dtype=torch.float32)
        difference = (current - target).abs()
        distance = difference.norm(p=p)
        converged = bool(distance <= tolerance)
        if verbose and not converged:
            print(
                f"{execute_until_converged.__name__}) Step: {step} | Joints: {len(joints)} |"
                f" Last joint: {joints[-1]} |"
                f" Converged: {converged} |"
                f" Distance: {distance:.4f} |"
                f" Difference: {[round(v, 3) for v in difference.tolist()]}"
            )
        if converged:
            return


class LinkPath(PosePath):
    """Path of world-frame poses for a robot link."""

    def __init__(
        self,
        world: World,
        link: str,
        poses: Pose,
        grasps: Optional[List[Any]] = None,
        placement: Optional[Placement] = WORLD,
    ) -> None:
        super().__init__(world, poses, placement=placement)
        self.link = link
        self.grasps = list(grasps or [])

    @property
    def grasped(self) -> List[str]:
        """Return the names of the objects held by this path's grasps."""
        return [grasp.obj for grasp in self.grasps]

    @property
    def arm(self) -> str:
        """The arm that owns this path's link."""
        return self.world.get_link_arm(self.link)

    @property
    def arm_joints(self) -> List[str]:
        """The joints of the arm that owns this path's link."""
        return self.world.get_arm_joints(self.arm)

    def spheres(self, **kwargs: Any) -> torch.Tensor:
        """Return the arm's hand spheres at each pose; kwargs forward to
        :meth:`World.get_hand_spheres`."""
        return self.world.get_hand_spheres(self.arm, self.poses, **kwargs)

    def execute(
        self,
        relax_collisions: bool = True,
        max_difference: float = INF,
        strict: bool = False,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Iterator[None]:
        """Track the path with IK, yielding once per step.

        Each solve is warm-started with the previous step's solutions and, for
        branch continuity, the successful solution nearest the current arm
        configuration is applied (the state is left unchanged when a solve
        fails). ``max_difference`` bounds the per-joint jump allowed between the
        current state and the IK target; larger jumps trigger a warning, or a
        ``ValueError`` when ``strict`` is set. ``relax_collisions`` retries a
        failed solve with objects deactivated.
        """
        # NVIDIA
        from schedulestream.applications.custream2.streams import _get_goal_tool_pose

        ik_solver = self.world.get_arm_ik_solver(self.arm)
        self.world.set_arm_ik_weights(self.arm)
        hand_links = self.world.get_hand_links(self.arm)
        start_time = time.time()
        worst_difference = 0.0
        num_waypoints = 0
        with self.world.sphere_context(links=hand_links):
            self.world.disable_link_spheres(hand_links)
            for index in range(self.length):
                world_js = self.world.reorder_joint_state(self.world.get_joint_state())
                zero_joint_state_derivatives(world_js)
                goal = _get_goal_tool_pose(
                    self.world,
                    self.arm,
                    self.pose(index),
                    tool_frames=ik_solver.kinematics.tool_frames,
                )
                arm_js = extract_joint_state(world_js, ik_solver.joint_names)
                seed_config = arm_js.position.reshape(1, 1, -1)
                result = ik_solver.solve_pose(
                    goal, arm_js, seed_config=seed_config, return_seeds=self.world.ik_seeds
                )
                if relax_collisions and ((result is None) or not bool(result.success.any())):
                    with self.world.active_context(objects=self.world.object_names):
                        self.world.set_objects_active(active=False, names=self.world.object_names)
                        result = ik_solver.solve_pose(
                            goal, arm_js, seed_config=seed_config, return_seeds=self.world.ik_seeds
                        )
                    if verbose and (result is not None) and bool(result.success.any()):
                        print(
                            f"{self}) waypoint {index}/{self.length}: solved with collisions"
                            " relaxed"
                        )
                if (result is None) or not bool(result.success.any()):
                    message = f"{self}) waypoint {index}/{self.length}: IK failed, skipping"
                    if strict:
                        raise RuntimeError(message)
                    if verbose:
                        print(message)
                    continue
                js = result.js_solution
                positions = js.position.reshape(-1, js.position.shape[-1])
                solutions = JointState.from_position(positions, joint_names=list(js.joint_names))
                arm_positions = extract_joint_state(solutions, self.arm_joints).position
                current = extract_joint_state(world_js, self.arm_joints).position.to(
                    arm_positions.device
                )
                distances = torch.norm(arm_positions - current, dim=-1)
                distances[~result.success.reshape(-1)] = torch.inf
                index_best = int(distances.argmin())
                solution = JointState.from_position(
                    positions[index_best].unsqueeze(0), joint_names=list(js.joint_names)
                )
                target = extract_joint_state(solution, self.arm_joints).position
                difference = float((target.flatten() - current.flatten()).abs().max())
                worst_difference = max(worst_difference, difference)
                if difference > max_difference:
                    message = (
                        f"{self}) waypoint {index}/{self.length}: difference {difference:.4f} "
                        f"exceeds max_difference {max_difference:.4f}"
                    )
                    if strict:
                        raise RuntimeError(message)
                    if verbose:
                        print(message)
                num_waypoints += 1
                yield from execute_until_converged(self.world, self.arm_joints, target.flatten())
        if verbose:
            print(
                f"{self}) Arm: {self.arm} | Waypoints: {num_waypoints}/{self.length} | "
                f"Max difference: {math.degrees(worst_difference):.4f} deg | "
                f"Elapsed: {time.time() - start_time:.2f}s"
            )

    def solve_joint_command(self, **kwargs: Any) -> Optional[Trajectory]:
        """Execute the IK path and return the resulting joint-space Trajectory,
        or None if execution raises a RuntimeError."""
        joint_states = []
        try:
            for _ in self.execute(**kwargs):
                joint_states.append(self.world.get_joint_state(self.arm_joints))
        except RuntimeError:
            return None
        js = concatenate_joint_states(joint_states)
        js.dt = torch.tensor(self.world.time_step, device=js.position.device)
        return Trajectory(self.world, js, grasps=self.grasps, placement=self.placement)

    def __str__(self) -> str:
        return f"lp{id(self) % 1000}"

    __repr__ = __str__


class ObjectPath(PosePath):
    """Path of world-frame poses for an object."""

    def __init__(
        self, world: World, obj: str, poses: Pose, placement: Optional[Placement] = WORLD
    ) -> None:
        super().__init__(world, poses, placement=placement)
        self.obj = obj

    def spheres(self, **kwargs: Any) -> torch.Tensor:
        """Return the object's collision spheres at each pose."""
        obj_spheres = self.world.to_device(self.world.get_object(self.obj).spheres_tensor)
        return transform_spheres(self.poses, obj_spheres)

    def execute(self, **kwargs: Any) -> Iterator[None]:
        """Set the object pose for each waypoint and yield once per step."""
        for index in range(self.length):
            self.world.set_object_pose(self.obj, self.pose(index), collision=False)
            yield

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.obj})"

    __repr__ = __str__


class HandCommand(Command):
    def __init__(self, world: World, arm: str, conf: Any) -> None:
        super().__init__(world)
        self.arm = arm
        self.conf = conf

    @property
    def finger_joints(self) -> List[str]:
        """Finger joints driven by this gripper command."""
        return self.world.get_finger_joints(self.arm)

    def execute(self, **kwargs: Any) -> Iterator[None]:
        """Drive the gripper to ``self.conf`` and yield."""
        joints = self.finger_joints
        if joints:
            self.world.set_joint_positions(joints, self.conf)
        yield

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.arm})"

    __repr__ = __str__


class Open(HandCommand):
    def __init__(self, world: World, arm: str) -> None:
        joints = world.get_finger_joints(arm)
        _, opened = world.get_finger_limits(joints) if joints else ([], [])
        super().__init__(world, arm, conf=opened)


class Close(HandCommand):
    def __init__(self, world: World, arm: str) -> None:
        joints = world.get_finger_joints(arm)
        closed, _ = world.get_finger_limits(joints) if joints else ([], [])
        super().__init__(world, arm, conf=closed)


class Attach(Command):
    def __init__(self, world: World, obj: str, link: str, grasp: Optional[Any] = None) -> None:
        super().__init__(world)
        self.obj = obj
        self.link = link
        self.grasp = grasp

    def execute(self, **kwargs: Any) -> Iterator[None]:
        """Attach obj to link in the world at the stored grasp pose and yield."""
        grasp_pose = self.grasp.pose if self.grasp is not None else None
        self.world.attach_object(self.obj, self.link, grasp_pose=grasp_pose)
        yield

    def __str__(self) -> str:
        return f"Attach({self.obj}, {self.link})"

    __repr__ = __str__


class Detach(Command):
    def __init__(self, world: World, link: str) -> None:
        super().__init__(world)
        self.link = link

    def execute(self, **kwargs: Any) -> Iterator[None]:
        """Detach all objects from link in the world and yield."""
        self.world.detach_objects(links=[self.link])
        yield

    def __str__(self) -> str:
        return f"Detach({self.link})"

    __repr__ = __str__
