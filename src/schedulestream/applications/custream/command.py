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
from collections.abc import Sequence
from functools import cached_property
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional

# Third Party
import numpy as np
import torch
import trimesh
from curobo.types.math import Pose
from curobo.types.robot import JointState
from scipy.interpolate import interp1d

# NVIDIA
from schedulestream.applications.blocksworld.visualize import linear_curve
from schedulestream.applications.custream.spheres import add_spheres
from schedulestream.applications.custream.utils import (
    INF,
    concatenate_joint_states,
    extract_joint_state,
    multiply_poses,
    to_cpu,
    transform_spheres,
)
from schedulestream.common.utils import flatten, remove_duplicates

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.applications.custream.state import Attachment, State
    from schedulestream.applications.custream.world import World


class Command:
    def __init__(
        self, world: World, state: Optional[State] = None, metadata: Optional[dict] = None
    ):
        # NVIDIA
        from schedulestream.applications.custream.state import State

        self.world = world
        if state is None:
            state = State(world)
        self.state = state
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    @property
    def time_step(self) -> float:
        return self.world.time_step

    @property
    def objects(self) -> List[str]:
        return self.state.objects

    @property
    def attachments(self) -> List[Attachment]:
        return self.state.attachments

    def iterator(self, **kwargs: Any) -> Iterator[None]:
        raise NotImplementedError()

    def execute(self, **kwargs: Any) -> Iterator[None]:
        for _ in self.iterator(**kwargs):
            self.world.set()
            yield None

    def _decompose(self) -> Iterator["Command"]:
        yield self

    def decompose(self, **kwargs: Any) -> Iterator["Command"]:
        for command in self._decompose():
            for _ in command.execute(**kwargs):
                pass
            yield command

    def __str__(self) -> str:
        prefix = self.__class__.__name__.lower()[0]
        return f"{prefix}{id(self) % 1000}"

    __repr__ = __str__


class Commands(Command):
    def __init__(self, world: World, commands: Iterable[Command], **kwargs: Any):
        super().__init__(world, **kwargs)
        self.commands = list(Commands.flatten(commands))

    @staticmethod
    def flatten(commands: Iterable[Command]) -> Iterator[Command]:
        for command in commands:
            if isinstance(command, Commands):
                yield from Commands.flatten(command.commands)
            else:
                yield command

    @property
    def arms(self) -> List[str]:
        return remove_duplicates(command.arm for command in self.commands)

    @property
    def arm(self) -> str:
        assert len(self.arms) == 1, self.arms
        return self.arms[0]

    @property
    def trajectories(self) -> List["Trajectory"]:
        return [traj for traj in self.commands if isinstance(traj, Trajectory)]

    @cached_property
    def joint_state(self) -> JointState:
        return concatenate_joint_states(traj.joint_state for traj in self.trajectories)

    def __len__(self) -> int:
        return sum(map(len, self.commands))

    @property
    def duration(self) -> float:
        return sum(command.duration for command in self.commands)

    def iterator(self, **kwargs: Any) -> Iterator[None]:
        for command in self.commands:
            for step, _ in enumerate(command.iterator(**kwargs)):
                yield command, step

    def _decompose(self) -> Iterator["Command"]:
        for command in self.commands:
            yield from command._decompose()


class Composite(Command):
    def __init__(self, world: World, commands: Iterable[Command], **kwargs: Any):
        super().__init__(world, **kwargs)
        self.commands = list(Composite.flatten(commands))

    def __len__(self) -> int:
        return max(map(len, self.commands), default=0)

    @property
    def arms(self) -> List[str]:
        return remove_duplicates(flatten(command.arms for command in self.commands))

    @property
    def duration(self) -> float:
        return max([command.duration for command in self.commands], default=0)

    @staticmethod
    def flatten(commands: Iterable[Command]) -> Iterator[Command]:
        for command in commands:
            if isinstance(command, Composite):
                yield from Composite.flatten(command.commands)
            else:
                yield command

    def iterator(self, **kwargs: Any) -> Iterator[None]:
        iterators = [command.iterator(**kwargs) for command in self.commands]
        for _ in zip_longest(*iterators):
            yield

    def _decompose(self) -> Iterator["Command"]:
        iterators = [command._decompose() for command in self.commands]
        for commands in zip_longest(*iterators):
            commands = [command for command in commands if command is not None]
            yield Composite(self.world, commands)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.commands})"

    __repr__ = __str__


class Attach(Command):
    def __init__(
        self,
        world: World,
        obj: str,
        parent: Optional[str] = None,
        pose: Optional[Pose] = None,
        num_steps: int = 1,
        **kwargs: Any,
    ):
        super().__init__(world, **kwargs)
        self.obj = obj
        self.parent = parent
        self.pose = pose
        self.num_steps = num_steps

    @property
    def arm(self) -> Optional[str]:
        if self.parent not in self.world.tool_links:
            return None
        return self.world.get_link_arm(self.parent)

    def __len__(self) -> int:
        return self.num_steps

    @property
    def duration(self) -> float:
        return self.num_steps * self.time_step

    def attachment(self) -> Attachment:
        # NVIDIA
        from schedulestream.applications.custream.state import Attachment

        return Attachment(self.world, self.obj, parent=self.parent, pose=self.pose)

    def iterator(self, **kwargs: Any) -> Iterator[None]:
        if self.arm is None:
            parent_pose = self.world.get_object_pose(self.parent)
            pose = multiply_poses(parent_pose, self.pose)
            self.world.set_object_pose(self.obj, pose)
        else:
            self.world.attach_object(self.obj, self.parent, grasp_pose=self.pose)
        for _ in range(self.num_steps):
            yield

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.obj}, {self.parent})"

    __repr__ = __str__


class Detach(Command):
    def __init__(self, world: World, parent: str, num_steps: int = 1, **kwargs: Any):
        super().__init__(world, **kwargs)
        self.parent = parent
        self.num_steps = num_steps

    @property
    def arm(self) -> Optional[str]:
        if self.parent not in self.world.tool_links:
            return None
        return self.world.get_link_arm(self.parent)

    def __len__(self) -> int:
        return self.num_steps

    @property
    def duration(self) -> float:
        return self.num_steps * self.time_step

    def iterator(self, **kwargs: Any) -> Iterator[None]:
        self.world.detach_objects(links=[self.parent])
        for _ in range(self.num_steps):
            yield

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.parent})"

    __repr__ = __str__


class Open(Command):
    def __init__(self, world: World, arm: str, num_steps: Optional[int] = 10, **kwargs: Any):
        super().__init__(world, **kwargs)
        assert num_steps >= 1
        self.arm = arm
        self.num_steps = num_steps

    def __len__(self) -> int:
        return self.num_steps

    @property
    def duration(self) -> float:
        return self.num_steps * self.time_step

    @property
    def gripper_joints(self) -> List[str]:
        return self.world.get_gripper_joints(self.arm)

    def trajectory(self) -> Optional["Trajectory"]:
        # NVIDIA
        from schedulestream.applications.custream.grasp import open_gripper

        joint_state = open_gripper(self.world, self.arm, num_steps=self.num_steps)
        return Trajectory(self.world, joint_state, metadata=self.metadata)

    def iterator(self, **kwargs: Any) -> Iterator[None]:
        trajectory = self.trajectory()
        yield from trajectory.iterator(**kwargs)

    def _decompose(self) -> Iterator["Command"]:
        trajectory = self.trajectory()
        yield from trajectory._decompose()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.arm})"

    __repr__ = __str__


class Close(Command):
    def __init__(self, world: World, arm: str, num_steps: Optional[int] = 10, **kwargs: Any):
        super().__init__(world, **kwargs)
        assert num_steps >= 1
        self.arm = arm
        self.num_steps = num_steps

    def __len__(self) -> int:
        return self.num_steps

    @property
    def duration(self) -> float:
        return self.num_steps * self.time_step

    @property
    def gripper_joints(self) -> List[str]:
        return self.world.get_gripper_joints(self.arm)

    def trajectory(self) -> Optional["Trajectory"]:
        # NVIDIA
        from schedulestream.applications.custream.grasp import (
            close_until_contact,
            interpolate_gripper,
        )

        if not self.gripper_joints:
            return None

        joint_state = close_until_contact(self.world, self.arm)
        if self.num_steps is not None:
            closed_conf = to_cpu(joint_state.position[-1])
            joint_state = interpolate_gripper(
                self.world, self.arm, closed_conf, num_steps=self.num_steps
            )
        return Trajectory(self.world, joint_state, metadata=self.metadata)

    def iterator(self, **kwargs: Any) -> Iterator[None]:
        trajectory = self.trajectory()
        if trajectory is not None:
            yield from trajectory.iterator(**kwargs)

    def _decompose(self) -> Iterator["Command"]:
        trajectory = self.trajectory()
        if trajectory is not None:
            yield from trajectory._decompose()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.arm})"

    __repr__ = __str__


class Trajectory(Command):
    def __init__(self, world: World, joint_state: JointState, **kwargs: Any):
        super().__init__(world, **kwargs)
        assert joint_state is not None
        assert len(joint_state.shape) == 2, joint_state.shape
        self.joint_state = joint_state

    def __getitem__(self, index: int) -> JointState:
        return self.get_index(index)

    def __len__(self) -> int:
        return self.length

    @property
    def commands(self) -> List[Command]:
        return [self]

    @property
    def joints(self) -> List[str]:
        return self.joint_state.joint_names

    @property
    def links(self) -> List[str]:
        return self.world.yourdf.get_active_links(self.joints)

    @property
    def dim(self) -> int:
        return len(self.joints)

    @property
    def length(self) -> int:
        return len(self.joint_state)

    @property
    def duration(self) -> float:
        return self.length * self.time_step

    @property
    def arms(self) -> List[str]:
        return [
            arm
            for arm in self.world.arms
            if set(self.joints)
            & (set(self.world.get_arm_joints(arm)) | set(self.world.get_gripper_joints(arm)))
        ]

    @property
    def arm(self) -> Optional[str]:
        if len(self.arms) == 1:
            return self.arms[0]
        return None

    @property
    def grasps(self) -> List["Grasp"]:
        # NVIDIA
        from schedulestream.applications.custream.grasp import Grasp

        return [grasp for grasp in self.state.attachments if isinstance(grasp, Grasp)]

    @property
    def has_active(self) -> bool:
        return bool(set(self.joints) & set(self.world.active_joints))

    @property
    def has_locked(self) -> bool:
        return bool(set(self.joints) & set(self.world.locked_joints))

    @property
    def positions(self) -> torch.Tensor:
        return self.joint_state.position

    @property
    def differences(self) -> torch.Tensor:
        return self.joint_state.position[:-1, :] - self.joint_state.position[1:, :]

    @property
    def distances(self):
        return torch.norm(self.differences, dim=1, p=INF)

    @property
    def times(self) -> np.ndarray:
        return np.cumsum([0.0] + (len(self) - 1) * [self.time_step])

    @property
    def curve(self) -> interp1d:
        positions = to_cpu(self.positions)
        return linear_curve(self.times, positions)

    @property
    def start(self) -> JointState:
        return self.joint_state[:1]

    @property
    def end(self) -> JointState:
        return self.joint_state[-1:]

    @property
    def reversed_joint_state(self) -> JointState:
        return JointState.from_state_tensor(
            torch.flip(self.joint_state.get_state_tensor(), dims=[0]),
            joint_names=self.joints,
            dof=len(self.joints),
        )

    @property
    def full_joint_state(self) -> JointState:
        return self.world.complete_joint_state(self.joint_state)

    def extract_joint_state(self, joints: List[str]) -> JointState:
        return extract_joint_state(self.joint_state, joints)

    def extract_trajectory(self, joints: List[str]) -> "Trajectory":
        joint_state = self.extract_joint_state(joints)
        return self.__class__(self.world, joint_state, metadata=self.metadata)

    def get_index(self, index: int) -> JointState:
        index = index % len(self)
        return self.joint_state[index : index + 1]

    def set_index(self, index: int, locked: bool = False, **kwargs: Any) -> None:
        self.world.set_joint_state(self.get_index(index), locked=locked, **kwargs)

    def iterator(self, **kwargs: Any) -> Iterator[None]:
        for index in range(len(self.joint_state)):
            self.set_index(index, **kwargs)
            yield

    def configuration(self, index: int, **kwargs: Any) -> "Configuration":
        # NVIDIA
        from schedulestream.applications.custream.state import Configuration

        return Configuration(self.world, self.get_index(index), metadata=self.metadata, **kwargs)

    def _decompose(self) -> Iterator["Configuration"]:
        for index in range(len(self.joint_state)):
            yield self.configuration(index)

    def reverse(self, state: Optional[State] = None) -> "Trajectory":
        if state is None:
            state = self.state
        return self.__class__(
            self.world, self.reversed_joint_state, state=state, metadata=self.metadata
        )

    def clone(self, state: Optional[State] = None) -> "Trajectory":
        if state is None:
            state = self.state
        return self.__class__(self.world, self.joint_state, state=state, metadata=self.metadata)

    @property
    def spheres(self) -> torch.Tensor:
        positions = self.full_joint_state.position
        link_state = self.world.get_link_state(confs=positions)
        indices = torch.cat([self.world.get_sphere_indices(link) for link in self.links])
        spheres = link_state.link_spheres_tensor[:, indices, :]

        return spheres

    def scene(self, scene: Optional[trimesh.Scene] = None) -> trimesh.Scene:
        if scene is None:
            scene = self.world.objects_scene()
        spheres_world = self.spheres.reshape(-1, self.spheres.shape[-1])
        return add_spheres(spheres_world, scene=scene)

    def show(self, **kwargs: Any) -> None:
        return self.scene().show(**kwargs)


class ArmPath(Command, Sequence):
    def __init__(self, world: World, arm: str, pose: Pose, **kwargs: Any):
        super().__init__(world, **kwargs)
        self.arm = arm
        self.pose = pose

    def __getitem__(self, index: int) -> Pose:
        return self.pose[index]

    def __len__(self) -> int:
        return len(self.pose)

    @property
    def link(self):
        return self.world.get_arm_link(self.arm)

    def reverse(self) -> "ArmPath":
        return self.__class__(
            self.world, self.arm, self.pose[::-1], state=self.state, metadata=self.metadata
        )

    @property
    def spheres(self) -> torch.Tensor:
        spheres = self.world.get_rigid_spheres(self.link, inactive=True)
        spheres = spheres[spheres[..., 3] > 0.0]
        return transform_spheres(self.pose, spheres)

    def scene(self, scene: Optional[trimesh.Scene] = None) -> trimesh.Scene:
        if scene is None:
            scene = self.world.objects_scene()
        spheres_world = self.spheres.reshape(-1, self.spheres.shape[-1])
        return add_spheres(spheres_world, scene=scene)
