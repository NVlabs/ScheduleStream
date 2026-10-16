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
from typing import TYPE_CHECKING, Any, Iterator, List, Optional

# Third Party
import torch
from curobo.types import JointState, Pose

# NVIDIA
from schedulestream.applications.custream2.command import (
    Trajectory,
    get_placement_parent,
)
from schedulestream.applications.custream2.utils import (
    WORLD,
    extract_joint_state,
    multiply_poses,
    to_cpu,
    unit_pose,
)
from schedulestream.common.utils import instance_cache

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.applications.custream2.placement import Placement
    from schedulestream.applications.custream2.world import World


class Attachment:
    """Base class for object attachments: stores parent_from_obj pose."""

    def __init__(
        self,
        world: World,
        obj: str,
        parent: Optional[str] = None,
        pose: Optional[Pose] = None,
        label: Optional[str] = None,
    ) -> None:
        self.world = world
        self.obj = obj
        self.parent = parent
        if pose is None:
            obj_pose = world.get_object(obj).get_pose()
            parent_pose = self.get_parent_pose()
            pose = multiply_poses(parent_pose.inverse(), obj_pose)
        self.pose = pose
        self.label = label

    @property
    def objects(self) -> List[str]:
        """Return object names involved in this attachment (obj and parent, if any)."""
        objs = [self.obj]
        if self.parent is not None:
            objs.append(self.parent)
        return objs

    @property
    def body(self):
        """Return the Object corresponding to self.obj."""
        return self.world.get_object(self.obj)

    def get_parent_pose(self) -> Pose:
        """Return the world pose of the parent: EE link pose or object pose."""
        if self.parent is None:
            return unit_pose()
        if self.parent in self.world.tool_frames:
            return self.world.get_link_pose(self.parent)
        return self.world.get_object(self.parent).get_pose()

    def get_pose(self) -> Pose:
        """Return the object's absolute world pose by composing parent and local poses."""
        return multiply_poses(self.get_parent_pose(), self.pose)

    def set(self, **kwargs: Any) -> None:
        """Apply this attachment's world pose to the object in the world."""
        self.world.set_object_pose(self.obj, self.get_pose(), **kwargs)

    def __str__(self) -> str:
        prefix = self.__class__.__name__[0].lower()
        return f"{prefix}{id(self) % 1000}"

    __repr__ = __str__
    __hash__ = object.__hash__
    __eq__ = object.__eq__


class Configuration:
    """Opaque token representing an arm joint configuration, optionally relative to the reference
    frame of a ``placement``, like a Trajectory's."""

    def __init__(
        self,
        world: Any,
        joint_state: JointState,
        grasps: Optional[List[Any]] = None,
        placement: Optional[Placement] = WORLD,
        label: Optional[str] = None,
    ) -> None:
        self.world = world
        self.joint_state = joint_state
        self.grasps = list(grasps or [])
        self.label = label
        self.placement = placement
        self.metadata: dict = {}
        pos = joint_state.position
        if pos.ndim != 2 or pos.shape[0] != 1:
            raise ValueError(
                "Configuration expects a 1D joint state [1, n_joints], got shape"
                f" {tuple(pos.shape)}"
            )

    @property
    def parent(self) -> Optional[str]:
        """The body of the reference placement, or WORLD."""
        return get_placement_parent(self.placement)

    @property
    def holding(self) -> List[str]:
        """Return the names of the objects held by this configuration's grasps."""
        return [grasp.obj for grasp in self.grasps]

    @property
    def joints(self) -> List[str]:
        """Return the ordered list of joint names in this configuration."""
        return list(self.joint_state.joint_names)

    @property
    def active_joints(self) -> List[str]:
        """Return the subset of joint names that are active in the world."""
        active = set(self.world.active_joints)
        return [j for j in self.joints if j in active]

    @property
    def locked_joints(self) -> List[str]:
        """Return the subset of joint names that are locked in the world."""
        locked = set(self.world.locked_joints)
        return [j for j in self.joints if j in locked]

    @property
    def arm_joints(self) -> List[str]:
        """Return the subset of joint names that belong to any arm in this configuration."""
        all_arm = {j for a in self.world.arms for j in self.world.get_arm_joints(a)}
        return [j for j in self.joints if j in all_arm]

    @property
    def arms(self) -> List[str]:
        """Return the arm identifiers covered by this configuration's joint names."""
        js = set(self.joints)
        return [a for a in self.world.arms if set(self.world.get_arm_joints(a)) & js]

    @property
    def position(self) -> torch.Tensor:
        """Return joint positions as a 1D tensor."""
        return self.joint_state.position[0]

    def set(self) -> None:
        """Apply this configuration's joint positions to the world."""
        self.world.set_joint_positions(self.joints, to_cpu(self.position))

    def rest_trajectory(self, steps: int) -> Trajectory:
        """Return a static Trajectory that holds this configuration for the given number of steps.
        """
        pos = self.joint_state.position.expand(steps, -1).clone()
        js = JointState.from_position(pos, joint_names=self.joints)
        js.dt = torch.tensor(self.world.time_step, device=pos.device)
        return Trajectory(self.world, js)

    def extract_joint_state(self, joints: Optional[List[str]] = None) -> JointState:
        """Return this configuration's joint state restricted to ``joints``
        (default: the world's active joints present in this configuration)."""
        if joints is None:
            joints = self.world.active_joints
        joints = list(filter(set(self.joints).__contains__, joints))
        return extract_joint_state(self.joint_state, joints)

    def extract_configuration(
        self, joints: Optional[List[str]] = None, **kwargs: Any
    ) -> Configuration:
        """Return a new Configuration containing only the specified joints."""
        return Configuration(self.world, self.extract_joint_state(joints), **kwargs)

    def dump(self, digits: int = 3) -> None:
        """Print joint positions for debugging."""
        joint_positions = {
            j: round(float(p), digits) for j, p in zip(self.joints, to_cpu(self.position))
        }
        print(f"Joints ({len(joint_positions)}): {joint_positions}")

    def execute(self, **kwargs: Any) -> Iterator[None]:
        """Apply this configuration to the world and yield once."""
        self.set()
        yield

    def __str__(self) -> str:
        return f"q{id(self) % 1000}"

    __repr__ = __str__
    __hash__ = object.__hash__
    __eq__ = object.__eq__


class State:
    """Snapshot of arm configuration + object placements for TAMP."""

    def __init__(
        self,
        world: Any,
        conf: Optional[Configuration] = None,
        attachments: Optional[List[Any]] = None,
    ) -> None:
        self.world = world
        self.conf = conf
        self.attachments = list(attachments or [])

    @property
    def joints(self) -> List[str]:
        """Joint names from the stored configuration, or empty if none."""
        if self.conf is None:
            return []
        return self.conf.joints

    @property
    def arms(self) -> List[str]:
        """Arm identifiers from the stored configuration, or empty if none."""
        if self.conf is None:
            return []
        return self.conf.arms

    @property
    def objects(self) -> List[str]:
        """Names of all attached objects."""
        return [p.obj for p in self.attachments]

    @property
    def grasps(self) -> List[Any]:
        """Attachments that are Grasp instances."""
        # NVIDIA
        from schedulestream.applications.custream2.grasp import Grasp

        return [a for a in self.attachments if isinstance(a, Grasp)]

    @property
    def placements(self) -> List[Any]:
        """Attachments that are Placement instances."""
        # NVIDIA
        from schedulestream.applications.custream2.placement import Placement

        return [a for a in self.attachments if isinstance(a, Placement)]

    def configuration(self, joints: List[str], **kwargs: Any) -> Optional[Configuration]:
        """Extract a Configuration restricted to the given joint names."""
        if self.conf is None:
            return None
        return self.conf.extract_configuration(joints, **kwargs)

    @instance_cache
    def arm_configuration(self, arm: str, **kwargs: Any) -> Optional[Configuration]:
        """Return the cached Configuration for the given arm."""
        return self.configuration(self.world.get_arm_joints(arm), **kwargs)

    def set(self, **kwargs: Any) -> None:
        """Apply the stored configuration and all attachment poses to the world."""
        if self.conf is not None:
            self.conf.set()
        self.world.attached_objects.clear()
        for attachment in self.attachments:
            attachment.set(**kwargs)
        for grasp in self.grasps:
            self.world.attach_object(grasp.obj, grasp.parent)

    def __enter__(self) -> "State":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.set()

    def show(self, **kwargs: Any) -> None:
        """Apply this state to the world and open the visualizer."""
        self.set()
        self.world.show(**kwargs)

    def dump(self) -> None:
        """Print a summary of joint and object counts."""
        print(f"State) Joints: {len(self.joints)} | Objects: {len(self.objects)}")
        if self.conf is not None:
            self.conf.dump()

    def __str__(self) -> str:
        return f"State(arms={self.arms}, objects={self.objects})"

    __repr__ = __str__
