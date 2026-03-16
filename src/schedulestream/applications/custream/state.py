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
from functools import cache, cached_property
from typing import Any, Dict, List, Optional

# Third Party
import numpy as np
from curobo.types.math import Pose

# NVIDIA
from schedulestream.applications.custream.command import Attach, Trajectory
from schedulestream.applications.custream.object import Object
from schedulestream.applications.custream.utils import (
    multiply_poses,
    pos_quat_from_pose,
    to_cpu,
    unit_pose,
)
from schedulestream.common.utils import Context, safe_zip


class Assignment(Context):
    def set(self) -> None:
        raise NotImplementedError()

    def __str__(self) -> str:
        prefix = self.__class__.__name__.lower()[0]
        return f"{prefix}{id(self) % 1000}"

    __repr__ = __str__


class Attachment(Assignment):
    def __init__(
        self, world: "World", obj: str, pose: Optional[Pose] = None, parent: Optional[str] = None
    ):
        self.world = world
        self.obj = obj
        self.parent = parent
        if pose is None:
            obj_pose = world.get_node_pose(obj)
            parent_pose = self.get_parent_pose()
            pose = multiply_poses(parent_pose.inverse(), obj_pose)
        self.pose = pose

    @property
    def nodes(self) -> List[str]:
        nodes = [self.obj]
        if self.parent is not None:
            nodes.append(self.parent)
        return nodes

    @property
    def objects(self) -> List[str]:
        return list(filter(self.world.is_object, self.nodes))

    def get_parent_pose(self) -> Pose:
        if self.parent is None:
            return unit_pose()
        if self.parent in self.world.tool_links:
            return self.world.get_link_pose(self.parent)
        return self.world.get_object_pose(self.parent)

    def get_pose(self) -> Pose:
        return multiply_poses(self.get_parent_pose(), self.pose)

    def set(self) -> None:
        self.world.set_object_pose(self.obj, self.get_pose())

    def command(self, **kwargs) -> Attach:
        return Attach(self.world, self.obj, self.parent, self.pose, **kwargs)

    def show(self, **kwargs: Any) -> None:
        self.set()
        self.world.show(**kwargs)

    def dump(self) -> None:
        position, quaternion = pos_quat_from_pose(self.pose)
        print(
            f"Object: {self.obj} | Parent: {self.parent} | Position: {np.round(position, 3)} |"
            f" Quaternion: {np.round(quaternion, 3)}"
        )

    def __str__(self):
        return f"{self.__class__.__name__}({self.obj}, parent={self.parent})"

    __repr__ = __str__


class Configuration(Trajectory, Assignment):
    @cached_property
    def joint_positions(self) -> Dict[str, float]:
        return dict(safe_zip(self.joints, to_cpu(self.positions[0])))

    def get_joint_position(self, joint: str) -> float:
        assert joint in self.joint_positions, (joint, self.joints)
        return self.joint_positions[joint]

    def get_joint_positions(self, joints: List[str]) -> List[float]:
        return list(map(self.get_joint_position, joints))

    def extract_configuration(self, joints: List[str]) -> "Configuration":
        joint_state = self.extract_joint_state(joints)
        return Configuration(self.world, joint_state)

    def set(self) -> None:
        self.world.set_joint_state(self.joint_state, locked=True)

    def rest_trajectory(self, steps: int, **kwargs: Any) -> Trajectory:
        joint_state = self.world.to_joint_state(
            confs=self.positions[:1].repeat(steps, 1), joints=self.joints
        )
        return self.__class__(self.world, joint_state, **kwargs)

    def dump(self, digits: int = 3) -> None:
        joint_positions = {
            joint: round(position, ndigits=digits)
            for joint, position in self.joint_positions.items()
        }
        print(f"Joints ({len(joint_positions)}): {joint_positions}")

    def __str__(self) -> str:
        prefix = "q"
        return f"{prefix}{id(self) % 1000}"

    __repr__ = __str__


class State(Assignment):
    def __init__(
        self,
        world: "World",
        conf: Optional[Configuration] = None,
        attachments: List[Attachment] = None,
    ):
        self.world = world
        self.conf = conf
        self.attachments = list(attachments or [])

    @property
    def joints(self) -> List[str]:
        if self.conf is None:
            return []
        return self.conf.joints

    @property
    def arms(self) -> List[str]:
        if self.conf is None:
            return []
        return self.conf.arms

    @property
    def objects(self) -> List[str]:
        return [attachment.obj for attachment in self.attachments]

    def get_object(self, name: str) -> Object:
        return self.world.get_object(name)

    @cache
    def arm_configuration(self, arm: str) -> Configuration:
        assert self.conf is not None
        joints = self.world.get_arm_joints(arm)
        return self.conf.extract_configuration(joints)

    def set(self) -> None:
        if self.conf is not None:
            self.conf.set()
        for attachment in self.attachments:
            if attachment.obj in self.world.attached_objects:
                current_attachment = self.world.attached_objects[attachment.obj]
                self.world.detach_objects(links=[current_attachment.parent])
            attachment.set()

    def show(self, **kwargs: Any) -> None:
        self.set()
        self.world.show(**kwargs)

    def dump(self):
        print(
            f"{self.__class__.__name__}) Joints: {len(self.joints)} | Objects: {len(self.objects)}"
        )
        self.conf.dump()
        print(f"Attachments ({len(self.attachments)}):")
        for attachment in self.attachments:
            attachment.dump()

    def __str__(self):
        return f"{self.__class__.__name__}(joints={self.joints}, objects={self.objects})"

    __repr__ = __str__
