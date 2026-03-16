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
import math
from typing import Any, Iterator, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
from curobo.types.math import Pose
from curobo.types.robot import JointState

# NVIDIA
from schedulestream.applications.custream.object import GraspConfig, Object
from schedulestream.applications.custream.spheres import add_spheres
from schedulestream.applications.custream.state import Attachment
from schedulestream.applications.custream.utils import (
    PI,
    create_pose,
    interpolate,
    multiply_poses,
    to_cpu,
    transform_spheres,
)
from schedulestream.applications.custream.world import Conf, World
from schedulestream.applications.trimesh2d.samplers import sample_interval
from schedulestream.common.utils import INF, randomize, safe_zip, take


class Grasp(Attachment):
    def __init__(
        self,
        world: "World",
        obj: str,
        pose: Pose,
        link: str,
        joint_state: Optional[JointState] = None,
        **kwargs: Any,
    ):
        super().__init__(world, obj, pose, parent=link, **kwargs)
        self.joint_state = joint_state

    @property
    def body(self) -> Object:
        return self.world.get_object(self.obj)

    @property
    def spheres(self) -> torch.Tensor:
        return self.body.get_spheres_tensor()

    def get_approach_pose(self, link_distance: float = 0.0, object_distance: float = 0.0) -> Pose:
        return multiply_poses(
            create_pose(z=link_distance), self.pose, create_pose(z=object_distance).inverse()
        )

    def get_object_pose(self, link_pose: Pose, **kwargs: Any) -> Pose:
        pose = self.get_approach_pose(**kwargs)
        return multiply_poses(link_pose, pose)

    def get_link_pose(self, obj_pose: Pose, **kwargs: Any) -> Pose:
        pose = self.get_approach_pose(**kwargs)
        return multiply_poses(obj_pose, pose.inverse())

    def attach(self) -> None:
        self.world.attach_object(self.obj, self.parent, grasp_pose=self.pose, attach=True)

    def detach(self) -> None:
        self.world.detach_objects(links=[self.parent], detach=True)

    def __str__(self):
        prefix = self.__class__.__name__[0].lower()
        return f"{prefix}{id(self) % 1000}"

    __repr__ = __str__


PITCH_INTERVALS = {
    "top": (np.pi, np.pi),
    "side": (np.pi / 2, np.pi / 2),
    "upper": (np.pi - np.pi / 2, np.pi),
    "all": (0.0, 2 * np.pi),
}


def get_pitch_interval(
    pitch_interval: Union[str, Tuple[float, float]] = "upper"
) -> Tuple[float, float]:
    if not isinstance(pitch_interval, str):
        return pitch_interval
    if pitch_interval not in PITCH_INTERVALS:
        raise ValueError(pitch_interval)
    return PITCH_INTERVALS[pitch_interval]


def sphere_grasp_generator(
    dimensions: np.ndarray, pitch_interval: Tuple[float, float]
) -> Iterator[Pose]:
    while True:
        pitch = np.random.uniform(*pitch_interval)
        yaw = np.random.uniform(low=0, high=2 * np.pi)
        yield create_pose(pitch=pitch, yaw=yaw)


def cylinder_grasp_generator(
    dimensions: np.ndarray, pitch_interval: Tuple[float, float], top_offset: float = INF
) -> Iterator[Pose]:
    assert top_offset >= 0.0
    height = dimensions[2]
    z = max(0.0, height / 2 - top_offset)
    while True:
        pitch = np.random.uniform(*pitch_interval)
        yaw = np.random.uniform(low=0, high=2 * np.pi)
        yield create_pose(z=z, pitch=pitch, yaw=yaw)


capsule_grasp_generator = cylinder_grasp_generator


def cuboid_grasp_generator(
    dimensions: np.ndarray,
    pitch_interval: Tuple[float, float],
    top_offset: float = INF,
    num_faces: int = 4,
) -> Iterator[Pose]:
    assert top_offset >= 0.0
    height = dimensions[2]
    z = max(0.0, height / 2 - top_offset)
    for pitch in sample_interval(*pitch_interval):
        for yaw in np.linspace(start=0, stop=2 * np.pi, endpoint=False, num=num_faces):
            yield multiply_poses(
                create_pose(pitch=pitch),
                create_pose(yaw=yaw),
                create_pose(z=-z),
            )


def primitive_grasp_generator(
    primitive: str,
    dimensions: np.ndarray,
    pitch_interval: Union[str, Tuple[float, float]],
    **kwargs: Any,
) -> Iterator[Pose]:
    pitch_interval = get_pitch_interval(pitch_interval)
    if primitive == "sphere":
        yield from sphere_grasp_generator(dimensions, pitch_interval)
    elif primitive == "cylinder":
        yield from cylinder_grasp_generator(dimensions, pitch_interval, **kwargs)
    elif primitive == "capsule":
        yield from capsule_grasp_generator(dimensions, pitch_interval, **kwargs)
    elif primitive == "cuboid":
        yield from cuboid_grasp_generator(dimensions, pitch_interval, **kwargs)
    else:
        raise ValueError(primitive)


def contact_grasp_generator(world: World, obj: Object, arm: str) -> Iterator[Grasp]:
    link = world.get_arm_link(arm)
    bounding_box = obj.bounding_box
    yaws = np.linspace(start=0.0, stop=2 * PI, num=4, endpoint=False)
    radii = [
        bounding_box.width / 2,
        bounding_box.depth / 2,
        bounding_box.width / 2,
        bounding_box.depth / 2,
    ]
    for yaw, radius in randomize(safe_zip(yaws, radii)):
        distance = radius + 4e-2
        grasp_pose = multiply_poses(
            create_pose(yaw=yaw + PI),
            create_pose(pitch=PI),
            create_pose(x=distance * math.cos(yaw), y=distance * math.sin(yaw)),
        )
        grasp_pose = multiply_poses(bounding_box.pose, grasp_pose)
        yield Grasp(world, obj.name, grasp_pose, link)


def handover_grasp_generator(
    world: World, obj: Object, arm: str, offset: float = 2e-2
) -> Iterator[Grasp]:
    link = world.get_arm_link(arm)
    bounding_box = obj.bounding_box
    yaws = np.linspace(start=0.0, stop=2 * PI, num=4, endpoint=False)
    radii = [
        bounding_box.width / 2,
        bounding_box.depth / 2,
        bounding_box.width / 2,
        bounding_box.depth / 2,
    ]
    z = max(0.0, bounding_box.height / 2 - offset)
    for i, (yaw, radius) in randomize(enumerate(safe_zip(yaws, radii))):
        orthogonal = radii[(i + 1) % len(radii)]
        if 2 * orthogonal > 0.08:
            continue
        distance = radius - offset
        grasp_pose = multiply_poses(
            create_pose(yaw=yaw + PI),
            create_pose(pitch=PI),
            create_pose(x=distance * math.cos(yaw), y=distance * math.sin(yaw), z=-z),
        )
        grasp_pose = multiply_poses(bounding_box.pose, grasp_pose)
        yield Grasp(world, obj.name, grasp_pose, link)


def grasp_generator(
    world: World, obj: Union[str, Object], arm: Optional[str] = None, **kwargs: Any
) -> Iterator[Grasp]:
    if arm is None:
        arm = world.arms[0]
    link = world.get_arm_link(arm)

    obj = world.get_object(obj)
    grasp_config = obj.grasp_config
    if grasp_config is None:
        grasp_config = GraspConfig()
    primitive = grasp_config.primitive
    if primitive is None:
        primitive = obj.primitive
    assert primitive is not None

    bounding_box = obj.bounding_box
    dimensions = bounding_box.dimensions
    grasp_poses = primitive_grasp_generator(
        primitive,
        dimensions,
        grasp_config.pitch_interval,
        top_offset=grasp_config.top_offset,
        **kwargs,
    )
    for grasp_pose in take(grasp_poses, n=grasp_config.max_grasps):
        grasp_pose = multiply_poses(bounding_box.pose, grasp_pose)
        yield Grasp(world, obj.name, grasp_pose, link)
        if grasp_config.reverse:
            grasp_pose = multiply_poses(grasp_pose, create_pose(roll=PI))
            yield Grasp(world, obj.name, grasp_pose, link)


def gripper_step_sizes(world: World, arm: str, num_steps: int = 20) -> np.ndarray:
    gripper_joints = world.get_gripper_joints(arm)
    closed_conf, opened_conf = world.get_gripper_limits(gripper_joints)
    distances = np.absolute(np.array(opened_conf) - np.array(closed_conf))
    step_sizes = distances / num_steps
    return step_sizes


def get_gripper_steps(
    world: World, arm: str, start_conf: Conf, end_conf: Conf, **kwargs: Any
) -> int:
    distances = np.absolute(np.array(end_conf) - np.array(start_conf))
    step_sizes = gripper_step_sizes(world, arm, **kwargs)
    num_steps = math.ceil(np.max(np.divide(distances, step_sizes))) + 1
    return int(num_steps)


def interpolate_gripper(
    world: World,
    arm: str,
    end_conf: Conf,
    start_conf: Optional[Conf] = None,
    num_steps: Optional[int] = None,
    **kwargs: Any,
):
    gripper_joints = world.get_gripper_joints(arm)
    if start_conf is None:
        start_conf = world.get_joint_positions(gripper_joints)
    if num_steps is None:
        num_steps = get_gripper_steps(world, arm, start_conf, end_conf, **kwargs)
    path = list(interpolate(start_conf, end_conf, num_steps=num_steps))
    return world.to_joint_state(joints=gripper_joints, confs=path)


def open_gripper(
    world: World,
    arm: Optional[str] = None,
    **kwargs: Any,
) -> JointState:
    if arm is None:
        arm = world.default_arm
    gripper_joints = world.get_gripper_joints(arm)
    _, opened_conf = world.get_gripper_limits(gripper_joints)
    return interpolate_gripper(world, arm, opened_conf, **kwargs)


def normalized_gripper_positions(world: World, arm: Optional[str] = None):
    if arm is None:
        arm = world.default_arm
    gripper_joints = world.get_gripper_joints(arm)
    current_conf = world.get_joint_positions(gripper_joints)
    closed_conf, opened_conf = map(np.array, world.get_gripper_limits(gripper_joints))
    return (current_conf - closed_conf) / (opened_conf - closed_conf)


def close_until_contact(
    world: World,
    arm: Optional[str] = None,
    link_pose: Optional[Pose] = None,
    start_conf: Optional[Conf] = None,
    debug: bool = False,
    verbose: bool = False,
    **kwargs,
) -> JointState:
    verbose |= debug
    if arm is None:
        arm = world.arms[0]
    gripper_joints = world.get_gripper_joints(arm)
    closed_conf, _ = world.get_gripper_limits(gripper_joints)
    if start_conf is None:
        start_conf = world.get_joint_positions(gripper_joints)
    close_path = to_cpu(
        interpolate_gripper(
            world, arm, start_conf=start_conf, end_conf=closed_conf, **kwargs
        ).position
    )
    num_steps = len(close_path)

    tool_link = world.get_arm_link(arm)
    if link_pose is None:
        link_pose = world.get_link_pose(tool_link)
    joint_state = {}
    max_step = 0
    for i, gripper_joint in enumerate(gripper_joints):
        gripper_link = world.yourdf.get_joint_child(gripper_joint)
        spheres = world.get_rigid_spheres(gripper_link)
        poses = []
        with world.yourdf.state(joints=gripper_joints):
            for conf in close_path:
                world.yourdf.set_joint_position(gripper_joint, conf[i])
                poses.append(world.get_node_pose(gripper_link, parent_node=tool_link))
        spheres = transform_spheres(multiply_poses(link_pose, Pose.cat(poses)), spheres)
        distances = world.get_sphere_distances(spheres)
        collisions = distances > 0.0
        indices = sorted(to_cpu(torch.nonzero(torch.any(collisions, dim=-1))[:, 0]))
        step = indices[0] if indices else len(close_path) - 1
        max_step = max(max_step, step)
        position = close_path[step][i]
        joint_state[gripper_joint] = position
        world.yourdf.set_joint_position(gripper_joint, position)
        if verbose:
            print(
                f"Joint: {gripper_joint} | Step: {step}/{len(close_path)} | Position:"
                f" {position:.3f}"
            )
        if debug:
            add_spheres(spheres[: step + 1], scene=world.scene.copy()).show()
    contact_positions = list(map(joint_state.get, gripper_joints))
    contact_path = list(interpolate(start_conf, contact_positions, num_steps=max_step + 1))
    world.yourdf.set_joint_positions(gripper_joints, start_conf)
    return world.to_joint_state(joints=gripper_joints, confs=contact_path)
