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
import contextlib
import math
import random
from functools import partial
from typing import Annotated, Any, Iterable, Iterator, List, Literal, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
import trimesh
from curobo.types.math import Pose
from curobo.types.robot import JointState
from trimesh.transformations import (
    quaternion_about_axis,
    quaternion_from_euler,
    quaternion_from_matrix,
    quaternion_slerp,
)

# NVIDIA
from schedulestream.common.utils import get_pairs, remove_duplicates

Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]
Matrix = Annotated[np.dtype[np.floating[Any]], Literal["N", "N"]]
Color = Union[Vector, str]
Quaternion = Vector


PI = np.pi
INF = np.inf


def set_seed(**kwargs: Any) -> int:
    # NVIDIA
    from schedulestream.applications.trimesh2d.utils import set_seed

    seed = set_seed(**kwargs)
    torch.manual_seed(seed)
    return seed


@contextlib.contextmanager
def numpy_random_context(seed: Optional[int] = None):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def convex_combination(vector1: Vector, position2: Vector, weight: float) -> Vector:
    return (1 - weight) * np.array(vector1) + weight * np.array(position2)


def interpolate(vector1: Vector, vector2: Vector, num_steps: int) -> Iterator[Vector]:
    for weight in np.linspace(start=0.0, stop=1.0, num=num_steps, endpoint=True):
        yield convex_combination(vector1, vector2, weight)


def position_combination(position1: Vector, position2: Vector, weight: float) -> Vector:
    return convex_combination(position1, position2, weight)


def quaternion_combination(
    quaternion1: Quaternion, quaternion2: Quaternion, weight: float
) -> Quaternion:
    return quaternion_slerp(quaternion1, quaternion2, weight)


def pose_combination(pose1: Pose, pose2: Pose, weight: float, **kwargs: Any) -> Pose:
    device = pose1.position.device
    position1, quaternion1 = pos_quat_from_pose(pose1)
    position2, quaternion2 = pos_quat_from_pose(pose2)
    position = position_combination(position1, position2, weight)
    quaternion = quaternion_combination(quaternion1, quaternion2, weight)
    return Pose(
        position=torch.Tensor(np.array([position])).to(device),
        quaternion=torch.Tensor(np.array([quaternion])).to(device),
        **kwargs,
    )


def get_pose_distance(pose1: Pose, pose2: Pose, pos_step: float = 1e-2, ori_step: float = PI / 16):
    pos_distance, ori_distance = pose1.distance(pose2)
    return max(pos_distance / pos_step, ori_distance / ori_step)


def interpolate_poses(pose1: Pose, pose2: Pose, **kwargs: Any) -> Iterator[Pose]:
    pose_distance = get_pose_distance(pose1, pose2, **kwargs)
    num_steps = int(math.ceil(pose_distance)) + 1
    for weight in np.linspace(start=0.0, stop=1.0, num=num_steps, endpoint=True):
        yield pose_combination(pose1, pose2, weight)


def interpolate_pose_list(poses: List[Pose], **kwargs: Any) -> Iterator[Pose]:
    if not poses:
        return
    yield poses[0]
    for pose1, pose2 in get_pairs(poses):
        for i, pose in enumerate(interpolate_poses(pose1, pose2, **kwargs)):
            if i != 0:
                yield pose


def to_cpu(tensor: torch.Tensor) -> np.ndarray:
    if not isinstance(tensor, torch.Tensor):
        return np.array(tensor)
    return tensor.cpu().numpy()


def vector_norm(vector: Vector, norm: Optional[float] = None) -> float:
    return float(np.linalg.norm(vector, ord=norm))


vector_length = partial(vector_norm, norm=2)


def normalize_vector(vector: Vector) -> Vector:
    vector = np.array(vector)
    norm = vector_length(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def quat_from_euler(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> Quaternion:
    return quaternion_from_euler(roll, pitch, yaw)


def quat_from_axis(axis: Vector, angle: Optional[float] = None) -> Quaternion:
    if angle is None:
        angle = vector_length(axis)
    return quaternion_about_axis(angle, axis)


def quat_from_matrix(matrix: Matrix) -> Vector:
    return quaternion_from_matrix(matrix[:3, :3])


def pose_from_matrix(matrix: Matrix) -> Pose:
    pose = Pose.from_matrix(matrix)
    return pose


def matrix_from_pose(pose: Pose) -> Matrix:
    [matrix] = pose.get_numpy_matrix()
    return matrix


def pose_from_vector(vector: Vector, **kwargs: Any) -> Pose:
    pose = Pose.from_list(vector, q_xyzw=False, **kwargs)
    return pose


def vector_from_pose(pose: Pose) -> Vector:
    [vector] = pose.get_pose_vector()
    return vector


def to_pose(pose: Any, **kwargs: Any) -> Pose:
    if pose is None:
        return pose
    if isinstance(pose, Pose):
        return pose
    pose = np.array(pose)
    if pose.shape == (7,):
        return pose_from_vector(pose, **kwargs)
    if pose.shape == (4, 4):
        return pose_from_matrix(pose, **kwargs)
    raise NotImplementedError(pose)


def to_matrix(pose: Any) -> Matrix:
    return matrix_from_pose(to_pose(pose))


def to_vector(pose: Any) -> Matrix:
    return vector_from_pose(to_pose(pose))


def create_position(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector:
    return np.array([x, y, z])


create_quaternion = quat_from_euler


def pose_from_pos_quat(
    position: Optional[Vector] = None, quaternion: Optional[Vector] = None, **kwargs: Any
) -> Pose:
    if position is None:
        position = create_position()
    if quaternion is None:
        quaternion = create_quaternion()
    return to_pose(np.concatenate([position, quaternion]), **kwargs)


def rotation_from_pose(pose: Pose) -> Pose:
    device = pose.quaternion.device
    position = torch.zeros(*pose.quaternion.shape[:-1], 3).to(device)
    return Pose(position=position, quaternion=pose.quaternion)


def position_from_pose(pose: Pose) -> Vector:
    [position] = to_cpu(pose.position)
    return position


def quaternion_from_pose(pose: Pose) -> Vector:
    [quaternion] = to_cpu(pose.quaternion)
    return quaternion


def pos_quat_from_pose(pose: Pose) -> Tuple[Vector, Vector]:
    return position_from_pose(pose), quaternion_from_pose(pose)


def create_pose(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    **kwargs,
) -> Pose:
    position = create_position(x, y, z)
    quaternion = create_quaternion(roll, pitch, yaw)
    return pose_from_pos_quat(position, quaternion, **kwargs)


def unit_pose(**kwargs: Any) -> Pose:
    return create_pose(**kwargs)


def multiply_pair(pose1: Pose, pose2: Pose) -> Pose:
    if len(pose1) == len(pose2):
        return pose1.multiply(pose2)
    assert len(pose1) == 1 or len(pose2) == 1
    if len(pose2) > len(pose1):
        pose1 = pose1.repeat(len(pose2))
    if len(pose1) > len(pose2):
        pose2 = pose2.repeat(len(pose1))
    return pose1.multiply(pose2)


def multiply_poses(pose1: Pose, *poses: Pose) -> Pose:
    for pose2 in poses:
        pose1 = multiply_pair(pose1, pose2).clone()
    return pose1


def invert_pose(pose: Pose) -> Pose:
    return pose.inverse()


def multiply_matrices(matrix1: Matrix, *matrices: Matrix) -> Matrix:
    for matrix in matrices:
        matrix1 = matrix1 @ matrix
    return matrix1


def invert_matrix(matrix: Matrix) -> Matrix:
    return np.linalg.inv(matrix)


def transform_points(pose: Pose, points: torch.Tensor) -> torch.Tensor:
    assert pose.n_goalset == 1
    if not isinstance(points, torch.Tensor):
        points = torch.Tensor(np.array(points)).to(pose.position.device)
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    if points.shape[0] == 1:
        points = points.repeat(pose.batch, 1, 1)
    return pose.batch_transform_points(points)


def transform_spheres(pose: Pose, spheres: torch.Tensor) -> torch.Tensor:
    assert pose.n_goalset == 1
    if len(spheres.shape) == 2:
        spheres = spheres.unsqueeze(0)
    if spheres.shape[0] == 1:
        spheres = spheres.repeat(pose.batch, 1, 1)
    transformed_centers = transform_points(pose, spheres[:, :, :3])
    transformed_spheres = torch.cat([transformed_centers, spheres[:, :, 3:]], dim=2)
    return transformed_spheres


def concatenate_joint_states(joint_states: Iterable[JointState]) -> JointState:
    joint_states = list(joint_states)
    assert joint_states
    joint_names = joint_states[0].joint_names
    for joint_state in joint_states:
        assert joint_state.joint_names == joint_names
    return JointState.from_state_tensor(
        torch.cat([joint_state.get_state_tensor() for joint_state in joint_states], dim=-2),
        joint_names=joint_names,
        dof=joint_states[0].position.shape[-1],
    )


def extract_joint_state(
    joint_state: Optional[JointState], joints: List[str]
) -> Optional[JointState]:
    if joint_state is None:
        return joint_state
    indices = list(map(joint_state.joint_names.index, joints))
    return joint_state.index_dof(torch.as_tensor(indices, device=joint_state.tensor_args.device))


def merge_joint_states(joint_state1: JointState, joint_state2: JointState) -> JointState:
    assert len(joint_state1) == len(joint_state2), (len(joint_state1), len(joint_state2))
    new_joints = [
        joint for joint in joint_state2.joint_names if joint not in joint_state1.joint_names
    ]
    if not new_joints:
        return joint_state1.clone()
    new_state = extract_joint_state(joint_state2, new_joints)
    joints = remove_duplicates(joint_state1.joint_names + joint_state2.joint_names)
    return JointState.from_position(
        torch.cat([joint_state1.position, new_state.position], dim=-1), joint_names=joints
    )


def to_float_color(uinit8_color: np.ndarray) -> np.ndarray:
    return (np.array(uinit8_color).astype(np.float32) / (2**8 - 1)).tolist()


def create_frame(length: float = 0.1, pose: Optional[Pose] = None) -> trimesh.Trimesh:
    geometry = trimesh.creation.axis(
        origin_size=length * 0.1, axis_radius=length * 0.05, axis_length=length * 1.0
    )
    if pose is not None:
        matrix = matrix_from_pose(pose)
        geometry.apply_transform(matrix)
    return geometry


def create_poses(
    poses: List[Pose], scene: Optional[trimesh.Scene] = None, **kwargs: Any
) -> trimesh.Scene:
    if scene is None:
        scene = trimesh.Scene()
    for pose in poses:
        scene.add_geometry(create_frame(pose=pose, **kwargs))
    return scene


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    assert len(axis.shape) == len(angle.shape) == 2
    assert axis.shape[0] == angle.shape[0]
    assert axis.shape[1] == 3
    assert angle.shape[1] == 1
    return torch.cat([torch.cos(angle / 2), torch.sin(angle / 2) * axis], dim=1)


def sample_uniformly(lower: torch.Tensor, upper: torch.Tensor, num: int = 1) -> torch.Tensor:
    assert lower.shape == upper.shape
    extent = upper - lower
    dim = extent.shape[-1]
    difference = extent * torch.rand(size=(num, dim), device=lower.device, dtype=lower.dtype)
    return lower + difference


def select(
    sequence: Iterable[Any], reverse: bool = False, shuffle: bool = False, num: Optional[int] = None
) -> List[Any]:
    sequence = list(sequence)
    if reverse:
        sequence.reverse()
    if shuffle:
        random.shuffle(sequence)
    if (num is None) or (num >= len(sequence)):
        return sequence
    return sequence[:num]
