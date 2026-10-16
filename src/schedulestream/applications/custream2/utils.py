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
import contextlib
import gc
import math
import random
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult

# Third Party
import numpy as np
import numpy.typing as npt
import torch
import trimesh
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.state.state_joint_trajectory_ops import index_joint_state_dof
from curobo.types import JointState, Pose
from trimesh.transformations import (
    quaternion_about_axis,
    quaternion_from_euler,
    quaternion_from_matrix,
    quaternion_matrix,
    quaternion_slerp,
)

# NVIDIA
from schedulestream.common.utils import get_pairs, remove_duplicates

Vector = npt.NDArray[np.floating[Any]]
Matrix = npt.NDArray[np.floating[Any]]
Color = Union[Vector, str]
Quaternion = Vector

PI = np.pi
INF = np.inf
WORLD = None
GRAVITY = np.array([0.0, 0.0, -1.0])


def set_seed(**kwargs: Any) -> int:
    """Set the random seed for numpy and torch and return the seed value."""
    # NVIDIA
    from schedulestream.applications.trimesh2d.utils import set_seed

    seed = set_seed(**kwargs)
    torch.manual_seed(seed)
    return seed


@contextlib.contextmanager
def numpy_random_context(seed: Optional[int] = None):
    """Context manager that saves numpy random state, optionally seeds it, then restores."""
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def autograd_enabled():
    """Locally re-enable autograd for cuRobo calls made from an inference-mode context.

    Embedding hosts (e.g. IsaacLab data-generation loops) often run under
    ``torch.inference_mode()``, but cuRobo's IK / trajectory optimizers call ``backward()`` and
    therefore need autograd — otherwise they fail with "element 0 of tensors does not require grad
    and does not have a grad_fn". ``inference_mode(False)`` lifts inference mode; ``enable_grad``
    lifts no_grad.
    """
    with torch.inference_mode(False), torch.enable_grad():
        yield


def convex_combination(vector1: Vector, position2: Vector, weight: float) -> Vector:
    """Return the convex combination (1-weight)*v1 + weight*v2."""
    return (1 - weight) * np.array(vector1) + weight * np.array(position2)


def interpolate(vector1: Vector, vector2: Vector, num_steps: int) -> Iterator[Vector]:
    """Yield num_steps evenly-spaced interpolations from vector1 to vector2."""
    for weight in np.linspace(start=0.0, stop=1.0, num=num_steps, endpoint=True):
        yield convex_combination(vector1, vector2, weight)


def position_combination(position1: Vector, position2: Vector, weight: float) -> Vector:
    """Return the weighted average of two 3D positions."""
    return convex_combination(position1, position2, weight)


def quaternion_combination(
    quaternion1: Quaternion, quaternion2: Quaternion, weight: float
) -> Quaternion:
    """Return the spherical-linear interpolation between two quaternions."""
    return quaternion_slerp(quaternion1, quaternion2, weight)


def concatenate_poses(pose_list: List[Pose]) -> Pose:
    """Concatenate a list of Poses into a single batched Pose.

    Behaves like ``Pose.cat`` but iterates the input list once (collecting
    positions and quaternions together) and short-circuits when there is
    only one element.
    """
    if not pose_list:
        raise ValueError("concatenate_poses: empty pose list")
    if len(pose_list) == 1:
        p = pose_list[0]
        return Pose(position=p.position, quaternion=p.quaternion, normalize_rotation=False)
    positions = []
    quaternions = []
    for p in pose_list:
        positions.append(p.position)
        quaternions.append(p.quaternion)
    return Pose(
        position=torch.cat(positions),
        quaternion=torch.cat(quaternions),
        normalize_rotation=False,
    )


def pose_combination(pose1: Pose, pose2: Pose, weight: float, **kwargs: Any) -> Pose:
    """Return an interpolated Pose between pose1 and pose2 at the given weight."""
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
    """Return the number of interpolation steps needed to move from pose1 to pose2."""
    pos_distance, ori_distance = pose1.distance(pose2)
    return max(pos_distance / pos_step, ori_distance / ori_step)


def interpolate_poses(pose1: Pose, pose2: Pose, **kwargs: Any) -> Iterator[Pose]:
    """Yield a sequence of interpolated Poses stepping from pose1 to pose2."""
    pose_distance = get_pose_distance(pose1, pose2, **kwargs)
    num_steps = int(math.ceil(pose_distance)) + 1
    if num_steps == 1:
        yield pose1
    elif num_steps == 2:
        yield pose1
        yield pose2
    else:
        for weight in np.linspace(start=0.0, stop=1.0, num=num_steps, endpoint=True):
            yield pose_combination(pose1, pose2, weight)


def interpolate_pose_list(poses: List[Pose], **kwargs: Any) -> Iterator[Pose]:
    """Yield interpolated Poses along the entire sequence of input poses."""
    if not poses:
        return
    yield poses[0]
    for pose1, pose2 in get_pairs(poses):
        for i, pose in enumerate(interpolate_poses(pose1, pose2, **kwargs)):
            if i != 0:
                yield pose


def to_cpu(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a CPU numpy array (no-op for non-tensors)."""
    if not isinstance(tensor, torch.Tensor):
        return np.asarray(tensor)
    return tensor.detach().cpu().numpy()


def free_gpu_memory() -> None:
    """Drop Python references and release cached CUDA memory."""
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def vector_norm(vector: Vector, norm: Optional[float] = None) -> float:
    """Return the scalar norm of a vector."""
    return float(np.linalg.norm(vector, ord=norm))


vector_length = partial(vector_norm, norm=2)


def normalize_vector(vector: Vector) -> Vector:
    """Return a unit vector in the direction of the input, or the zero vector if it has zero norm.
    """
    vector = np.array(vector)
    norm = vector_length(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def angle_between(vector1: Vector, vector2: Vector) -> float:
    """Return the unsigned angle (radians) between two vectors in [0, pi]."""
    cos = float(np.dot(normalize_vector(vector1), normalize_vector(vector2)))
    return math.acos(max(-1.0, min(1.0, cos)))


def perpendicular_basis(axes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each unit vector in ``axes`` of shape ``(N, 3)``, return ``(e1, e2)``
    such that ``(axes, e1, e2)`` is a right-handed orthonormal basis.

    Uses world-up as the anchor for the first perpendicular axis, falling back
    to world-x when an axis is within ~8 degrees of vertical (where
    ``cross(up, axis)`` would degenerate).
    """
    up = np.where(
        np.abs(axes[:, 2:3]) < 0.99,
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    e1 = np.cross(up, axes)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(axes, e1)
    return e1, e2


def quat_from_euler(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> Quaternion:
    """Return a quaternion [w, x, y, z] from Euler roll-pitch-yaw angles."""
    return quaternion_from_euler(roll, pitch, yaw)


def quat_from_euler_batch(
    rolls: Optional[np.ndarray] = None,
    pitches: Optional[np.ndarray] = None,
    yaws: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a ``(N, 4)`` ``[w, x, y, z]`` quaternion array from same-length
    Euler-angle arrays. Matches :func:`quat_from_euler` (extrinsic XYZ via
    ``trimesh.transformations.quaternion_from_euler``) elementwise but
    vectorized — closed-form ``q = q_z(yaw) ⊗ q_y(pitch) ⊗ q_x(roll)``.

    Any of ``rolls`` / ``pitches`` / ``yaws`` may be omitted; missing axes
    default to a zero array sized to match whichever inputs are provided
    (or ``N=1`` when none are).
    """
    n = next((len(a) for a in (rolls, pitches, yaws) if a is not None), 1)
    zeros = np.zeros(n)
    rolls = zeros if rolls is None else rolls
    pitches = zeros if pitches is None else pitches
    yaws = zeros if yaws is None else yaws
    cr, sr = np.cos(rolls / 2.0), np.sin(rolls / 2.0)
    cp, sp = np.cos(pitches / 2.0), np.sin(pitches / 2.0)
    cy, sy = np.cos(yaws / 2.0), np.sin(yaws / 2.0)
    return np.stack(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ],
        axis=1,
    ).astype(np.float32)


def sample_yaw_quaternions(
    batch_size: int,
    yaw_interval: Tuple[float, float] = (-PI, +PI),
    yaw_steps: Optional[int] = None,
) -> np.ndarray:
    """Return a ``(batch_size, 4)`` array of ``[w, x, y, z]`` quaternions, each
    a yaw rotation about z sampled from ``yaw_interval``. When ``yaw_steps`` is
    set, draws from ``yaw_steps`` evenly-spaced yaws within the interval instead
    of uniformly."""
    yaw_lower, yaw_upper = yaw_interval
    if yaw_steps is None:
        yaws = np.random.uniform(yaw_lower, yaw_upper, batch_size)
    else:
        yaws = np.linspace(start=yaw_lower, stop=yaw_upper, num=yaw_steps, endpoint=False)
        yaws = np.random.choice(yaws, size=batch_size)
    return quat_from_euler_batch(yaws=yaws)


def quat_from_axis(axis: Vector, angle: Optional[float] = None) -> Quaternion:
    """Return a quaternion [w, x, y, z] for a rotation by angle around axis."""
    if angle is None:
        angle = vector_length(axis)
    return quaternion_about_axis(angle, axis)


def quat_from_matrix(matrix: Matrix) -> Vector:
    """Extract a quaternion [w, x, y, z] from the rotation part of a 4x4 matrix."""
    return quaternion_from_matrix(matrix[:3, :3])


def matrix_from_axis_angle(
    axis: npt.ArrayLike, angle: Optional[float] = None, position: Optional[npt.ArrayLike] = None
) -> Matrix:
    """Return a 4x4 float64 homogeneous matrix for a rotation about axis at position.

    If angle is None, it is taken from the length of axis (rotation-vector form).
    position sets the translation column (default origin). Stays in float64 (no
    curobo Pose round-trip, which is float32), so it is safe to feed to URDF
    exporters that decompose to roll-pitch-yaw near gimbal-lock orientations where
    a ~1e-7 perturbation would flip the result.
    """
    matrix = quaternion_matrix(quat_from_axis(np.asarray(axis, dtype=float), angle))
    if position is not None:
        matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def matrix_from_position(position: npt.ArrayLike) -> Matrix:
    """Return a 4x4 float64 homogeneous translation matrix (identity rotation)."""
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def pose_from_matrix(matrix: Matrix) -> Pose:
    """Convert a 4x4 homogeneous matrix to a Pose."""
    return Pose.from_matrix(matrix)


def iterate_poses(pose: Pose, batch_size: int = 1) -> Iterator[Pose]:
    """Yield slices of a batched ``Pose``, each of size up to ``batch_size``.

    With ``batch_size=1`` (default), yields each entry as a single-element
    ``Pose``. The trailing slice may be smaller when ``len(pose)`` is not a
    multiple of ``batch_size``.
    """
    for start in range(0, len(pose), batch_size):
        yield pose[start : start + batch_size]


def pose_from_axes(
    x_axis: Optional[np.ndarray] = None,
    y_axis: Optional[np.ndarray] = None,
    z_axis: Optional[np.ndarray] = None,
    position: Optional[np.ndarray] = None,
) -> Pose:
    """Build a batched ``Pose`` from per-row basis axes and translations.

    Each provided input is shape ``(N, 3)``. Omitted axes default to the
    corresponding world basis ``(1,0,0)/(0,1,0)/(0,0,1)`` repeated to ``N``;
    omitted ``position`` defaults to zeros. At least one argument must be
    provided so the batch size ``N`` can be inferred. Stacks the axes as
    columns of an ``(N, 4, 4)`` homogeneous transform and routes through
    :func:`pose_from_matrix` for the batched quaternion conversion.
    """
    n = next((len(a) for a in (x_axis, y_axis, z_axis, position) if a is not None), 1)
    T = np.tile(np.eye(4), (n, 1, 1))
    if x_axis is not None:
        T[:, :3, 0] = x_axis
    if y_axis is not None:
        T[:, :3, 1] = y_axis
    if z_axis is not None:
        T[:, :3, 2] = z_axis
    if position is not None:
        T[:, :3, 3] = position
    return pose_from_matrix(T)


def matrix_from_pose(pose: Pose) -> Matrix:
    """Convert a Pose to a 4x4 numpy homogeneous matrix."""
    [matrix] = pose.get_numpy_matrix()
    return matrix


def pose_from_vector(vector: Vector, **kwargs: Any) -> Pose:
    """Build a Pose from a 7-element [x, y, z, qw, qx, qy, qz] vector."""
    return Pose.from_list(vector, q_xyzw=False, **kwargs)


def vector_from_pose(pose: Pose) -> Vector:
    """Return a 7-element [x, y, z, qw, qx, qy, qz] array from a Pose."""
    [vector] = pose.get_pose_vector()
    return vector


def to_pose(pose: Any, **kwargs: Any) -> Pose:
    """Coerce a matrix, list, array, or existing Pose into a Pose object."""
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
    """Convert any pose-like value to a 4x4 numpy homogeneous matrix."""
    return matrix_from_pose(to_pose(pose))


def batch_pose(pose: Pose, batch: int = 1) -> Pose:
    """Return ``pose`` reshaped to 2D and broadcast to ``batch`` copies if it has a single entry."""
    pos = pose.position if pose.position.ndim == 2 else pose.position.unsqueeze(0)
    quat = pose.quaternion if pose.quaternion.ndim == 2 else pose.quaternion.unsqueeze(0)
    if pos.shape[0] == 1 and batch > 1:
        pos = pos.expand(batch, -1)
        quat = quat.expand(batch, -1)
    return Pose(position=pos, quaternion=quat)


def to_vector(pose: Any) -> Vector:
    """Convert any pose-like value to a 7-element [x, y, z, qw, qx, qy, qz] array."""
    return vector_from_pose(to_pose(pose))


def create_position(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector:
    """Return a 3D position vector from x, y, z components."""
    return np.array([x, y, z])


create_quaternion = quat_from_euler


def pose_from_pos_quat(
    position: Optional[Vector] = None, quaternion: Optional[Vector] = None, **kwargs: Any
) -> Pose:
    """Build a Pose from an optional position and quaternion, defaulting to identity."""
    if position is None:
        position = create_position()
    if quaternion is None:
        quaternion = create_quaternion()
    return to_pose(np.concatenate([position, quaternion]), **kwargs)


def rotation_from_pose(pose: Pose) -> Pose:
    """Return a Pose with zero translation and the same orientation as pose."""
    device = pose.quaternion.device
    position = torch.zeros(*pose.quaternion.shape[:-1], 3).to(device)
    return Pose(position=position, quaternion=pose.quaternion)


def position_from_pose(pose: Pose) -> Vector:
    """Extract the 3D translation vector from a Pose."""
    [position] = to_cpu(pose.position)
    return position


def quaternion_from_pose(pose: Pose) -> Vector:
    """Extract the quaternion [w, x, y, z] from a Pose."""
    [quaternion] = to_cpu(pose.quaternion)
    return quaternion


def pos_quat_from_pose(pose: Pose) -> Tuple[Vector, Vector]:
    """Return (position, quaternion) extracted from a Pose."""
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
    """Create a Pose from position (x, y, z) and Euler angles (roll, pitch, yaw)."""
    position = create_position(x, y, z)
    quaternion = create_quaternion(roll, pitch, yaw)
    return pose_from_pos_quat(position, quaternion, **kwargs)


def pose_from_axis_angle(
    axis: Vector,
    angle: Optional[float] = None,
    position: Optional[Vector] = None,
    **kwargs: Any,
) -> Pose:
    """Create a Pose rotated by angle (radians) about axis at position (default origin).

    If angle is None, it is taken from the length of axis (rotation-vector form).
    """
    return pose_from_pos_quat(position, quat_from_axis(axis, angle), **kwargs)


def unit_pose(**kwargs: Any) -> Pose:
    """Return an identity Pose (zero translation, no rotation)."""
    return create_pose(**kwargs)


def multiply_pair(pose1: Pose, pose2: Pose) -> Pose:
    """Multiply two Poses, broadcasting one to match the other's batch size if needed."""
    if len(pose1) == len(pose2):
        return pose1.multiply(pose2)
    assert len(pose1) == 1 or len(pose2) == 1
    if len(pose2) > len(pose1):
        pose1 = pose1.repeat(len(pose2))
    if len(pose1) > len(pose2):
        pose2 = pose2.repeat(len(pose1))
    return pose1.multiply(pose2)


def multiply_poses(pose1: Pose, *poses: Pose) -> Pose:
    """Compose a chain of Poses left-to-right and return the result."""
    for pose2 in poses:
        pose1 = multiply_pair(pose1, pose2).clone()
    return pose1


def invert_pose(pose: Pose) -> Pose:
    """Return the inverse of a Pose."""
    return pose.inverse()


def multiply_matrices(matrix1: Matrix, *matrices: Matrix) -> Matrix:
    """Compose a chain of 4x4 matrices left-to-right and return the result."""
    for matrix in matrices:
        matrix1 = matrix1 @ matrix
    return matrix1


def invert_matrix(matrix: Matrix) -> Matrix:
    """Return the inverse of a 4x4 homogeneous matrix."""
    return np.linalg.inv(matrix)


def transform_points(pose: Pose, points: torch.Tensor) -> torch.Tensor:
    """Apply pose to a batch of 3D points and return the transformed points."""
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(np.array(points), dtype=torch.float32)
    points = points.to(pose.position.device)
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    if points.shape[0] == 1:
        points = points.repeat(pose.batch_size, 1, 1)
    return pose.batch_transform_points(points)


def transform_spheres(pose: Pose, spheres: torch.Tensor) -> torch.Tensor:
    """Apply pose to a batch of spheres [N, S, 4] and return transformed [x, y, z, r] tensors."""
    spheres = spheres.to(pose.position.device)
    if len(spheres.shape) == 2:
        spheres = spheres.unsqueeze(0)
    if spheres.shape[0] == 1:
        spheres = spheres.repeat(pose.batch_size, 1, 1)
    transformed_centers = transform_points(pose, spheres[:, :, :3].contiguous())
    transformed_spheres = torch.cat([transformed_centers, spheres[:, :, 3:]], dim=2)
    return transformed_spheres


def zero_joint_state_derivatives(joint_state: JointState) -> JointState:
    """Populate velocity/acceleration/jerk with zeros matching the position."""
    joint_state.velocity = torch.zeros_like(joint_state.position)
    joint_state.acceleration = torch.zeros_like(joint_state.position)
    joint_state.jerk = torch.zeros_like(joint_state.position)
    return joint_state


def concatenate_joint_states(joint_states: Iterable[JointState]) -> JointState:
    """Stack multiple same-joint JointStates into a batched [N, n_joints] JointState."""
    joint_states = list(joint_states)
    assert joint_states, "No joint states to concatenate"
    joint_names = list(joint_states[0].joint_names)
    positions = torch.cat([js.position.reshape(-1, len(joint_names)) for js in joint_states], dim=0)
    return JointState.from_position(positions, joint_names=joint_names)


def stack_joint_states(joint_states: List[JointState], dim: int = 1) -> JointState:
    """Stack same-joint [batch, dof] JointStates along a new dim (default: [batch, steps, dof])."""
    assert joint_states, "No joint states to stack"
    positions = torch.stack([js.position for js in joint_states], dim=dim)
    return JointState.from_position(positions, joint_names=list(joint_states[0].joint_names))


def interpolate_joint_states(
    confs: List[JointState], num_steps: int, dt: Optional[float] = None
) -> JointState:
    """Linearly interpolate between consecutive IK confs for all batch elements simultaneously.

    ``dt`` (when given) is stored as a 1-element tensor so it survives
    batch indexing of the returned state.
    """
    device = confs[0].position.device
    segments = []
    for i in range(len(confs) - 1):
        p1 = confs[i].position
        p2 = confs[i + 1].position
        weights = torch.linspace(0.0, 1.0, num_steps, device=device).view(1, num_steps, 1)
        segments.append(p1.unsqueeze(1) + weights * (p2 - p1).unsqueeze(1))
    segments.append(confs[-1].position.unsqueeze(1))
    pos = torch.cat(segments, dim=1)
    joint_state = JointState.from_position(pos, joint_names=list(confs[0].joint_names))
    if dt is not None:
        joint_state.dt = torch.tensor([dt], dtype=torch.float32, device=device)
    return joint_state


def extract_joint_state(
    joint_state: Optional[JointState], joints: List[str]
) -> Optional[JointState]:
    """Return a JointState containing only the specified joints, or None if input is None.

    When ``joints`` already matches the input's joint names, the input is returned
    as-is (not a copy) — treat the result as read-only.
    """
    if joint_state is None:
        return joint_state
    if list(joints) == list(joint_state.joint_names):
        return joint_state
    indices = torch.as_tensor(
        list(map(joint_state.joint_names.index, joints)), device=joint_state.position.device
    )
    result = index_joint_state_dof(joint_state, indices)
    result.dt = joint_state.dt
    return result


def expand_joint_state(js: JointState, n: int) -> JointState:
    """Broadcast a single-batch JointState to n copies."""
    pos = js.position
    if pos.shape[0] == 1:
        pos = pos.expand(n, -1).clone()
    new_js = JointState.from_position(pos, joint_names=js.joint_names)
    new_js.jerk = None
    new_js.velocity = None
    new_js.acceleration = None
    return new_js


def merge_joint_states(joint_state1: JointState, joint_state2: JointState) -> JointState:
    """Merge two JointStates with potentially different joints into one.

    Joints already present in joint_state1 are taken from joint_state1; any
    additional joints found only in joint_state2 are appended.
    """
    new_joints = [j for j in joint_state2.joint_names if j not in joint_state1.joint_names]
    if not new_joints:
        return joint_state1.clone()
    extra = extract_joint_state(joint_state2, new_joints)
    all_joints = remove_duplicates(list(joint_state1.joint_names) + list(joint_state2.joint_names))
    merged_pos = torch.cat([joint_state1.position, extra.position], dim=-1)
    return JointState.from_position(merged_pos, joint_names=all_joints)


def to_float_color(uint8_color: np.ndarray) -> np.ndarray:
    """Convert an uint8 RGBA array to a float list in [0, 1]."""
    return (np.array(uint8_color).astype(np.float32) / (2**8 - 1)).tolist()


def create_frame(length: float = 0.1, pose: Optional[Pose] = None) -> trimesh.Trimesh:
    """Create a trimesh axis-frame geometry, optionally transformed by pose."""
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
    """Add axis-frame markers for each pose to a trimesh scene and return it."""
    if scene is None:
        scene = trimesh.Scene()
    for pose in poses:
        scene.add_geometry(create_frame(pose=pose, **kwargs))
    return scene


def quat_from_axis_angle(axis: torch.Tensor, angle: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return quaternions [batch, 4] in [w, x, y, z] from batched axis [batch, 3].

    If angle [batch, 1] is None, the per-row rotation angle is taken from the
    magnitude of axis (rotation-vector form). axis is normalized either way; a
    zero-magnitude row yields the identity quaternion.
    """
    assert len(axis.shape) == 2 and axis.shape[1] == 3
    norm = axis.norm(dim=1, keepdim=True)
    if angle is None:
        angle = norm
    assert len(angle.shape) == 2 and angle.shape[0] == axis.shape[0] and angle.shape[1] == 1
    unit_axis = axis / torch.clamp(norm, min=1e-12)
    return torch.cat([torch.cos(angle / 2), torch.sin(angle / 2) * unit_axis], dim=1)


def sample_se2(
    world: Any,
    position_interval: Tuple[Vector, Vector] = (np.zeros(3), np.zeros(3)),
    axis: Tuple[float, ...] = (0.0, 0.0, 1.0),
    angle_interval: Tuple[float, float] = (-PI, PI),
    angle_discretization: Optional[int] = None,
    batch_size: int = 1,
) -> "Pose":
    """Sample batch_size SE(2) poses with uniform position and yaw in angle_interval."""
    position_lower = world.to_device(np.array(position_interval[0]))
    position_upper = world.to_device(np.array(position_interval[1]))
    position = sample_uniformly(position_lower, position_upper, num=batch_size)

    ax = world.to_device(np.array(axis)).repeat(batch_size, 1)
    if angle_discretization is None:
        lo = world.to_device(np.expand_dims([angle_interval[0]], axis=-1))
        hi = world.to_device(np.expand_dims([angle_interval[1]], axis=-1))
        yaw = sample_uniformly(lo, hi, num=batch_size)
    else:
        angles = np.linspace(
            start=angle_interval[0],
            stop=angle_interval[1],
            num=angle_discretization,
            endpoint=False,
        )
        yaw = world.to_device(np.random.choice(angles, size=batch_size)).unsqueeze(1)
    quaternion = quat_from_axis_angle(ax, yaw)
    return Pose(position=position, quaternion=quaternion)


def compute_joint_distance(joint_state: JointState) -> float:
    """Return the total joint-space path length (sum of per-step L2 norms) of a trajectory."""
    if joint_state is None or joint_state.position.shape[0] == 0:
        return 0.0
    diffs = joint_state.position[:-1] - joint_state.position[1:]
    return float(torch.norm(diffs, dim=1).sum().item())


def compute_linear_duration(world: Any, joint_state: JointState) -> float:
    """Estimate trajectory duration in seconds using per-joint velocity limits."""
    if joint_state is None or joint_state.position.shape[0] <= 1:
        return 0.0
    joints = list(joint_state.joint_names)
    device = joint_state.position.device
    vel = world.get_velocity_limits(joints).to(device)
    diffs = (joint_state.position[:-1] - joint_state.position[1:]).abs()
    durations = (diffs / vel.unsqueeze(0)).amax(dim=1)
    return float(durations.sum().item())


def sample_uniformly(lower: torch.Tensor, upper: torch.Tensor, num: int = 1) -> torch.Tensor:
    """Sample num points uniformly at random within the axis-aligned box [lower, upper]."""
    assert lower.shape == upper.shape
    extent = upper - lower
    dim = extent.shape[-1]
    difference = extent * torch.rand(size=(num, dim), device=lower.device, dtype=lower.dtype)
    return lower + difference


def select(
    sequence: Iterable[Any], reverse: bool = False, shuffle: bool = False, num: Optional[int] = None
) -> List[Any]:
    """Return up to num items from sequence, optionally reversed or shuffled."""
    sequence = list(sequence)
    if reverse:
        sequence.reverse()
    if shuffle:
        random.shuffle(sequence)
    if (num is None) or (num >= len(sequence)):
        return sequence
    return sequence[:num]


def simplify_mesh(
    mesh: trimesh.Trimesh,
    convexify: bool = False,
    repair: bool = False,
    merge_digits: Optional[int] = None,
    max_faces: Optional[int] = None,
) -> trimesh.Trimesh:
    """Clean, optionally convexify, and decimate ``mesh``.

    Operates on a copy — the caller's mesh is left untouched. Dedupes
    coincident vertices and duplicate triangles, drops orphan vertices,
    reorients face normals to point outward (cuRobo's mesh SDF kernel is
    winding-sensitive), and fills small holes so the mesh is watertight
    enough for ``compute_local_sdf`` to sign points correctly. When
    ``convexify`` is set, replaces the mesh with its convex hull. When
    ``max_faces`` is set and the face count exceeds it, applies quadric
    decimation and re-runs ``fix_normals`` (decimation can flip normals on
    thin features). When ``merge_digits`` is None, the vertex-merge step
    is skipped. The result is repainted a uniform color equal to the input
    mesh's average color, since convexification/decimation discard the source
    per-vertex/face colors.
    """
    mesh = mesh.copy()
    try:
        avg_color = mesh.visual.main_color
    except Exception:
        avg_color = None
    if convexify:
        mesh = mesh.convex_hull
    if merge_digits is not None:
        mesh.merge_vertices(digits_vertex=merge_digits)
    if max_faces is not None and len(mesh.faces) > max_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
        trimesh.repair.fix_normals(mesh)
    if repair:
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
    if avg_color is not None:
        mesh.visual.vertex_colors = np.tile(avg_color, (len(mesh.vertices), 1))
    return mesh


def add_spheres(
    spheres: torch.Tensor,
    scene: Optional[trimesh.Scene] = None,
    color: tuple = (0, 128, 255, 60),
    parent_node_name: Optional[str] = None,
    min_radius: float = 0.0,
) -> trimesh.Scene:
    """Add icosphere meshes for each non-zero-radius sphere to the trimesh scene."""
    if scene is None:
        scene = trimesh.Scene()
    if spheres.numel() == 0:
        return scene
    data = spheres.reshape(-1, 4).cpu().numpy()
    for _, (x, y, z, r) in enumerate(data):
        if r <= min_radius:
            continue
        mesh = trimesh.creation.icosphere(radius=float(r), subdivisions=2)
        mesh.visual = trimesh.visual.color.ColorVisuals(mesh=mesh, face_colors=color)
        tf = np.eye(4)
        tf[:3, 3] = [x, y, z]
        scene.add_geometry(mesh, transform=tf, parent_node_name=parent_node_name)
    return scene


def get_optimized_plan(result: Optional[TrajOptSolverResult]) -> Optional[JointState]:
    """Extract the optimizer's knot trajectory, carrying its own knot dt, from a TrajOptSolverResult.

    Reads ``js_solution`` rather than ``interpolated_trajectory``: cuRobo's interpolation emits a
    whole number of samples per knot (``calculate_traj_steps``), which both coarsens the plan and
    misreports its duration -- for one Franka plan, 22 samples labelled 1.05 s standing in for 81
    knots spanning 0.72 s. Callers resample to their own dt (see :func:`retime.resample_waypoints`).

    Among successful (batch, seed) pairs, picks the shortest duration; with a fixed knot count that
    is the smallest dt.
    """
    if result is None:
        return None
    js: Optional[JointState] = result.js_solution
    if js is None:
        return None
    success: Optional[torch.Tensor] = result.success
    if success is None or not success.any():
        return None

    pos: torch.Tensor = js.position
    dt: Optional[torch.Tensor] = js.dt
    if pos.ndim == 4:
        flat_success = success.to(torch.bool).reshape(-1)
        durations = (
            dt.reshape(-1).to(torch.float32)
            if dt is not None and dt.numel() == flat_success.numel()
            else torch.zeros_like(flat_success, dtype=torch.float32)
        )
        masked = torch.where(flat_success, durations, torch.full_like(durations, float("inf")))
        flat_idx = int(masked.argmin().item())
        b, seed = flat_idx // pos.shape[1], flat_idx % pos.shape[1]
        sel_pos, sel_dt = pos[b, seed], (None if dt is None else dt.reshape(-1)[flat_idx])
    else:
        sel_pos, sel_dt = pos.reshape(-1, pos.shape[-1]), (
            None if dt is None else dt.reshape(-1)[0]
        )

    out: JointState = JointState.from_position(sel_pos, joint_names=js.joint_names)
    out.dt = sel_dt
    return out


def resample_polyline(points: torch.Tensor, num_samples: int, p: float = 2.0) -> torch.Tensor:
    """Resample a ``[P, d]`` polyline into ``num_samples`` arc-length-spaced points."""
    min_segment_length = 1e-8
    if points.shape[0] <= 1:
        return points
    seg = torch.norm(points[1:] - points[:-1], p=p, dim=-1)
    cum = torch.cat([torch.zeros(1, device=points.device, dtype=points.dtype), seg.cumsum(dim=0)])
    total = cum[-1].item()
    if total <= 0:
        return points[:1]
    t = torch.linspace(0.0, total, num_samples, device=points.device, dtype=points.dtype)
    idx = (torch.searchsorted(cum.contiguous(), t.contiguous(), right=True) - 1).clamp(
        0, len(seg) - 1
    )
    t0, t1 = cum[idx], cum[idx + 1]
    alpha = ((t - t0) / (t1 - t0).clamp(min=min_segment_length)).unsqueeze(1)
    return points[idx] + alpha * (points[idx + 1] - points[idx])


def scene_sphere_contacts(
    scene_coll: Any,
    spheres: torch.Tensor,
    collision_distance: float = 0.0,
    activation_distance: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sphere penetration depth and push-out direction against ``scene_coll``
    (a cuRobo scene collision checker); see :meth:`World.get_sphere_contacts`.

    Accepts spheres as ``[batch, horizon, S, 4]``, ``[batch, S, 4]``, or
    ``[S, 4]`` — the lower-rank forms are unsqueezed to match the cuRobo API
    and the added dims are squeezed back on return. ``collision_distance``
    inflates the sphere radii before the query; ``activation_distance`` is
    cuRobo's eta (a smoothed cost ramp of the same width).
    """
    device = scene_coll.device_cfg.device
    spheres = spheres.to(device)
    squeeze_batch = spheres.ndim == 2
    if squeeze_batch:
        spheres = spheres.unsqueeze(0)
    squeeze_horizon = spheres.ndim == 3
    if squeeze_horizon:
        spheres = spheres.unsqueeze(1)
    if collision_distance:
        spheres = spheres.clone()
        spheres[..., 3] = spheres[..., 3] + float(collision_distance)
    buffer = CollisionBuffer.from_shape(spheres.shape, scene_coll.device_cfg)
    weight = torch.ones(1, device=device, dtype=torch.float32)
    eta = torch.full((1,), float(activation_distance), device=device, dtype=torch.float32)
    distances = scene_coll.get_sphere_distance_raw(spheres, buffer, weight, eta)
    directions = -buffer.gradient[..., :3]
    if squeeze_horizon:
        distances = distances.squeeze(1)
        directions = directions.squeeze(1)
    if squeeze_batch:
        distances = distances.squeeze(0)
        directions = directions.squeeze(0)
    return distances, directions


def patch_v2_config(robot_config: dict, tool_links: List[str]) -> None:
    """Adapt a v1-style robot config dict for curobov2's MotionPlannerCfg."""
    kinematics = robot_config["robot_cfg"]["kinematics"]
    kinematics["tool_frames"] = tool_links
    for key in ("use_usd_kinematics", "ee_link", "link_names"):
        kinematics.pop(key, None)
    cspace = kinematics.get("cspace")
    if isinstance(cspace, dict) and "retract_config" in cspace:
        cspace["default_joint_position"] = cspace.pop("retract_config")
