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
"""Retime sparse joint-space waypoints into fixed-dt trajectories.

Two strategies for assigning per-segment durations before resampling at a fixed
control ``dt``:

  * :func:`retime_waypoints` -- per-joint velocity limits (cspace velocity).
  * :func:`retime_waypoints_pose` -- the arm tool frame's linear/angular velocity
    (so an SE(3) plan is timed by how fast the end-effector moves/rotates).
"""
from __future__ import annotations

# Standard Library
import math
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

# Third Party
import torch
from curobo.types import JointState, Pose

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.applications.custream2.world import World


_EPS = 1e-6


def slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Slerp ``q0`` to ``q1`` (``[..., 4]``) at ``alpha`` (broadcastable to ``[...]``).

    Quaternions are ``[w, x, y, z]``; the shorter arc is taken and near-parallel
    pairs fall back to normalized linear interpolation.
    """
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    a = alpha.unsqueeze(-1)
    small = sin_theta < _EPS
    c0 = torch.where(small, 1.0 - a, torch.sin((1.0 - a) * theta) / sin_theta)
    c1 = torch.where(small, a, torch.sin(a * theta) / sin_theta)
    quat = c0 * q0 + c1 * q1
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=_EPS)


def pose_deltas(
    positions_a: torch.Tensor,
    quats_a: torch.Tensor,
    positions_b: torch.Tensor,
    quats_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Elementwise translation distance [m] and geodesic rotation angle [rad] between paired poses.
    """
    linear = torch.norm(positions_a - positions_b, dim=-1)
    dot = (quats_a * quats_b).sum(dim=-1).abs().clamp(max=1.0)
    return linear, 2.0 * torch.acos(dot)


def lerp_poses(
    positions_a: torch.Tensor,
    quats_a: torch.Tensor,
    positions_b: torch.Tensor,
    quats_b: torch.Tensor,
    alpha: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolate paired poses at ``alpha``: linear on position, slerp on rotation.

    ``alpha`` broadcasts against the pose batch, so callers can evaluate a flat sequence of
    samples or a grid of ``[edges, steps]``.
    """
    pos = positions_a + alpha.unsqueeze(-1) * (positions_b - positions_a)
    return pos, slerp(quats_a, quats_b, alpha)


def interpolate_poses(
    positions: torch.Tensor,
    quats: torch.Tensor,
    seg: torch.Tensor,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample a pose polyline at ``num_samples`` points evenly spaced over cumulative ``seg``.

    ``seg`` is the per-segment weight -- SE(3) arc length or duration -- so callers choose the
    parameterization; positions interpolate linearly and quaternions slerp. Assumes ``seg`` sums
    above zero (callers guard the degenerate path).
    """
    device = positions.device
    cum = torch.cat([torch.zeros(1, device=device), seg.cumsum(dim=0)])
    total = float(cum[-1])
    t = torch.linspace(0.0, total, num_samples, device=device)
    idx = (torch.searchsorted(cum.contiguous(), t.contiguous(), right=True) - 1).clamp(
        0, len(seg) - 1
    )
    alpha = (t - cum[idx]) / (cum[idx + 1] - cum[idx]).clamp(min=_EPS)
    return lerp_poses(positions[idx], quats[idx], positions[idx + 1], quats[idx + 1], alpha)


def _interpolate_to_dt(
    positions: torch.Tensor,
    joint_names: List[str],
    seg_dur: torch.Tensor,
    dt: float,
) -> JointState:
    """Resample joint ``positions`` (with per-segment durations ``seg_dur``) to a fixed ``dt``."""
    device = positions.device
    cum_t = torch.cat([torch.zeros(1, device=device), seg_dur.cumsum(dim=0)])
    total = cum_t[-1].item()

    n_steps = max(2, int(torch.ceil(torch.tensor(total / dt)).item()) + 1)
    t = torch.linspace(0.0, total, n_steps, device=device)

    idx = torch.searchsorted(cum_t.contiguous(), t.contiguous(), right=True) - 1
    idx = idx.clamp(0, len(seg_dur) - 1)
    alpha = ((t - cum_t[idx]) / (cum_t[idx + 1] - cum_t[idx]).clamp(min=1e-8)).unsqueeze(1)
    interp = positions[idx] + alpha * (positions[idx + 1] - positions[idx])

    js = JointState.from_position(interp, joint_names=joint_names)
    js.dt = torch.tensor(dt, device=device)
    return js


def resample_waypoints(
    world: World,
    waypoints: Optional[JointState],
    dt: Optional[float] = None,
) -> Optional[JointState]:
    """Resample an already-timed trajectory onto a fixed ``dt`` grid, preserving its duration.

    Unlike :func:`retime_waypoints`, which re-derives segment durations from velocity limits and
    so discards the planner's velocity/acceleration profile, this keeps the planner's own timing:
    the input samples are uniform in time, so only their spacing changes. Needed for cuRobo, whose
    own interpolation emits an integer number of samples per optimizer knot -- see
    ``calculate_traj_steps`` -- which floors the waypoint count at the knot count.
    """
    if waypoints is None:
        return None
    if dt is None:
        dt = world.time_step
    positions = waypoints.position
    if positions.shape[0] < 2:
        return waypoints
    source_dt = waypoints.dt
    source_dt = dt if source_dt is None else float(source_dt.flatten()[0])
    seg_dur = torch.full((positions.shape[0] - 1,), source_dt, device=positions.device)
    return _interpolate_to_dt(positions, list(waypoints.joint_names), seg_dur, dt)


def retime_waypoints_batch(
    world: World,
    waypoints: JointState,
    dt: Optional[float] = None,
    velocity_scale: Optional[float] = None,
) -> List[JointState]:
    """Batched :func:`retime_waypoints` over ``waypoints`` with position ``[batch, n_poses, dof]``.

    Returns one fixed-``dt`` JointState per batch row (lengths vary per row).
    """
    if dt is None:
        dt = world.time_step
    if velocity_scale is None:
        velocity_scale = world.tamp_config.velocity_scale
    joint_names = list(waypoints.joint_names)
    positions = waypoints.position
    device = positions.device
    batch, n_poses, dof = positions.shape
    if n_poses < 2:
        positions = positions.expand(batch, 2, dof) if n_poses == 1 else positions
        n_poses = 2
    vel = world.get_velocity_limits(joint_names).to(device) * velocity_scale
    diffs = (positions[:, 1:] - positions[:, :-1]).abs()
    seg_dur = (diffs / vel).amax(dim=-1)
    seg_dur = torch.where(
        seg_dur.sum(dim=-1, keepdim=True) < dt,
        torch.full_like(seg_dur, dt / seg_dur.shape[-1]),
        seg_dur,
    )
    cum_t = torch.cat([torch.zeros(batch, 1, device=device), seg_dur.cumsum(dim=-1)], dim=-1)
    total = cum_t[:, -1]
    lengths = (torch.ceil(total / dt).long() + 1).clamp(min=2)
    ticks = torch.arange(int(lengths.max().item()), device=device).unsqueeze(0)
    t = (ticks / (lengths - 1).unsqueeze(1)).clamp(max=1.0) * total.unsqueeze(1)
    idx = torch.searchsorted(cum_t.contiguous(), t.contiguous(), right=True) - 1
    idx = idx.clamp(0, n_poses - 2)
    alpha = (
        (t - cum_t.gather(1, idx))
        / (cum_t.gather(1, idx + 1) - cum_t.gather(1, idx)).clamp(min=1e-8)
    ).unsqueeze(-1)
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, dof)
    p0 = positions.gather(1, gather_idx)
    p1 = positions.gather(1, gather_idx + 1)
    interp = p0 + alpha * (p1 - p0)

    dt_tensor = torch.tensor(dt, device=device)
    joint_states = []
    for length, row in zip(lengths.tolist(), interp):
        js = JointState.from_position(row[:length], joint_names=joint_names)
        js.dt = dt_tensor
        joint_states.append(js)
    return joint_states


def retime_waypoints(
    world: World,
    waypoints: Optional[JointState],
    dt: Optional[float] = None,
    velocity_scale: Optional[float] = None,
) -> Optional[JointState]:
    """Interpolate joint-state waypoints to a fixed ``dt`` using per-joint velocity limits.

    Returns ``None`` when ``waypoints`` is ``None`` (e.g. a failed plan).
    """
    if waypoints is None:
        return None
    joint_names = list(waypoints.joint_names)
    positions = waypoints.position
    if dt is None:
        dt = world.time_step
    if velocity_scale is None:
        velocity_scale = world.tamp_config.velocity_scale

    vel = world.get_velocity_limits(joint_names).to(positions.device) * velocity_scale
    diffs = (positions[1:] - positions[:-1]).abs()
    seg_dur = (diffs / vel.unsqueeze(0)).amax(dim=1)
    if float(seg_dur.sum()) < dt:
        seg_dur = torch.full_like(seg_dur, dt / seg_dur.shape[0])
    return _interpolate_to_dt(positions, joint_names, seg_dur, dt)


def pose_segment_durations(
    world: World,
    poses: Pose,
    position_velocity: Optional[float] = None,
    orientation_velocity: Optional[float] = None,
) -> torch.Tensor:
    """Per-segment durations ``[n_seg]`` along a batched Pose path: how long
    each segment takes with translation capped at ``position_velocity`` m/s and
    rotation at ``orientation_velocity`` rad/s (defaults from TAMPConfig)."""
    if position_velocity is None:
        position_velocity = world.tamp_config.position_velocity
    if orientation_velocity is None:
        orientation_velocity = world.tamp_config.orientation_velocity
    position = poses.position
    quaternion = poses.quaternion
    linear, angular = pose_deltas(position[:-1], quaternion[:-1], position[1:], quaternion[1:])
    return torch.maximum(linear / position_velocity, angular / orientation_velocity)


def retime_poses(
    world: World,
    poses: Optional[Pose],
    dt: Optional[float] = None,
    **kwargs: Any,
) -> Optional[Pose]:
    """Resample a sparse Pose path to a fixed ``dt`` using the pose velocity caps.

    The SE(3) analogue of :func:`retime_waypoints`: per-segment durations come
    from :func:`pose_segment_durations` (``**kwargs`` forwards there, e.g.
    ``position_velocity``/``orientation_velocity``), positions interpolate
    linearly and quaternions slerp. Returns ``None`` when ``poses`` is ``None``.
    """
    if poses is None:
        return None
    if dt is None:
        dt = world.time_step
    positions, quaternions = poses.position, poses.quaternion
    if positions.shape[0] < 2:
        return poses
    device = positions.device
    seg_dur = pose_segment_durations(world, poses, **kwargs)
    if float(seg_dur.sum()) < dt:
        seg_dur = torch.full_like(seg_dur, dt / seg_dur.shape[0])
    n_steps = max(2, int(math.ceil(float(seg_dur.sum()) / dt)) + 1)
    pos, quat = interpolate_poses(positions, quaternions, seg_dur, n_steps)
    return Pose(position=pos, quaternion=quat)


def retime_waypoints_pose(
    world: World,
    arm: str,
    waypoints: Optional[JointState],
    dt: Optional[float] = None,
    **kwargs: Any,
) -> Optional[JointState]:
    """Interpolate joint-space waypoints to a fixed ``dt`` using the tool frame's pose velocity.

    Like :func:`retime_waypoints`, but each segment's duration comes from how
    far ``arm``'s tool pose moves between waypoints, found via forward
    kinematics, rather than from per-joint velocity limits. ``**kwargs``
    forwards to :func:`pose_segment_durations` (e.g. ``position_velocity``/
    ``orientation_velocity``). Returns ``None`` when ``waypoints`` is ``None``.
    """
    if waypoints is None:
        return None
    joint_names = list(waypoints.joint_names)
    positions = waypoints.position
    if dt is None:
        dt = world.time_step
    device = positions.device

    tool_link = world.get_tool_link(arm)
    tool_poses = world.get_tool_poses(world.reorder_joint_state(waypoints))[tool_link]
    seg_dur = pose_segment_durations(world, tool_poses, **kwargs).to(device).clamp(min=dt)
    return _interpolate_to_dt(positions, joint_names, seg_dur, dt)
