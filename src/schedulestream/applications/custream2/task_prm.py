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
"""GPU-batched PRM motion planner over SE(3) tool poses for custream2.

A pose-space sibling of :mod:`joint_prm`. Instead of planning in joint space,
this planner:

  1. Randomly samples collision-free joint configurations (checked in joint
     space, exactly like :mod:`joint_prm`).
  2. Runs forward kinematics to map each sample to its arm tool-frame pose,
     giving SE(3) roadmap vertices (start and goal tool poses are nodes 0/1).
  3. Connects nearest-neighbour vertices and checks each edge by interpolating
     in SE(3) (linear position + quaternion slerp) and collision-checking ONLY
     the links/spheres rigidly attached to the tool frame (the hand body), which
     are transformed rigidly with the interpolated pose -- no IK per edge.
  4. Runs Dijkstra, then locally shortcuts the SE(3) pose path.
  5. Grounds the final pose path into joint space with IK: first enforcing
     collisions and continuity (seeding each solve from the previous solution);
     if that fails it relaxes both; waypoints with no IK solution are skipped.
     IK success statistics are printed at the end.

Used as the ``--motion task`` option inside ``motion_stream``.
"""
from __future__ import annotations

# Standard Library
from typing import Any, Optional, Tuple

# Third Party
import torch
from curobo.types import JointState, Pose

# NVIDIA
from schedulestream.applications.custream2.joint_prm import (
    EPSILON,
    _build_graph,
    _conf_costs,
    _path_weight,
    _shortest_path,
)
from schedulestream.applications.custream2.retime import (
    interpolate_poses,
    lerp_poses,
    pose_deltas,
)
from schedulestream.common.utils import INF, current_time, elapsed_time, profiler


def _pose_distances(
    positions_a: torch.Tensor,
    quats_a: torch.Tensor,
    positions_b: torch.Tensor,
    quats_b: torch.Tensor,
    orientation_weight: float,
) -> torch.Tensor:
    """Elementwise SE(3) distance ``||dp|| + orientation_weight * angle`` between paired poses."""
    linear, angular = pose_deltas(positions_a, quats_a, positions_b, quats_b)
    return linear + orientation_weight * angular


def _pose_distance_matrix(
    positions: torch.Tensor, quats: torch.Tensor, orientation_weight: float
) -> torch.Tensor:
    """Pairwise SE(3) distance ``||dp|| + orientation_weight * angle`` -> ``[M, M]``.

    The gram matrix is accumulated in float64: ``acos`` has an unbounded derivative at ``dot = 1``
    (1/sqrt(1 - dot^2)), so the ~1e-7 difference between an sgemm and an elementwise dot becomes a
    first-figure error in the angle for the near-identical poses that densified paths are made of --
    measured 4x on consecutive samples, and it inflated every short edge's rotation cost.
    """
    pos_d = torch.cdist(positions, positions)
    dot = (quats.double() @ quats.double().t()).abs().clamp(max=1.0)
    angle = 2.0 * torch.acos(dot).to(positions.dtype)
    return pos_d + orientation_weight * angle


def _pose_path_length(
    positions: torch.Tensor, quats: torch.Tensor, orientation_weight: float
) -> float:
    """Total SE(3) length along a pose polyline (0 for < 2 poses)."""
    if positions.shape[0] < 2:
        return 0.0
    seg = _pose_distances(positions[:-1], quats[:-1], positions[1:], quats[1:], orientation_weight)
    return float(seg.sum())


def _resample_poses(
    positions: torch.Tensor,
    quats: torch.Tensor,
    num_samples: int,
    orientation_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample a pose polyline to ``num_samples`` evenly spaced (by SE(3) arc-length) poses.

    Spacing uses the same ``||dp|| + orientation_weight * angle`` length as
    :func:`_pose_path_length`, so an in-place reorientation is not collapsed to a single sample.
    """
    num = positions.shape[0]
    if num < 2:
        return positions, quats
    seg = _pose_distances(positions[:-1], quats[:-1], positions[1:], quats[1:], orientation_weight)
    if float(seg.sum()) <= EPSILON:
        return positions, quats
    return interpolate_poses(positions, quats, seg, num_samples)


def _pose_costs(
    world: Any,
    arm: str,
    poses: Pose,
    activation_distance: float = 1e-2,
    collision_distance: float = 0.0,
    max_batch: int = 2**14,
    **kwargs: Any,
) -> torch.Tensor:
    """Per-pose collision cost for ``arm``'s rigid hand spheres at every ``poses``.

    Any sphere within ``collision_distance`` of the scene costs INF; otherwise the cost is the
    mean intrusion past ``activation_distance``. The two margins are queried separately: cuRobo's
    inflated-sphere depth is not a clipped distance, so deriving the hard test from the soft query
    mis-rejects clear configs at wider margins.
    Queries run in chunks of ``max_batch`` poses to bound peak GPU memory:
    the edge sweep submits every edge interior at once (~300k poses at 30k
    edges), which OOMs when the process shares the GPU with Isaac.
    kwargs forward to :meth:`World.get_hand_spheres`.
    """
    num = poses.position.shape[0]
    if num == 0:
        return torch.zeros(0, device=poses.position.device)
    costs = []
    for start in range(0, num, max_batch):
        chunk = Pose(
            position=poses.position[start : start + max_batch],
            quaternion=poses.quaternion[start : start + max_batch],
        )
        spheres = world.get_hand_spheres(arm, chunk, inactive=False, **kwargs)
        colliding = (
            world.get_sphere_distances(spheres, collision_distance=collision_distance) > 0.0
        ).any(dim=-1)
        penetration = world.get_sphere_distances(spheres, collision_distance=activation_distance)
        active = (spheres[..., 3] > 0.0).sum(dim=-1).clamp(min=1)
        costs.append(torch.where(colliding, INF, penetration.sum(dim=-1) / active))
    return torch.cat(costs)


def _edge_costs_se3(
    world: Any,
    arm: str,
    positions: torch.Tensor,
    quats: torch.Tensor,
    ei: torch.Tensor,
    ej: torch.Tensor,
    num_steps: int,
    orientation_weight: float,
    activation_distance: float = 1e-2,
    collision_distance: float = 0.0,
    **kwargs: Any,
) -> torch.Tensor:
    """Mean :func:`_pose_costs` over the edge's SE(3) interior.

    Averaged rather than summed so the scale is a depth independent of ``num_steps``, then scaled
    by the edge's SE(3) distance so a long edge grazing over the same fraction of its length costs
    more than a short one: inf when any interpolated pose collides (an infinite weight deletes the
    edge in ``_build_graph``)."""
    if ei.numel() == 0:
        return torch.zeros(0, device=positions.device)
    weights = torch.linspace(0.0, 1.0, num_steps + 2, device=positions.device)[1:-1]
    pos, quat = lerp_poses(
        positions[ei].unsqueeze(1),
        quats[ei].unsqueeze(1),
        positions[ej].unsqueeze(1),
        quats[ej].unsqueeze(1),
        weights.view(1, -1),
    )
    interp = Pose(position=pos.reshape(-1, 3), quaternion=quat.reshape(-1, 4))
    costs = _pose_costs(
        world,
        arm,
        interp,
        activation_distance=activation_distance,
        collision_distance=collision_distance,
        **kwargs,
    )
    mean = costs.reshape(ei.shape[0], num_steps).mean(dim=1)
    length = _pose_distances(positions[ei], quats[ei], positions[ej], quats[ej], orientation_weight)
    return torch.where(torch.isfinite(mean), mean * length, mean)


def plan_tool_roadmap(
    world: Any,
    arm: str,
    start_pose: Pose,
    goal_pose: Pose,
    num_confs: int = 10000,
    degree: int = 10,
    num_steps: int = 10,
    sample_factor: int = 8,
    orientation_weight: Optional[float] = None,
    start_goal_edge: bool = False,
    collision_weight: float = 1e6,
    activation_distance: float = 1e-2,
    collision_distance: float = 0.0,
    verbose: bool = True,
    **kwargs: Any,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Build an SE(3) tool-pose roadmap and return the pose path ``(positions, quats)`` or ``None``.

    ``start_pose``/``goal_pose`` are the arm's tool poses at the path endpoints.
    ``orientation_weight`` defaults to ``TAMPConfig.orientation_weight``.
    """
    t0 = current_time()
    if orientation_weight is None:
        orientation_weight = world.tamp_config.orientation_weight
    device = world.device
    tool_link = world.get_tool_link(arm)
    arm_joints = world.get_arm_joints(arm)

    sampled = world.sample_joint_states(
        num_confs * sample_factor, joints=arm_joints, collisions=True, verbose=False
    )
    sampled = JointState.from_position(
        sampled.position[:num_confs], joint_names=list(sampled.joint_names)
    )
    tool_poses = world.get_tool_poses(sampled)[tool_link]

    positions = torch.cat([start_pose.position, goal_pose.position, tool_poses.position], dim=0).to(
        device
    )
    quats = torch.cat(
        [start_pose.quaternion, goal_pose.quaternion, tool_poses.quaternion], dim=0
    ).to(device)
    num_nodes = positions.shape[0]
    if num_nodes < 2:
        if verbose:
            print(
                f"{plan_tool_roadmap.__name__}) No collision-free samples | Elapsed:"
                f" {elapsed_time(t0):.3f} sec"
            )
        return None

    dmat = _pose_distance_matrix(positions, quats, orientation_weight)
    masked = dmat.clone()
    masked.fill_diagonal_(INF)
    k = min(degree, num_nodes - 1)
    knn = torch.topk(masked, k=k, largest=False).indices

    rows = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
    cols = knn.reshape(-1)
    lo = torch.minimum(rows, cols)
    hi = torch.maximum(rows, cols)
    codes = lo * num_nodes + hi
    if start_goal_edge:
        codes = torch.cat([codes, torch.tensor([1], device=device, dtype=codes.dtype)])
    codes = torch.unique(codes)
    ei = torch.div(codes, num_nodes, rounding_mode="floor")
    ej = codes % num_nodes

    edge_costs = _edge_costs_se3(
        world,
        arm,
        positions,
        quats,
        ei,
        ej,
        num_steps,
        orientation_weight,
        activation_distance=activation_distance,
        collision_distance=collision_distance,
        **kwargs,
    )
    weights = dmat[ei, ej]
    if collision_weight:
        weights = weights + collision_weight * edge_costs
    graph = _build_graph(num_nodes, ei, ej, weights=weights)
    path = _shortest_path(graph, 0, 1)

    if verbose:
        cost = _path_weight(graph, path)
        length = (
            _pose_path_length(positions[path], quats[path], orientation_weight)
            if path is not None
            else float("inf")
        )
        start_degree, goal_degree = int(graph.indptr[1] - graph.indptr[0]), int(
            graph.indptr[2] - graph.indptr[1]
        )
        print(
            f"{plan_tool_roadmap.__name__}) Nodes: {num_nodes} | Edges:"
            f" {int(torch.isfinite(edge_costs).sum())}/{ei.shape[0]} | Start/Goal degree:"
            f" {start_degree}/{goal_degree} | Solved: {path is not None} | Waypoints:"
            f" {0 if path is None else len(path)} | Length: {length:.3f} | Cost: {cost:.3f} |"
            f" Elapsed: {elapsed_time(t0):.3f} sec"
        )
    if path is None:
        return None
    return positions[path], quats[path]


def shortcut_tool_path(
    world: Any,
    arm: str,
    positions: torch.Tensor,
    quats: torch.Tensor,
    num_samples: int = 100,
    num_steps: int = 10,
    orientation_weight: Optional[float] = None,
    edge_fraction: float = 1.0,
    collision_weight: float = 1e6,
    activation_distance: float = 1e-2,
    collision_distance: float = 0.0,
    verbose: bool = False,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Densify the SE(3) path and reconnect non-adjacent poses to remove detours."""
    t0 = current_time()
    if orientation_weight is None:
        orientation_weight = world.tamp_config.orientation_weight
    if positions.shape[0] < 3:
        return positions, quats
    rp, rq = _resample_poses(positions, quats, num_samples, orientation_weight)
    num_nodes = rp.shape[0]
    if num_nodes < 3:
        return positions, quats

    pairs = torch.triu_indices(num_nodes, num_nodes, offset=1, device=rp.device)
    ei, ej = pairs[0], pairs[1]
    dmat = _pose_distance_matrix(rp, rq, orientation_weight)
    if edge_fraction < 1.0:
        keep = max(1, int(edge_fraction * ei.shape[0]))
        order = torch.topk(dmat[ei, ej], k=keep, largest=False).indices
        ei, ej = ei[order], ej[order]
    edge_costs = _edge_costs_se3(
        world,
        arm,
        rp,
        rq,
        ei,
        ej,
        num_steps,
        orientation_weight,
        activation_distance=activation_distance,
        collision_distance=collision_distance,
        **kwargs,
    )
    weights = dmat[ei, ej]
    if collision_weight:
        weights = weights + collision_weight * edge_costs
    graph = _build_graph(num_nodes, ei, ej, weights=weights)
    path = _shortest_path(graph, 0, num_nodes - 1)
    if path is None:
        return positions, quats

    new_p, new_q = rp[path], rq[path]
    old_len = _pose_path_length(positions, quats, orientation_weight)
    new_len = _pose_path_length(new_p, new_q, orientation_weight)
    if verbose:
        print(
            f"{shortcut_tool_path.__name__}) Samples: {num_nodes} | {old_len:.3f} ->"
            f" {new_len:.3f} | Cost: {_path_weight(graph, path):.3f} | Elapsed:"
            f" {elapsed_time(t0):.3f} sec"
        )
    if new_len >= old_len - EPSILON:
        return positions, quats
    return new_p, new_q


def plan_task_path(
    world: Any,
    arm: str,
    goal_js: JointState,
    start_js: Optional[JointState] = None,
    shortcut_iterations: int = 3,
    convergence: float = 1e-2,
    num_steps: int = 10,
    orientation_weight: Optional[float] = None,
    collision_weight: float = 1e6,
    activation_distance: Optional[float] = None,
    collision_distance: float = 0.0,
    edge_fraction: float = 1.0,
    profile: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> Optional[Pose]:
    """Plan an arm trajectory through SE(3) tool poses.

    ``goal_js``/``start_js`` are arm-joint states (``start_js`` defaults to the
    world's current arm configuration). Returns the tool-pose waypoints as a
    batched Pose, or ``None`` when no path is found. ``activation_distance``
    defaults to ``TAMPConfig.motion_activation_distance``.
    """
    t0 = current_time()
    if orientation_weight is None:
        orientation_weight = world.tamp_config.orientation_weight
    if activation_distance is None:
        activation_distance = world.tamp_config.motion_activation_distance
    arm_joints = world.get_arm_joints(arm)
    if start_js is None:
        start_js = world.get_joint_state(arm_joints)

    start_full = world.reorder_joint_state(start_js)
    goal_full = world.reorder_joint_state(goal_js)
    endpoint_free = torch.isfinite(
        _conf_costs(
            world,
            torch.cat([start_full.position, goal_full.position], dim=0),
            collision_distance=collision_distance,
        )
    )
    if not bool(endpoint_free.all()):
        if verbose:
            print(
                f"{plan_task_path.__name__}) Infeasible endpoints | Start free:"
                f" {bool(endpoint_free[0])} | Goal free: {bool(endpoint_free[1])} | Elapsed:"
                f" {elapsed_time(t0):.3f} sec"
            )
        return None

    with profiler(field="tottime" if profile else None):
        result = plan_tool_roadmap(
            world,
            arm,
            world.get_tool_pose(arm, start_full),
            world.get_tool_pose(arm, goal_full),
            num_steps=num_steps,
            orientation_weight=orientation_weight,
            collision_weight=collision_weight,
            activation_distance=activation_distance,
            collision_distance=collision_distance,
            verbose=verbose,
            **kwargs,
        )
    if result is None:
        if verbose:
            print(f"{plan_task_path.__name__}) Solved: False | Elapsed: {elapsed_time(t0):.3f} sec")
        return None
    positions, quats = result

    for _ in range(shortcut_iterations):
        old_len = _pose_path_length(positions, quats, orientation_weight)
        positions, quats = shortcut_tool_path(
            world,
            arm,
            positions,
            quats,
            num_steps=2 * num_steps,
            orientation_weight=orientation_weight,
            collision_weight=collision_weight,
            activation_distance=activation_distance,
            collision_distance=collision_distance,
            edge_fraction=edge_fraction,
            verbose=verbose,
        )
        new_len = _pose_path_length(positions, quats, orientation_weight)
        if old_len <= 0.0 or (old_len - new_len) / old_len <= convergence:
            break

    waypoints = Pose(position=positions, quaternion=quats)
    if verbose:
        print(
            f"{plan_task_path.__name__}) Solved: True |"
            f" Waypoints: {waypoints.position.shape[0]} |"
            f" Length: {_pose_path_length(positions, quats, orientation_weight):.3f} |"
            f" Orientation weight: {orientation_weight:.3f} |"
            f" Elapsed: {elapsed_time(t0):.3f} sec"
        )
    return waypoints
