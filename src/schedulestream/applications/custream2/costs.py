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
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

# Third Party
import numpy as np
import torch
from curobo.types import JointState, Pose

# NVIDIA
from schedulestream.applications.custream2.utils import (
    concatenate_joint_states,
    extract_joint_state,
    to_cpu,
)
from schedulestream.common.utils import current_time, elapsed_time, safe_zip

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.applications.custream2.world import World


def _interpolate_joint_positions(
    joint_state1: JointState, joint_state2: JointState, num_steps: int
) -> JointState:
    """Linearly interpolate position between two same-shape JointStates at
    ``num_steps + 1`` points and return a flattened ``[B*(num_steps+1)]``
    JointState. Index ``i=0`` is ``joint_state1`` and ``i=num_steps`` is
    ``joint_state2``; intermediate indices subdivide the straight-line path
    in joint space at equal alpha.
    """
    p1 = joint_state1.position
    p2 = joint_state2.position
    alphas = torch.linspace(0.0, 1.0, num_steps + 1, device=p1.device).view(1, -1, 1)
    interp = p1.unsqueeze(1) + alphas * (p2 - p1).unsqueeze(1)
    return JointState.from_position(
        interp.reshape(-1, p1.shape[-1]), joint_names=list(joint_state1.joint_names)
    )


def compute_joint_distances(
    joint_state1: JointState,
    joint_state2: JointState,
    num_steps: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    """Per-pair joint-space distance between two same-size joint states.

    Mirrors :func:`compute_ee_distances` but on the raw joint vectors rather
    than the tool-link positions. ``joint_state1`` and ``joint_state2`` must
    have matching batch size ``B``; returns a ``[B]`` tensor of L2 distances
    between their joint positions. ``**kwargs`` is forwarded to
    ``torch.norm`` so callers can swap in a different ``p``-norm if needed.

    ``num_steps`` is accepted for API parity with :func:`compute_ee_distances`
    and :func:`compute_joint_sphere_distances` but is a no-op here: the interpolated
    path is a straight line in joint space, so the sum of ``num_steps``
    segment norms collapses to the endpoint norm for any ``num_steps >= 1``.
    """
    if joint_state1.position.shape != joint_state2.position.shape:
        raise ValueError(
            "compute_joint_distances: joint states differ in shape "
            f"{tuple(joint_state1.position.shape)} vs {tuple(joint_state2.position.shape)}"
        )
    if joint_state1.position.shape[0] == 0:
        return torch.zeros(0, device=joint_state1.position.device)
    return torch.norm(joint_state2.position - joint_state1.position, dim=1, **kwargs)


def compute_pose_distances(
    world: World,
    pose1: Pose,
    pose2: Pose,
    position_weight: Optional[float] = None,
    orientation_weight: Optional[float] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Per-pair weighted SE(3) distance between two same-size batched Poses.

    Returns ``position_weight * ||dp|| + orientation_weight * angle`` as a
    ``[B]`` tensor. The weights default to the inverses of the TAMPConfig
    velocities, making the distance a travel-time estimate.
    ``**kwargs`` is forwarded to ``torch.norm``.
    """
    if position_weight is None:
        position_weight = 1.0 / world.tamp_config.position_velocity
    if orientation_weight is None:
        orientation_weight = 1.0 / world.tamp_config.orientation_velocity
    pos_d = torch.norm(pose2.position - pose1.position, dim=-1, **kwargs)
    dot = (pose1.quaternion * pose2.quaternion).sum(dim=-1).abs().clamp(max=1.0)
    ori_d = 2.0 * torch.acos(dot)
    return position_weight * pos_d + orientation_weight * ori_d


def compute_ee_distances(
    world: World,
    joint_state1: JointState,
    joint_state2: JointState,
    num_steps: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    """Per-pair end-effector distance between two same-size joint states.

    Linearly interpolates ``num_steps + 1`` joint configurations along the
    straight-line path from ``joint_state1`` to ``joint_state2`` in joint
    space, runs batched FK on the flattened ``B*(num_steps+1)`` batch,
    then sums the per-segment translation and orientation deltas to form a
    discrete arc-length distance per pair as a ``[B]`` tensor.
    ``num_steps=1`` reproduces the original endpoint-only cost.
    ``**kwargs`` is forwarded to :func:`compute_pose_distances`
    (e.g. ``position_weight``/``orientation_weight``).
    """
    if joint_state1.position.shape != joint_state2.position.shape:
        raise ValueError(
            "compute_ee_distance: joint states differ in shape "
            f"{tuple(joint_state1.position.shape)} vs {tuple(joint_state2.position.shape)}"
        )
    B = joint_state1.position.shape[0]
    if B == 0:
        return torch.zeros(0, device=joint_state1.position.device)
    interp = _interpolate_joint_positions(joint_state1, joint_state2, num_steps)
    full = world.reorder_joint_state(interp)
    tool_poses = world.compute_kinematics(full).tool_poses
    positions = tool_poses.position[:, 0, 0, :].reshape(B, num_steps + 1, 3)
    quaternions = tool_poses.quaternion[:, 0, 0, :].reshape(B, num_steps + 1, 4)
    seg = compute_pose_distances(
        world,
        Pose(
            position=positions[:, :-1].reshape(-1, 3), quaternion=quaternions[:, :-1].reshape(-1, 4)
        ),
        Pose(
            position=positions[:, 1:].reshape(-1, 3), quaternion=quaternions[:, 1:].reshape(-1, 4)
        ),
        **kwargs,
    )
    return seg.reshape(B, num_steps).sum(dim=1)


def compute_sphere_distances(
    spheres1: torch.Tensor,
    spheres2: torch.Tensor,
    weighted: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Per-pair sphere translation distance between two same-shape
    ``[..., S, 4]`` sphere sets, summed over spheres -> ``[...]``.

    When ``weighted`` (default), each sphere's contribution is scaled by its
    volume ``4/3*pi*r^3`` using the endpoint-average radius (clamped
    non-negative) so larger limbs dominate. ``**kwargs`` forwards to
    ``torch.norm``.
    """
    seg = torch.norm(spheres2[..., :3] - spheres1[..., :3], dim=-1, **kwargs)
    if not weighted:
        return seg.sum(dim=-1)
    radii = torch.clamp((spheres1[..., 3] + spheres2[..., 3]) / 2.0, min=0.0)
    volumes = 4.0 / 3.0 * torch.pi * radii.pow(3)
    return (volumes * seg).sum(dim=-1)


def compute_task_sphere_distances(
    world: World,
    arm: str,
    poses1: Pose,
    poses2: Pose,
    weighted: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Per-pair translation distance of ``arm``'s rigid hand spheres between
    two same-size batched tool Poses -> ``[B]``.

    The hand spheres are repositioned rigidly at each pose (no IK), so this is
    the task-space analogue of :func:`compute_joint_sphere_distances` with
    ``hands_only``. ``weighted``/``**kwargs`` forward to
    :func:`compute_sphere_distances`.
    """
    spheres1 = world.get_hand_spheres(arm, poses1, inactive=False)
    spheres2 = world.get_hand_spheres(arm, poses2, inactive=False)
    return compute_sphere_distances(spheres1, spheres2, weighted=weighted, **kwargs)


def compute_joint_sphere_distances(
    world: World,
    joint_state1: JointState,
    joint_state2: JointState,
    num_steps: int = 1,
    weighted: bool = True,
    hands_only: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Per-pair robot-sphere translation distance between two same-size joint states.

    Linearly interpolates ``num_steps + 1`` joint configurations along the
    straight-line joint-space path between the two states, runs batched FK
    on the flattened ``B*(num_steps+1)`` batch, and sums per-segment
    per-sphere translation deltas to form a discrete arc-length distance per
    pair. When ``weighted=True`` (default), each segment-sphere contribution
    is scaled by its volume ``4/3·π·r³`` using the segment-endpoint average
    radius (clamped non-negative) so larger limbs dominate the cost.
    ``num_steps=1`` reproduces the original endpoint-only cost.

    When ``hands_only`` is set, only spheres rigidly attached to any arm's
    hand links (union of :meth:`World.get_hand_sphere_indices` across
    ``world.arms``) contribute — the returned distance then measures how
    far the grippers move between the two configurations, ignoring
    elbow/torso motion. Default keeps every robot sphere in the cost.

    ``**kwargs`` is forwarded to ``torch.norm``.
    """
    if joint_state1.position.shape != joint_state2.position.shape:
        raise ValueError(
            "compute_joint_sphere_distances: joint states differ in shape "
            f"{tuple(joint_state1.position.shape)} vs {tuple(joint_state2.position.shape)}"
        )
    B = joint_state1.position.shape[0]
    if B == 0:
        return torch.zeros(0, device=joint_state1.position.device)
    interp = _interpolate_joint_positions(joint_state1, joint_state2, num_steps)
    full = world.reorder_joint_state(interp)
    spheres = world.get_spheres(full)
    if hands_only:
        hand_indices = sorted({i for arm in world.arms for i in world.get_hand_sphere_indices(arm)})
        hand_indices = torch.as_tensor(
            hand_indices,
            dtype=torch.long,
            device=spheres.device,
        )
        spheres = spheres[..., hand_indices, :]
    S = spheres.shape[1]
    spheres = spheres.reshape(B, num_steps + 1, S, 4)
    seg = compute_sphere_distances(spheres[:, :-1], spheres[:, 1:], weighted=weighted, **kwargs)
    return seg.sum(dim=1)


COSTS = [
    "zero",
    "random",
    "joint",
    "task",
    "task-position",
    "task-rotation",
    "sphere",
    "task-sphere",
]


def compute_distances(
    world: World,
    joint_state1: JointState,
    joint_state2: JointState,
    cost: Optional[str] = "sphere",
    num_steps: int = 1,
    p: int = 2,
) -> torch.Tensor:
    """Dispatch to the per-pair distance cost.

    ``None`` is treated as ``"joint"`` (the p-norm joint distance). Metrics:
      * ``"zero"`` — constant zero (uniform cost: search order alone decides).
      * ``"random"`` — uniform random in [0, 1) per pair (baseline; seeded via ``set_seed``).
      * ``"joint"`` — p-norm distance in joint space.
      * ``"task"`` — weighted tool-frame translation + rotation arc length.
      * ``"task-position"`` — tool-frame translation arc length only.
      * ``"task-rotation"`` — tool-frame geodesic rotation arc length (radians).
      * ``"sphere"`` — volume-weighted translation over all robot spheres.
      * ``"task-sphere"`` — volume-weighted translation over the hand spheres only.
    """
    if cost == "zero":
        return torch.zeros(joint_state1.position.shape[0], device=joint_state1.position.device)
    if cost == "random":
        return torch.rand(joint_state1.position.shape[0], device=joint_state1.position.device)
    if cost is None or cost == "joint":
        return compute_joint_distances(joint_state1, joint_state2, num_steps=num_steps, p=p)
    if cost == "task":
        return compute_ee_distances(world, joint_state1, joint_state2, num_steps=num_steps, p=p)
    if cost == "task-position":
        return compute_ee_distances(
            world,
            joint_state1,
            joint_state2,
            num_steps=num_steps,
            p=p,
            position_weight=1.0,
            orientation_weight=0.0,
        )
    if cost == "task-rotation":
        return compute_ee_distances(
            world,
            joint_state1,
            joint_state2,
            num_steps=num_steps,
            p=p,
            position_weight=0.0,
            orientation_weight=1.0,
        )
    if cost == "sphere":
        return compute_joint_sphere_distances(
            world, joint_state1, joint_state2, num_steps=num_steps, p=p
        )
    if cost == "task-sphere":
        return compute_joint_sphere_distances(
            world,
            joint_state1,
            joint_state2,
            num_steps=num_steps,
            p=p,
            hands_only=True,
        )
    raise NotImplementedError(f"compute_distances: unsupported cost {cost!r}")


def path_distance(world: World, path: Optional[JointState], **kwargs: Any) -> float:
    """Return the total arc length of a ``path`` :class:`JointState`.

    Sums :func:`compute_distances` over consecutive waypoint pairs. ``None``
    yields ``inf`` and fewer than two waypoints yields ``0``. ``**kwargs`` is
    forwarded to :func:`compute_distances`.
    """
    if path is None:
        return float("inf")
    pos = path.position
    if pos.shape[0] < 2:
        return 0.0
    names = list(path.joint_names)
    js1 = JointState.from_position(pos[:-1], joint_names=names)
    js2 = JointState.from_position(pos[1:], joint_names=names)
    return compute_distances(world, js1, js2, **kwargs).sum().item()


def distance_function(
    world: Any,
    batch_inputs: List[Tuple[str, Any, Any]],
    holding_weight: float = 1.0,
    verbose: bool = False,
    **kwargs: Any,
) -> List[float]:
    """Return per-pair distances for a batch of (arm, conf, conf) tuples.

    ``cost`` selects the distance space:
      * ``"joint"`` (default) — L-``ord`` norm in the arm's joint subspace.
      * ``"task"`` — L-``ord`` norm between the arms' tool-link positions
        (orientation ignored), computed via batched FK on the full joint
        state. Calls :func:`compute_ee_distances` per pair internally.
      * ``"sphere"`` — volume-weighted L-``ord`` sum over robot collision
        spheres; calls :func:`compute_joint_sphere_distances` internally.
      * ``"task-position"`` / ``"task-rotation"`` / ``"task-sphere"`` — tool-frame
        translation-only, rotation-only, and hand-spheres-only variants.

    ``num_steps`` controls discrete arc-length integration along the
    straight-line joint-space interpolant between each pair: ``num_steps=1``
    keeps the original endpoint-only cost, higher values sum the cost
    over ``num_steps`` segments. Joint-space distances are unaffected (the
    interpolant is a straight line, so the segments sum to the endpoint).

    ``holding_weight`` scales the distance of any pair whose endpoint
    configurations hold a grasp (e.g. > 1 penalizes moving while carrying).

    All inputs in ``batch_inputs`` must share the same ``arm``; results
    are returned in the original ``batch_inputs`` order. When ``verbose``
    is set, prints the arm, cost, batch size, and elapsed time.
    """
    if not batch_inputs:
        return []
    t0 = current_time()
    arms, conf1s, conf2s = safe_zip(*batch_inputs)
    holding = [bool(conf1.grasps or conf2.grasps) for conf1, conf2 in zip(conf1s, conf2s)]
    unique_arms = set(arms)
    assert (
        len(unique_arms) == 1
    ), f"distance_function: batch must share one arm, got {sorted(unique_arms)}"
    [arm] = unique_arms
    joint_state1 = concatenate_joint_states(c1.joint_state for c1 in conf1s)
    joint_state2 = concatenate_joint_states(c2.joint_state for c2 in conf2s)
    distances = compute_distances(world, joint_state1, joint_state2, **kwargs)
    result = to_cpu(distances).tolist()
    result = [
        holding_weight * distance if held else distance for distance, held in zip(result, holding)
    ]
    if verbose:
        print(
            f"distance_function) Arm: {arm} | Cost: {kwargs.get('cost', 'sphere')}"
            f" | Steps: {kwargs.get('num_steps', 1)}"
            f" | Batch: {len(batch_inputs)}"
            f" | Distance: [{min(result):.3f}, {max(result):.3f}]"
            f" | Elapsed: {elapsed_time(t0):.3f} sec"
        )
    return result
