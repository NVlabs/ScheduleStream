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
"""GPU-batched PRM-style cspace motion planner for custream2.

Adapted from skillgen_dev/mimicgen/curobo/motion.py: sample collision-free
configurations, build a nearest-neighbour roadmap, batch-check every candidate
edge against the scene/self collision checker on the GPU, then run Dijkstra over
the surviving edges. Planning happens over a subset of joints while the rest
of the active joints are held at the start configuration.

On top of the single-shot roadmap this module adds:
  * ``max_distance`` informed sampling — like BIT*, samples are pruned to the
    ellipsoid ``dist(start, q) + dist(q, goal) < max_distance`` so replanning
    with the best length found so far focuses search on shorter solutions.
  * ``shortcut_path`` — densifies a path and reconnects non-adjacent waypoints
    via an all-pairs roadmap to remove detours.
  * ``plan_joint_path`` — alternates informed roadmap replanning and local
    shortcutting, keeping the shortest collision-free plan.

Used as an alternative to ``World.plan_cspace`` inside ``motion_stream``.
"""
from __future__ import annotations

# Standard Library
import math
from typing import Any, List, Optional

# Third Party
import numpy as np
import torch
from curobo.types import JointState
from scipy import sparse
from scipy.sparse import csgraph

# NVIDIA
from schedulestream.applications.custream2.costs import compute_distances, path_distance
from schedulestream.applications.custream2.utils import resample_polyline
from schedulestream.common.utils import INF, current_time, elapsed_time, profiler

EPSILON = 1e-6


def _conf_costs(
    world: Any,
    full_confs: torch.Tensor,
    activation_distance: float = 1e-2,
    collision_distance: float = 0.0,
    max_batch: int = 2**15,
) -> torch.Tensor:
    """Per-config collision cost for ``[M, dof_active]`` confs.

    A config costs INF when any sphere is within ``collision_distance`` of the scene or the robot
    self-collides; otherwise the cost is the mean intrusion past ``activation_distance`` (0 when
    fully clear), so the roadmap can be biased away from grazing obstacles rather than only
    rejecting contact. The two margins are queried separately: cuRobo's inflated-sphere depth is
    not a clipped distance (at a 5 cm margin it reports up to 8.5 cm on configs clear at 0), so
    deriving the hard test from the soft query mis-rejects clear configs.

    Self-collision is a hard constraint that ignores both distances: neighbouring links always sit
    within any useful margin of each other, so a graded term would tax every config.

    Queries run in chunks of ``max_batch`` configurations to bound peak GPU memory: edge validation
    submits every edge interior at once (~600k configs at 60k edges), which OOMs when the process
    shares the GPU with Isaac.
    """
    if full_confs.shape[0] == 0:
        return torch.zeros(0, device=full_confs.device)
    self_cost = world.self_collision_cost
    costs = []
    for chunk in torch.split(full_confs, max_batch):
        num = chunk.shape[0]
        spheres = world.get_spheres(
            JointState.from_position(chunk, joint_names=world.active_joints)
        )
        colliding = (
            world.get_sphere_distances(spheres, collision_distance=collision_distance) > 0.0
        ).any(dim=-1)
        penetration = world.get_sphere_distances(spheres, collision_distance=activation_distance)
        self_cost.setup_batch_tensors(num, 1)
        self_colliding = self_cost.forward(spheres.unsqueeze(1)).reshape(-1) > 0.0
        active = (spheres[..., 3] > 0.0).sum(dim=-1).clamp(min=1)
        costs.append(torch.where(colliding | self_colliding, INF, penetration.sum(dim=-1) / active))
    return torch.cat(costs)


def _edge_costs(
    world: Any,
    start_full: torch.Tensor,
    joint_idx: List[int],
    confs: torch.Tensor,
    ei: torch.Tensor,
    ej: torch.Tensor,
    num_steps: int,
    activation_distance: float = 1e-2,
    collision_distance: float = 0.0,
) -> torch.Tensor:
    """Mean :func:`_conf_costs` over the edge's interior.

    Averaged rather than summed so the scale is a depth independent of ``num_steps``, matching the
    per-config normalization, then scaled by the edge's plain joint distance (ignoring ``cost``)
    so a long edge grazing over the same fraction of its length costs more than a short one. INF
    when any interior config collides, which both marks the edge unusable and deletes it in
    :func:`_build_graph` (which drops non-finite weights).
    """
    if ei.numel() == 0:
        return torch.zeros(0, device=confs.device)
    device, dtype, d = confs.device, confs.dtype, confs.shape[1]
    weights = torch.linspace(0.0, 1.0, num_steps + 2, device=device, dtype=dtype)[1:-1]
    a = confs[ei].unsqueeze(1)
    b = confs[ej].unsqueeze(1)
    interp = a * (1 - weights).view(1, -1, 1) + b * weights.view(1, -1, 1)
    joint_confs = interp.reshape(-1, d)
    full = start_full.unsqueeze(0).repeat(joint_confs.shape[0], 1)
    full[:, joint_idx] = joint_confs
    costs = _conf_costs(
        world,
        full,
        activation_distance=activation_distance,
        collision_distance=collision_distance,
    )
    mean = costs.reshape(ei.shape[0], num_steps).mean(dim=1)
    length = torch.norm(confs[ei] - confs[ej], dim=-1)
    return torch.where(torch.isfinite(mean), mean * length, mean)


def _edge_weights(
    world: Any,
    start_full: torch.Tensor,
    joint_idx: List[int],
    confs: torch.Tensor,
    ei: torch.Tensor,
    ej: torch.Tensor,
    cost: Optional[str],
    p: float,
) -> torch.Tensor:
    """Per-edge weights for the roadmap over edges ``(ei, ej)``.

    ``cost=None`` uses the ``p``-norm joint distance between the edge's
    endpoints; otherwise each edge is weighted by :func:`compute_distances` in
    that space (``"joint"``/``"task"``/``"sphere"``), expanding each endpoint to its
    full active config (non-planned joints held at ``start_full``) so FK-based
    costs (``task``/``sphere``) are correct.
    """
    if cost is None or ei.numel() == 0:
        return torch.norm(confs[ei] - confs[ej], p=p, dim=-1)
    active_names = world.active_joints
    full_i = start_full.unsqueeze(0).repeat(ei.shape[0], 1)
    full_i[:, joint_idx] = confs[ei]
    full_j = start_full.unsqueeze(0).repeat(ej.shape[0], 1)
    full_j[:, joint_idx] = confs[ej]
    return compute_distances(
        world,
        JointState.from_position(full_i, joint_names=active_names),
        JointState.from_position(full_j, joint_names=active_names),
        cost=cost,
        p=int(p),
    )


def _build_graph(
    num_nodes: int,
    ei: torch.Tensor,
    ej: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
) -> sparse.csr_matrix:
    """Build the undirected weighted roadmap (CSR adjacency) over the ``valid`` edges.

    ``weights`` defaults to unit edge weights; ``valid`` is a bool mask selecting
    which edges to include and defaults to keeping every edge. Edges with a
    non-finite weight are always dropped.
    """
    if weights is None:
        weights = torch.ones(ei.shape[0], device=ei.device)
    if valid is None:
        valid = torch.ones(ei.shape[0], dtype=torch.bool, device=ei.device)
    mask = valid & torch.isfinite(weights)
    sources = ei[mask].cpu().numpy()
    targets = ej[mask].cpu().numpy()
    edge_weights = np.maximum(weights[mask].cpu().numpy().astype(np.float64), 1e-12)
    return sparse.csr_matrix(
        (
            np.concatenate([edge_weights, edge_weights]),
            (np.concatenate([sources, targets]), np.concatenate([targets, sources])),
        ),
        shape=(num_nodes, num_nodes),
    )


def _shortest_path(graph: sparse.csr_matrix, source: int, target: int) -> Optional[List[int]]:
    """Dijkstra over a prebuilt roadmap adjacency; return the node path or None."""
    distances, predecessors = csgraph.dijkstra(
        graph, directed=False, indices=source, return_predecessors=True
    )
    if not np.isfinite(distances[target]):
        return None
    path = [target]
    while path[-1] != source:
        path.append(int(predecessors[path[-1]]))
    return path[::-1]


def _path_weight(graph: sparse.csr_matrix, path: Optional[List[int]]) -> float:
    if path is None:
        return float("inf")
    return float(sum(graph[i, j] for i, j in zip(path[:-1], path[1:])))


def plan_roadmap(
    world: Any,
    goal_js: JointState,
    joints: List[str],
    start_js: Optional[JointState] = None,
    num_confs: int = 5000,
    degree: int = 10,
    num_steps: int = 10,
    max_distance: float = math.inf,
    sample_factor: int = 8,
    start_goal_edge: bool = False,
    p: float = 2.0,
    cost: Optional[str] = "joint",
    collision_weight: float = 1e6,
    activation_distance: float = 1e-2,
    collision_distance: float = 0.0,
    verbose: bool = True,
) -> Optional[JointState]:
    """Plan a collision-free path over ``joints`` from ``start_js`` to ``goal_js``.

    ``goal_js`` is a full active-joint state (batch 1). ``start_js`` defaults to
    the world's current (initial) configuration when omitted. When
    ``max_distance`` is finite, sampled configurations are pruned to those with
    ``dist(start, q) + dist(q, goal) < max_distance`` (informed sampling).
    Returns a sparse waypoint :class:`JointState` over ``joints``
    (start ... goal), or ``None`` when no path is found.

    ``cost`` selects how edge weights are computed: ``None`` (default) uses the
    ``p``-norm joint distance between endpoints; otherwise edges are weighted by
    :func:`compute_distances` in that space (``"joint"``/``"task"``/``"sphere"``),
    which for ``task``/``sphere`` runs FK on each endpoint's full active config.
    """
    t0 = current_time()
    device = world.device
    if start_js is None:
        start_js = world.get_joint_state()
    joint_idx = world.active_joint_indices(joints)

    start_full = world.reorder_joint_state(start_js).position.reshape(-1).to(device)
    goal_full = world.reorder_joint_state(goal_js).position.reshape(-1).to(device)
    start_q, goal_q = start_full[joint_idx], goal_full[joint_idx]

    sampled = world.sample_joint_states(
        num_confs * sample_factor,
        joints=joints,
        collisions=True,
        verbose=False,
    )
    samples = sampled.position[:, joint_idx].to(device)
    if math.isfinite(max_distance):
        detour = torch.norm(samples - start_q, p=p, dim=-1) + torch.norm(
            samples - goal_q, p=p, dim=-1
        )
        samples = samples[detour < max_distance]
    samples = samples[:num_confs]

    confs = torch.cat([start_q.unsqueeze(0), goal_q.unsqueeze(0), samples], dim=0)
    num_nodes = confs.shape[0]
    if num_nodes < 2:
        if verbose:
            print(
                f"{plan_roadmap.__name__}) No collision-free samples | Elapsed:"
                f" {elapsed_time(t0):.3f} sec"
            )
        return None

    dmat = torch.cdist(confs, confs, p=p)
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

    edge_costs = _edge_costs(
        world,
        start_full,
        joint_idx,
        confs,
        ei,
        ej,
        num_steps,
        activation_distance=activation_distance,
        collision_distance=collision_distance,
    )
    weights = _edge_weights(world, start_full, joint_idx, confs, ei, ej, cost, p)
    if collision_weight:
        weights = weights + collision_weight * edge_costs
    graph = _build_graph(num_nodes, ei, ej, weights=weights, valid=torch.isfinite(edge_costs))
    path = _shortest_path(graph, 0, 1)
    joint_state = (
        None if path is None else JointState.from_position(confs[path], joint_names=list(joints))
    )

    if verbose:
        num_components, labels = csgraph.connected_components(graph, directed=False)
        comp_sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        max_shown = 8
        sizes_str = ", ".join(map(str, comp_sizes[:max_shown]))
        if num_components > max_shown:
            sizes_str += ", ..."
        degrees = np.diff(graph.indptr)
        min_degree = int(degrees.min())
        min_degree_count = int((degrees == min_degree).sum())
        distance = path_distance(world, joint_state, cost=cost, p=p)
        path_cost = _path_weight(graph, path)
        print(
            f"{plan_roadmap.__name__}) Nodes: {num_nodes} | Edges:"
            f" {int(torch.isfinite(edge_costs).sum())}/{ei.shape[0]} | Components: [{sizes_str}]"
            f" ({num_components}) | Min degree: {min_degree} ({min_degree_count}) | Start/Goal"
            f" degree: {degrees[0]}/{degrees[1]} | Solved: {path is not None} | Waypoints:"
            f" {0 if path is None else len(path)} | Distance: {distance:.3f} | Cost:"
            f" {path_cost:.3f} | Elapsed: {elapsed_time(t0):.3f} sec"
        )
    return joint_state


def shortcut_path(
    world: Any,
    waypoints: JointState,
    start_js: Optional[JointState] = None,
    num_samples: int = 100,
    num_steps: int = 10,
    p: float = 2.0,
    cost: Optional[str] = "joint",
    edge_fraction: float = 0.25,
    collision_weight: float = 1e6,
    activation_distance: float = 1e-2,
    collision_distance: float = 0.0,
    verbose: bool = False,
) -> JointState:
    """Return a (possibly) shorter path by reconnecting waypoints with shortcuts.

    Densifies ``waypoints`` to ``num_samples`` arc-length samples, then builds an
    all-pairs roadmap over them and runs Dijkstra; collision-free straight
    shortcuts between non-adjacent samples replace detours. Plans over
    ``waypoints``' own joints; ``start_js`` (the config holding non-planned
    joints) defaults to the world's current configuration. Returns the input
    unchanged when no shorter path is found.

    ``cost`` weights the shortcut edges the same way as :func:`plan_roadmap`
    (see :func:`_edge_weights`). ``edge_fraction`` in ``(0, 1]`` keeps only the
    shortest fraction of the all-pairs candidate edges (by joint distance) before
    collision-checking, cutting the number of expensive edge checks; ``1.0``
    keeps them all.
    """
    t0 = current_time()
    if waypoints.position.shape[0] <= 2:
        return waypoints
    device = world.device
    joints = list(waypoints.joint_names)
    if start_js is None:
        start_js = world.get_joint_state()
    joint_idx = world.active_joint_indices(joints)
    start_full = world.reorder_joint_state(start_js).position.reshape(-1).to(device)

    confs = resample_polyline(waypoints.position.to(device), num_samples, p=p)
    num_nodes = confs.shape[0]
    if num_nodes < 3:
        return waypoints

    pairs = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    ei, ej = pairs[0], pairs[1]
    if edge_fraction < 1.0:
        edge_dist = torch.norm(confs[ei] - confs[ej], p=p, dim=-1)
        keep = max(1, int(edge_fraction * ei.shape[0]))
        order = torch.topk(edge_dist, k=keep, largest=False).indices
        ei, ej = ei[order], ej[order]
    edge_costs = _edge_costs(
        world,
        start_full,
        joint_idx,
        confs,
        ei,
        ej,
        num_steps,
        activation_distance=activation_distance,
        collision_distance=collision_distance,
    )
    weights = _edge_weights(world, start_full, joint_idx, confs, ei, ej, cost, p)
    if collision_weight:
        weights = weights + collision_weight * edge_costs
    graph = _build_graph(num_nodes, ei, ej, weights=weights, valid=torch.isfinite(edge_costs))
    path = _shortest_path(graph, 0, num_nodes - 1)
    if path is None:
        return waypoints

    new_waypoints = JointState.from_position(confs[path], joint_names=list(joints))
    if verbose:
        print(
            f"{shortcut_path.__name__}) Samples: {num_nodes} |"
            f" {path_distance(world, waypoints, cost=cost, p=p):.3f} ->"
            f" {path_distance(world, new_waypoints, cost=cost, p=p):.3f} |"
            f" Cost: {_path_weight(graph, path):.3f} |"
            f" Elapsed: {elapsed_time(t0):.3f} sec"
        )
    if (
        path_distance(world, new_waypoints, cost=cost, p=p)
        >= path_distance(world, waypoints, cost=cost, p=p) - EPSILON
    ):
        return waypoints
    return new_waypoints


def _shortcut_until_converged(
    world: Any,
    path: JointState,
    start_js: Optional[JointState],
    shortcut_iterations: int,
    num_steps: int,
    p: float,
    cost: Optional[str],
    convergence: float,
    verbose: bool,
    **kwargs: Any,
) -> JointState:
    """Repeatedly :func:`shortcut_path` ``path`` (extra ``kwargs`` forwarded to it).

    Stops after ``shortcut_iterations`` rounds or once a round improves the
    length by less than ``convergence`` (fractional). Returns the best path found.
    """
    t0 = current_time()
    start_distance = path_distance(world, path, cost=cost, p=p)
    distance = start_distance
    iterations = 0
    for _ in range(shortcut_iterations):
        iterations += 1
        new = shortcut_path(
            world,
            path,
            start_js=start_js,
            num_steps=2 * num_steps,
            cost=cost,
            p=p,
            verbose=False,
            **kwargs,
        )
        new_distance = path_distance(world, new, cost=cost, p=p)
        percent = (distance - new_distance) / distance if distance > 0 else 0.0
        if new_distance < distance:
            path, distance = new, new_distance
        if percent <= convergence:
            break
    if verbose:
        improvement = (start_distance - distance) / start_distance if start_distance > 0 else 0.0
        print(
            f"shortcut) Iterations: {iterations}/{shortcut_iterations} |"
            f" Distance: {start_distance:.3f} -> {distance:.3f} ({100 * improvement:.1f}%) |"
            f" Elapsed: {elapsed_time(t0):.3f} sec"
        )
    return path


def plan_joint_path(
    world: Any,
    goal_js: JointState,
    start_js: Optional[JointState] = None,
    roadmap_iterations: int = 1,
    shortcut_iterations: int = 3,
    convergence: float = 1e-2,
    num_steps: int = 10,
    p: float = 2.0,
    cost: Optional[str] = "joint",
    collision_weight: float = 1e6,
    activation_distance: Optional[float] = None,
    collision_distance: float = 0.0,
    edge_fraction: float = 0.25,
    profile: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Optional[JointState]:
    """Alternate informed roadmap replanning and local shortcutting.

    Plans over the joints named in ``goal_js``; ``start_js`` defaults to the
    world's current (initial) configuration, and any joints it omits are
    completed from that same state. Each roadmap iteration
    replans with ``max_distance`` set to the best path length found so far
    (informed sampling), then locally shortcuts the best path. Stops early when
    an iteration improves the length by less than ``convergence`` (fractional).
    Returns the shortest waypoint path or None. ``activation_distance`` defaults
    to ``TAMPConfig.motion_activation_distance``.

    ``edge_fraction`` is explicit rather than left to ``kwargs``: kwargs fan out to both
    :func:`plan_roadmap` and :func:`shortcut_path`, and it is only accepted by the latter.
    """
    t0 = current_time()
    if activation_distance is None:
        activation_distance = world.tamp_config.motion_activation_distance
    joints = list(goal_js.joint_names)
    if start_js is None:
        start_js = world.get_joint_state()

    start_full = world.reorder_joint_state(start_js).position.reshape(1, -1)
    goal_full = world.reorder_joint_state(goal_js).position.reshape(1, -1)
    endpoint_free = torch.isfinite(
        _conf_costs(
            world,
            torch.cat([start_full, goal_full], dim=0),
            activation_distance=activation_distance,
            collision_distance=collision_distance,
        )
    )
    if not bool(endpoint_free.all()):
        print(
            f"{plan_joint_path.__name__}) Infeasible endpoints |"
            f" Start free: {bool(endpoint_free[0])} | Goal free: {bool(endpoint_free[1])} |"
            f" Elapsed: {elapsed_time(t0):.3f} sec"
        )
        return None

    best: Optional[JointState] = None
    best_distance = math.inf
    for i in range(roadmap_iterations):
        prev_distance = best_distance

        with profiler(field="tottime" if profile else None):
            candidate = plan_roadmap(
                world,
                goal_js,
                joints,
                start_js=start_js,
                num_steps=num_steps,
                max_distance=best_distance,
                cost=cost,
                p=p,
                collision_weight=collision_weight,
                activation_distance=activation_distance,
                collision_distance=collision_distance,
                verbose=verbose,
                **kwargs,
            )
        if candidate is not None:
            candidate_distance = path_distance(world, candidate, cost=cost, p=p)
            if candidate_distance < best_distance:
                best, best_distance = candidate, candidate_distance
        if best is None:
            break

        best = _shortcut_until_converged(
            world,
            best,
            start_js,
            shortcut_iterations,
            2 * num_steps,
            p,
            cost,
            convergence,
            verbose,
            edge_fraction=edge_fraction,
            collision_weight=collision_weight,
            activation_distance=activation_distance,
            collision_distance=collision_distance,
        )
        best_distance = path_distance(world, best, cost=cost, p=p)

        if math.isfinite(prev_distance):
            percent = (prev_distance - best_distance) / prev_distance if prev_distance > 0 else 0.0
            if percent <= convergence:
                break

    print(
        f"{plan_joint_path.__name__}) Solved: {best is not None} | Cost: {cost} | Distance:"
        f" {best_distance:.3f} | Waypoints: {0 if best is None else best.position.shape[0]} |"
        f" Elapsed: {elapsed_time(t0):.3f} sec"
    )
    return best
