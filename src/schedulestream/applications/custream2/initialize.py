#!/usr/bin/env python3
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
import math
import random
from itertools import islice
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
from curobo.types import Pose

# NVIDIA
from schedulestream.applications.custream2.collision import object_object_collision
from schedulestream.applications.custream2.object import PlacementConfig
from schedulestream.applications.custream2.placement import Placement, placement_stream
from schedulestream.applications.custream2.state import Configuration, State
from schedulestream.applications.custream2.utils import (
    GRAVITY,
    PI,
    concatenate_poses,
    pose_from_pos_quat,
    position_from_pose,
    quat_from_euler,
    to_matrix,
    transform_spheres,
)
from schedulestream.applications.custream2.world import World
from schedulestream.common.graph import topological_sort
from schedulestream.common.utils import current_time, elapsed_time, profiler


def get_box_supporting_object(
    world: World,
    obj: str,
    supports: Optional[List[Any]] = None,
    z_threshold: float = -1e-3,
) -> Optional[str]:
    """Return the name of the object supporting ``obj`` at its current pose
    (the nearest ``supports`` surface at or below its base, default: all
    objects), or None when nothing is underneath."""
    if supports is None:
        supports = world.objects
    child_obj = world.get_object(obj)
    base_center = position_from_pose(child_obj.get_bottom_pose())
    supports = [
        o
        for o in supports
        if (o != child_obj) and (base_center - o.get_bounding_box().top.center)[2] >= z_threshold
    ]
    if not supports:
        return None
    return min(supports, key=lambda o: o.get_bounding_box().top.distance(base_center)).name


def get_contact_supporting_object(
    world: World,
    obj: str,
    supports: Optional[List[Any]] = None,
    contact_distance: float = 1e-2,
    tolerance: float = PI / 4,
    verbose: bool = False,
) -> Optional[str]:
    """Contact-based alternative to :func:`get_box_supporting_object`: assign each
    of ``obj``'s spheres to the closest ``supports`` member (default: all
    objects) whose push-out direction approximately opposes gravity (within
    ``tolerance`` radians of world +z), then return the support owning the
    most spheres, or None when nothing is in contact. ``contact_distance``
    inflates the contact margin so resting (non-penetrating) contact still
    registers.
    """
    start_time = current_time()
    if supports is None:
        supports = world.objects
    body = world.get_object(obj)
    spheres_world = transform_spheres(body.get_pose(), body.spheres_tensor)
    names: List[str] = []
    depths: List[torch.Tensor] = []
    for support in supports:
        if support.name == obj:
            continue
        support_pose = support.get_pose()
        spheres_local = transform_spheres(support_pose.inverse(), spheres_world)
        distances, directions = support.get_sphere_contacts(
            spheres_local, collision_distance=contact_distance
        )
        rotation = torch.as_tensor(
            to_matrix(support_pose)[:3, :3], dtype=torch.float32, device=directions.device
        )
        up_local = rotation.T @ torch.tensor(
            -GRAVITY, dtype=torch.float32, device=directions.device
        )
        cosines = (torch.nn.functional.normalize(directions, dim=-1) * up_local).sum(dim=-1)
        aligned = (distances > 0.0) & (cosines >= math.cos(tolerance))
        depths.append(torch.where(aligned, distances, torch.zeros_like(distances)).flatten().cpu())
        names.append(support.name)
    if not names:
        return None

    stacked = torch.stack(depths)
    closest = stacked.argmax(dim=0)
    contact = stacked.amax(dim=0) > 0.0
    num_spheres = stacked.shape[1]
    counts = torch.bincount(closest[contact], minlength=len(names))
    best_count = int(counts.max())
    best_name = names[int(counts.argmax())] if best_count else None
    if verbose and (best_name is not None):
        print(
            f"{get_contact_supporting_object.__name__}) Object: {obj} |"
            f" Support: {best_name} | Contacts: {best_count}/{num_spheres}"
            f" ({best_count / num_spheres:.1%}) |"
            f" Elapsed: {elapsed_time(start_time):.3f} sec"
        )
    return best_name


def get_stacked_objects(
    world: World,
    names: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, str]:
    """Return {top_obj_name: bottom_obj_name} based on current object positions."""
    if names is None:
        names = world.movable_names
    parent_from_child = {}
    for child_name in names:
        parent = get_contact_supporting_object(world, child_name, **kwargs)
        if parent is not None:
            parent_from_child[child_name] = parent
    return parent_from_child


def order_stacked_objects(
    world: World, stacked_objects: Optional[Dict[str, str]] = None, **kwargs: Any
) -> Dict[str, Optional[str]]:
    """Return {obj_name: parent_name} sorted bottom-up by stacking order."""
    if stacked_objects is None:
        stacked_objects = get_stacked_objects(world, **kwargs)
    stacked_orders = {(bottom, top) for top, bottom in stacked_objects.items()}
    object_names = world.object_names
    object_names = sorted(
        object_names, key=lambda o: np.linalg.norm(world.get_object(o).extent), reverse=True
    )
    object_names = topological_sort(stacked_orders, object_names)
    return {obj: stacked_objects.get(obj, None) for obj in object_names}


def arm_object_unreachable(
    world: World,
    batch_inputs: List[Tuple[Optional[str], str, Placement]],
    reachable_fraction: float = 1.0,
    verbose: bool = True,
) -> List[bool]:
    """Return whether each (arm, object, placement) is unreachable by the arm.

    When ``arm`` is None, every world arm is tried and an input is reachable
    (False) if any arm can reach it, i.e. unreachable only when all arms fail.

    Per-arm reachability: transforms ``obj``'s collision spheres into world
    frame at each candidate placement's pose and checks the sphere centers
    against the arm's reachability :class:`OccupancyGrid`. A placement is
    considered reachable when at least ``reachable_fraction`` of the sphere
    centers (and no fewer than one) fall inside the grid: 0 requires any
    sphere, 1 requires every sphere.
    """
    t0 = current_time()
    arms, objs, placements = zip(*batch_inputs)
    [arm] = set(arms)
    [obj] = set(objs)
    if arm is None:
        unreachable = [True] * len(placements)
        for other_arm in world.arms:
            results = arm_object_unreachable(
                world,
                [(other_arm, obj, p) for p in placements],
                reachable_fraction=reachable_fraction,
                verbose=False,
            )
            unreachable = [u and r for u, r in zip(unreachable, results)]
        if verbose:
            print(
                f"{arm_object_unreachable.__name__}) Arm: any | Object: {obj} |"
                f" Unreachable: {sum(unreachable)}/{len(unreachable)} |"
                f" Elapsed: {elapsed_time(t0):.3f} sec"
            )
        return unreachable
    grid = world.get_reachability_grid(arm)
    body = world.get_object(obj)
    pose_batch = concatenate_poses([p.get_pose() for p in placements])
    sphere_centers = transform_spheres(pose_batch, body.spheres_tensor)[..., :3]
    n_poses, n_spheres, _ = sphere_centers.shape
    contains = grid.contains(sphere_centers.reshape(n_poses * n_spheres, 3))
    contains = contains.view(n_poses, n_spheres)
    min_contained = max(1, math.ceil(reachable_fraction * n_spheres))
    unreachable = (contains.sum(dim=-1) < min_contained).tolist()
    if verbose:
        print(
            f"{arm_object_unreachable.__name__}) Arm: {arm} | Object: {obj} |"
            f" Unreachable: {sum(unreachable)}/{len(unreachable)} |"
            f" Elapsed: {elapsed_time(t0):.3f} sec"
        )
    return unreachable


def sample_states(
    world: World,
    stacked_objects: Optional[Dict[str, str]] = None,
    arms: Optional[Union[bool, List[Optional[str]]]] = False,
    num_states: int = 1,
    collision_distance: Optional[float] = 1e-2,
    batch_size: int = 1000,
    verbose: bool = True,
    **kwargs: Any,
) -> List[State]:
    """Sample up to num_states collision-free placement States for movable objects.

    ``stacked_objects`` is a ``{top: bottom}`` mapping describing which object
    sits on which. When ``None`` (default), it is inferred from the world via
    :func:`get_stacked_objects`. Pass an explicit dict to override the
    auto-detected stack relations (e.g. when stack semantics come from a task
    definition rather than current object poses).

    ``arms`` is the set of arms a placement must be reachable by at least one
    of; it defaults to ``world.arms``.
    """
    if stacked_objects is None:
        stacked_objects = get_stacked_objects(world)
    if arms is None:
        arms = world.arms
    elif arms is True:
        arms = [None]
    elif arms is False:
        arms = []
    stackable_names = set(stacked_objects)
    ordered_stacks = order_stacked_objects(world, stacked_objects)

    t0 = current_time()
    rollouts: List[Dict[str, Placement]] = [{} for _ in range(num_states)]
    active = [True] * num_states

    for obj_name, parent_name in ordered_stacks.items():
        if not any(active):
            break
        if obj_name not in stackable_names:
            placement = Placement(world, obj_name)
            for result in rollouts:
                result[obj_name] = placement
            continue
        if parent_name is None:
            raise NotImplementedError(f"Sampled object {obj_name!r} has no parent")

        candidate_indices: List[int] = []
        candidates: List[Placement] = []
        offsets: List[int] = [0]
        candidate_cache: Dict[Placement, List[Placement]] = {}
        for i in range(num_states):
            if not active[i]:
                offsets.append(offsets[-1])
                continue
            parent_placement = rollouts[i][parent_name]
            cached = candidate_cache.get(parent_placement)
            if cached is None:
                gen = placement_stream(
                    world,
                    obj_name,
                    parent_name,
                    parent_placement,
                    object_proximity=0.0,
                    batch_size=batch_size,
                    **kwargs,
                )
                rollout_candidates = list(islice(gen, batch_size))
                candidate_cache[parent_placement] = rollout_candidates
            else:
                rollout_candidates = list(cached)
                random.shuffle(rollout_candidates)
            candidate_indices.extend([i] * len(rollout_candidates))
            candidates.extend(rollout_candidates)
            offsets.append(len(candidates))

        valid = [True] * len(candidates)
        for arm in arms:
            batch = [(arm, obj_name, c) for c in candidates]
            results = arm_object_unreachable(world, batch)
            valid = [v and not r for v, r in zip(valid, results)]
        if collision_distance is not None:
            coll_batch: List[Tuple[str, Placement, str, Placement]] = []
            batch_to_k: List[int] = []
            for i in range(num_states):
                if not active[i]:
                    continue
                other_items = [
                    (n, p)
                    for n, p in rollouts[i].items()
                    if n != parent_name and n in stackable_names
                ]
                if not other_items:
                    continue
                lo, hi = offsets[i], offsets[i + 1]
                for k in range(lo, hi):
                    for other_name, other_p in other_items:
                        coll_batch.append((obj_name, candidates[k], other_name, other_p))
                        batch_to_k.append(k)
            if coll_batch:
                coll_results = object_object_collision(
                    world,
                    coll_batch,
                    collision_distance=collision_distance,
                    verbose=False,
                )
                for k, result in zip(batch_to_k, coll_results):
                    if result:
                        valid[k] = False

        for i in range(num_states):
            if not active[i]:
                continue
            lo, hi = offsets[i], offsets[i + 1]
            placement = next((candidates[k] for k in range(lo, hi) if valid[k]), None)
            if placement is None:
                active[i] = False
            else:
                rollouts[i][obj_name] = placement

    states: List[State] = []
    for i, placements in enumerate(rollouts):
        if active[i] and placements:
            conf = Configuration(world, world.get_joint_state())
            states.append(State(world, conf=conf, attachments=list(placements.values())))
    if verbose:
        print(
            f"sample_states) Sampled: {len(states)}/{num_states} |"
            f" Elapsed: {elapsed_time(t0):.3f} sec"
        )
    return states


def generate_states(
    world: World,
    attempts: int = 5,
    batch_size: int = 1,
    display: bool = False,
    **kwargs: Any,
) -> Iterator[State]:
    """Indefinitely yield valid States, retrying sampling up to attempts times per batch."""
    while True:
        for _ in range(attempts):
            states = sample_states(world, num_states=batch_size, **kwargs)
            if states:
                break
        else:
            break
        for state in states:
            state.set()
            if display:
                state.show()
            yield state


if __name__ == "__main__":
    # Standard Library
    import argparse

    # NVIDIA
    from schedulestream.applications.custream2.animate import animate_states
    from schedulestream.applications.custream2.scene import PRIMITIVES, SCENES, create_franka_world
    from schedulestream.applications.custream2.utils import set_seed
    from schedulestream.applications.trimesh2d.utils import get_video_path, save_frames

    parser = argparse.ArgumentParser(
        description="Sample and animate collision-free placement states for a Franka scene."
    )
    parser.add_argument(
        "scene", nargs="?", default="unpack", choices=list(SCENES), help="Scene to load."
    )
    parser.add_argument("--num-states", type=int, default=10, help="Number of states to sample.")
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Placement candidates sampled per attempt."
    )
    parser.add_argument("--num-robots", type=int, default=None, help="Number of robots.")
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        choices=["franka", "droid"],
        help="Robot variant; forwarded to load_robot_config.",
    )
    parser.add_argument(
        "--num-objects", type=int, default=2, help="Number of objects in the scene."
    )
    parser.add_argument(
        "--primitive",
        type=str,
        default=None,
        choices=PRIMITIVES,
        help="Movable object primitive (e.g. cuboid, ycb).",
    )
    parser.add_argument(
        "--steps-per-state", type=int, default=25, help="Number of animation steps per state."
    )
    parser.add_argument(
        "--distance", type=float, default=0.0, help="Collision distance for state sampling"
    )
    parser.add_argument("--headless", action="store_true", help="Sample states without animating")
    parser.add_argument("-v", "--video", type=str, nargs="?", const="", default=None)
    parser.add_argument(
        "--spheres",
        action="store_true",
        help="Visualize collision spheres (passes debug=True to World)",
    )
    parser.add_argument(
        "--reachable", action="store_true", help="Only sample placements reachable by an arm"
    )
    parser.add_argument(
        "--orientations",
        type=int,
        default=None,
        help="Number of stable resting orientations to sample per movable object",
    )
    parser.add_argument("--profile", action="store_true", help="Enables profiling")
    parser.add_argument(
        "--seed", nargs="?", const=None, default=0, type=int, help="The random seed."
    )
    args = parser.parse_args()

    set_seed(seed=args.seed)
    optional = dict(
        num_robots=args.num_robots,
        num_objects=args.num_objects,
        primitive=args.primitive,
        robot=args.robot,
    )
    kwargs = {key: value for key, value in optional.items() if value is not None}
    world = create_franka_world(args.scene, debug=args.spheres, **kwargs)
    if args.orientations is not None:
        for name in world.movable_names:
            body = world.get_object(name)
            if body.placement_config is None:
                body.placement_config = PlacementConfig()
            body.placement_config.stable_orientations = args.orientations
    with profiler(field="cumtime" if args.profile else None, num=25):
        states = sample_states(
            world,
            arms=args.reachable,
            num_states=args.num_states,
            batch_size=args.batch_size,
            collision_distance=args.distance,
            verbose=True,
        )
    video_path = get_video_path(args.video, name=args.scene)
    if states:
        if not args.headless or video_path is not None:
            frames = animate_states(
                states, steps_per_state=args.steps_per_state, record=(video_path is not None)
            )
            if video_path is not None:
                save_frames(frames, video_path=video_path)
    elif not args.headless:
        print("No valid state found")
        world.show()
