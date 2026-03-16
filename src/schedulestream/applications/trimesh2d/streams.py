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
import random
from itertools import cycle
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Third Party
import numpy as np
import trimesh

# NVIDIA
from schedulestream.applications.trimesh2d.geometry import (
    Position2,
    bounds_contain_vector,
    extend_bounds,
    get_mesh_bounds,
    get_mesh_center,
)
from schedulestream.applications.trimesh2d.samplers import Traj as Path
from schedulestream.applications.trimesh2d.samplers import interpolate_waypoints, to_bounds3d
from schedulestream.applications.trimesh2d.world import State2D, World2D


def sample_grasps(obj: str, num: int = 1) -> Iterator[Position2]:
    grasp = np.zeros(2)
    for _ in range(num):
        yield grasp.copy()


def get_placement_bounds(world: World2D, obj: str, support: str) -> Tuple[Position2, Position2]:
    obj_bounds = get_mesh_bounds(world.get_geometry(obj))
    support_bounds = get_mesh_bounds(world.current_geometry(support))

    y = support_bounds[1][1] - obj_bounds[0][1]
    x_lower = support_bounds[0][0] - obj_bounds[0][0]
    x_upper = support_bounds[1][0] - obj_bounds[1][0]
    placement_bounds = (np.array([x_lower, y]), np.array([x_upper, y]))
    return placement_bounds


def center_placement(world: World2D, obj: str, obj2: str) -> Position2:
    support_center = get_mesh_center(world.current_geometry(obj2))
    x = support_center[0]

    (_, y), (_, _) = get_placement_bounds(world, obj, obj2)
    placement = np.array([x, y])
    return placement


def stack_placement(world: World2D, obj: str, obj2: str) -> Position2:
    placement = center_placement(world, obj, obj2)
    world.set_conf(obj, placement)
    return placement


def sample_placements(
    world: World2D,
    obj: str,
    obj2: str,
    description: Optional[str] = None,
    p_valid: Optional[float] = None,
    display: bool = False,
) -> Iterator[Position2]:
    if obj == obj2:
        return

    placement_bounds = get_placement_bounds(world, obj, obj2)

    if display:
        placement_box = trimesh.primitives.Box(bounds=to_bounds3d(placement_bounds))
        world.scene.add_geometry(placement_box.as_outline())
        world.show()

    (x_lower, y), (x_upper, _) = placement_bounds
    if x_lower >= x_upper:
        placement = center_placement(world, obj, obj2)
        yield placement
        return
    while True:
        if (p_valid is not None) and (random.random() > p_valid):
            x = np.nan
        else:
            x = np.random.uniform(low=x_lower, high=x_upper)
        placement = np.array([x, y])
        yield placement


def test_placement(world: World2D, obj: str, placement: Position2, support: str) -> bool:
    placement_bounds = get_placement_bounds(world, obj, support)
    placement_bounds = extend_bounds(placement_bounds, extension=1e-2)
    return bounds_contain_vector(placement_bounds, placement)


def sample_ik(
    robot: str,
    obj: str,
    grasp: Position2,
    placement: Position2,
    reachable_objects: Optional[Dict[str, List[str]]] = None,
    initial_failures: Optional[int] = 0,
    p_success: Optional[float] = None,
) -> Iterator[Optional[Position2]]:
    if np.isnan(grasp).any() or np.isnan(placement).any():
        return None

    if (reachable_objects is not None) and (obj not in reachable_objects.get(robot, [])):
        return None

    if initial_failures is not None:
        for _ in range(initial_failures):
            yield None

    if (p_success is not None) and (random.random() > p_success):
        return None
    conf = placement - grasp
    yield conf


def plan_motion(
    robot: str, conf1: Position2, conf2: Position2, step_distance: float = 5e-2, **kwargs: Any
) -> Optional[Path]:
    if np.isnan(conf1).any() or np.isnan(conf2).any():
        return None
    approach = np.array([0.0, 0.1])
    waypoints = [conf1, conf1 + approach, conf2 + approach, conf2]
    path = list(interpolate_waypoints(waypoints, step_distance=step_distance, **kwargs))
    traj = Path(path)
    return traj


def get_duration(traj: Path, speed: float = 1.0) -> float:
    duration = traj.distance / speed
    return duration


def get_supporters(world: World2D) -> Dict[str, str]:
    objects = world.get_category_names(categories=["object"])
    supports = world.get_category_names(categories=["table", "region"])
    supports.sort(key=lambda s: world.get_geometry(s).area)

    object_supports = {}
    for obj in objects:
        placement = world.get_conf(obj)
        for support in supports:
            if test_placement(world, obj, placement, support):
                object_supports[obj] = support
                break
    return object_supports


def sample_state(world: World2D) -> State2D:
    object_supports = get_supporters(world)
    placement_generators = [
        cycle(sample_placements(world, obj, support)) for obj, support in object_supports.items()
    ]
    while True:
        for i, obj in enumerate(object_supports):
            placement = next(placement_generators[i])
            world.set_conf(obj, placement)
        break

    return world.current_state()
