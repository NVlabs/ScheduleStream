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
import itertools
import math
import random
from functools import cached_property
from typing import Any, Iterable, Iterator, List, Tuple

# Third Party
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.kdtree import KDTree

# NVIDIA
from schedulestream.applications.trimesh2d.geometry import (
    Path,
    Pose,
    Position,
    Position2,
    pose_from_conf,
)
from schedulestream.applications.trimesh2d.world import World
from schedulestream.common.utils import INF, flatten, get_pairs, negate_test


class Traj(list):
    @property
    def confs(self) -> List[Position2]:
        return list(self)

    def interpolate(self, **kwargs: Any) -> Iterator[Position2]:
        return interpolate_waypoints(self, **kwargs)

    @cached_property
    def curve(self) -> interp1d:
        # NVIDIA
        from schedulestream.applications.blocksworld.visualize import get_curve

        return get_curve(self)

    def sample(self, t: float) -> Position2:
        assert self.curve.x[0] <= t <= self.curve.x[-1], (self.curve.x[0], t, self.curve.x[-1])
        return self.curve(t)

    @property
    def start(self) -> Position2:
        return self[0]

    @property
    def end(self) -> Position2:
        return self[-1]

    @property
    def edge(self) -> Tuple[Position2, Position2]:
        return self.start, self.end

    @property
    def distance(self) -> float:
        return sum(get_distance(*pair) for pair in get_pairs(self))

    def reverse(self) -> "Traj":
        return Traj(self[::-1])

    def __str__(self) -> str:
        prefix = self.__class__.__name__.lower()[0]
        index = id(self) % 1000
        return f"{prefix}{index}"

    __repr__ = __str__


def to_bounds3d(bounds2d: Tuple[Position2, Position2], z: float = 0.0) -> Tuple[Position, Position]:
    return np.hstack([bounds2d, z * np.ones([2, 1])])


def get_distance(position1: Position2, position2: Position2, **kwargs: Any) -> float:
    return np.linalg.norm(np.array(position2) - np.array(position1), **kwargs)


def sample_interval(lower: float, upper: float) -> Iterable[float]:
    if lower > upper:
        return
    if lower == upper:
        yield lower
        return
    while True:
        yield random.uniform(lower, upper)


def interpolate(
    position1: Position2, position2: Position2, step_distance: float = 1e-2
) -> Iterator[Position2]:
    steps = int(math.ceil(get_distance(position1, position2) / step_distance))
    for w in np.linspace(start=0.0, stop=1.0, num=steps, endpoint=True):
        position = (1.0 - w) * np.array(position1) + w * np.array(position2)
        yield position


def interpolate_waypoints(waypoints: List[Position2], **kwargs: Any) -> Iterator[Position2]:
    if not waypoints:
        return
    yield waypoints[0]
    for waypoint1, waypoint2 in zip(waypoints[:-1], waypoints[1:]):
        for i, waypoint in enumerate(interpolate(waypoint1, waypoint2, **kwargs)):
            if i != 0:
                yield waypoint


def sample_confs(world: World, region: str) -> Iterator[Position2]:
    region = world.get_object(region)
    region_bounds = region.bounds
    while True:
        position = np.random.uniform(*region_bounds)
        conf = position[:2]
        yield conf


def test_contained(world: World, conf: Position2, region: str) -> bool:
    region = world.get_object(region)
    region_bounds2d = region.bounds[:, :2]

    return (
        np.less_equal(region_bounds2d[0], conf).all()
        and np.less_equal(conf, region_bounds2d[1]).all()
    )


def test_body_body_collision(
    world: World,
    body1: str,
    conf1: Position2,
    body2: str,
    conf2: Position2,
    debug: bool = False,
) -> bool:
    if body1 == body2:
        return False
    world.set_pose(body1, pose_from_conf(conf1))
    world.set_pose(body2, pose_from_conf(conf2))
    collision = world.check_pair(body1, body2)
    if debug:
        print(f"Body1: {body1} | Body2: {body2} | Collision: {collision}")
        world.show()
    return collision


def test_conf_collision(world: World, robot: str, conf: Position2, display: bool = False) -> bool:
    if len(world.collision_frames) <= 1:
        return False
    world.set_pose(robot, pose_from_conf(conf))
    movable_bodies = [body.name for body in world.movable_bodies]
    colliders = set(world.check_colliding(robot))
    collision = bool(colliders - set(movable_bodies))
    if display and collision:
        world.show()
    return collision


def test_traj_collision(world: World, robot: str, traj: Path) -> bool:
    return any(test_conf_collision(world, robot, conf) for conf in traj)


def test_robot_collision(
    world: World, robot1: str, traj1: Traj, robot2: str, conf2: Position2
) -> bool:
    if robot1 == robot2:
        return False
    path1 = list(traj1)
    path2 = [conf2]
    for conf1, conf2 in itertools.product(path1, path2):
        if test_body_body_collision(world, robot1, conf1, robot2, conf2):
            return True
    return False


test_collision_free = negate_test(test_conf_collision)


def compute_traj(conf1: Position2, conf2: Position2, **kwargs: Any) -> Traj:
    path = interpolate(conf1, conf2, **kwargs)
    return Traj(path)


def test_motion(
    world: World,
    robot: str,
    conf1: Position2,
    conf2: Position2,
    max_distance: float = np.inf,
    **kwargs: Any,
) -> bool:
    if conf1 is conf2:
        return False
    distance = get_distance(conf1, conf2)
    if distance > max_distance:
        return False
    traj = compute_traj(conf1, conf2, **kwargs)
    return test_traj_collision(world, robot, traj)


def compute_distance_edges(
    confs: List[Position2], max_distance: float = INF
) -> List[Tuple[np.array, np.array]]:
    edges = []
    for q1, q2 in itertools.combinations(confs, r=2):
        distance = get_distance(q2, q1)
        if distance <= max_distance:
            edges.append((q1, q2))
    return edges


def compute_degree_edges(
    confs: List[Position2], degree: int = 4
) -> List[Tuple[np.array, np.array]]:
    edges = []
    kdtree = KDTree(confs)
    for i, q in enumerate(confs):
        _, indices = kdtree.query(q, k=degree + 1)
        for j in indices:
            if (i != j) and (j <= len(confs) - 1):
                edges.append((confs[i], confs[j]))
    return edges


def compute_max_distance(confs: List[Position2], fraction: float = 0.9) -> float:
    distances = sorted(get_distance(*pair) for pair in itertools.combinations(confs, r=2))
    index = int(fraction * len(distances))
    max_distance = distances[index]
    return max_distance
