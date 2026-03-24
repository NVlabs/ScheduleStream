#! /usr/bin/env python3
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
import argparse
import itertools
from functools import partial
from typing import Any, List, Optional

# Third Party
import numpy as np
import trimesh

# NVIDIA
from schedulestream.algorithm.finite.solver import FINITE_ALGORITHMS, solve_finite
from schedulestream.algorithm.temporal import TemporalSolution
from schedulestream.algorithm.utils import Plan
from schedulestream.applications.trimesh2d.geometry import (
    create_bounds,
    create_box,
    create_line,
    create_lines,
    create_path,
    create_scene,
    create_sphere,
    pose_from_conf,
    position_from_conf,
    to_pose,
)
from schedulestream.applications.trimesh2d.samplers import (
    compute_degree_edges,
    compute_traj,
    sample_confs,
    test_contained,
    test_robot_collision,
    test_traj_collision,
    to_bounds3d,
)
from schedulestream.applications.trimesh2d.utils import (
    COLORS,
    apply_alpha,
    get_video_path,
    save_frames,
    set_random_seed,
)
from schedulestream.applications.trimesh2d.world import State2D, World2D
from schedulestream.common.utils import INF, SEPARATOR, profiler
from schedulestream.language.action import Action
from schedulestream.language.anonymous import rename_anonymous
from schedulestream.language.argument import OUTPUT
from schedulestream.language.connective import Conjunction
from schedulestream.language.durative import DurativeAction
from schedulestream.language.function import Evaluation, Function, NumericFunction
from schedulestream.language.predicate import Predicate, Type
from schedulestream.language.problem import Problem

Robot = Type()
Conf = Type()
Traj = Type()
Region = Type()

Motion = Predicate({"?conf1": Conf, "?traj": Traj, "?conf2": Conf})

Contained = Predicate(
    ["?conf", "?region"],
    condition=Conf("?conf") & Region("?region"),
)

WorldCollision = Predicate(["?robot", "?traj"], lazy=True)

RobotCollision = Predicate(["?robot1", "?traj1", "?robot2", "?conf2"], lazy=True)

Distance = NumericFunction(
    ["?conf1", "?conf2"],
    condition=Conf("?conf1") & Conf("?conf2"),
    definition=lambda q1, q2: np.linalg.norm(q2 - q1),
    lazy=False,
)

AtConf = Function(["?robot"], condition=Robot("?robot") & Conf(OUTPUT))


Visited = Predicate(["?robot", "?region"])

rename_anonymous(locals())


def create_sequential_actions(world: World2D) -> List[Action]:
    robots = world.get_category_names(categories=["robot"])
    robot_condition = Conjunction(
        *[~RobotCollision("?robot", "?traj", robot2, AtConf(robot2)) for robot2 in robots]
    )
    move = Action(
        parameters=["?robot", "?conf1", "?conf2", "?traj"],
        precondition=Robot("?robot")
        & Motion("?conf1", "?traj", "?conf2")
        & (AtConf("?robot") == "?conf1")
        & ~WorldCollision("?robot", "?traj")
        & robot_condition,
        effect=(AtConf("?robot") <= "?conf2"),
        cost=Distance("?conf1", "?conf2"),
    )

    visit = Action(
        parameters=["?robot", "?conf", "?region"],
        precondition=Robot("?robot")
        & Contained("?conf", "?region")
        & (AtConf("?robot") == "?conf"),
        effect=Visited("?robot", "?region"),
    )
    rename_anonymous(locals())
    return [move, visit]


def create_durative_actions(world: World2D) -> List[Action]:
    robots = world.get_category_names(categories=["robot"])
    robot_condition = Conjunction(
        *[~RobotCollision("?robot", "?traj", robot2, AtConf(robot2)) for robot2 in robots]
    )
    move = DurativeAction(
        parameters=["?robot", "?conf1", "?conf2", "?traj"],
        over_condition=Robot("?robot") & Motion("?conf1", "?traj", "?conf2"),
        start_condition=(AtConf("?robot") == "?conf1"),
        start_effect=(AtConf("?robot") <= "?traj"),
        end_effect=(AtConf("?robot") <= "?conf2"),
        min_duration=1.0,
    )

    visit = Action(
        parameters=["?robot", "?conf", "?region"],
        precondition=Robot("?robot")
        & Contained("?conf", "?region")
        & (AtConf("?robot") == "?conf"),
        effect=Visited("?robot", "?region"),
        cost=0.0,
    )
    rename_anonymous(locals())
    return [move, visit]


def create_world(num_robots: int = 1, height: float = 0.1):
    cspace_bounds = (np.zeros(2), np.ones(2))
    cspace = create_bounds(
        to_bounds3d(cspace_bounds),
        color=apply_alpha(COLORS["grey"], alpha=0.1),
        name="cspace",
    )
    colors = ["blue", "green"]

    size = 0.25
    target0 = create_bounds(
        to_bounds3d([np.ones(2) - size * np.ones(2), np.ones(2)]),
        color=COLORS[colors[0]],
        name="target0",
    )
    target1 = create_bounds(
        to_bounds3d([(0.0, 1.0 - size), (size, 1.0)]),
        color=COLORS[colors[1]],
        name="target1",
    )

    obstacles = [
        create_box(
            width=0.25,
            depth=0.25,
            height=height,
            color=COLORS["red"],
            pose=to_pose(x=0.5, y=0.5),
        ),
    ]

    collision_scene = create_scene(obstacles)
    confs = [(0.0, 0.0), (1.0, 0.0)]
    for i in range(num_robots):
        robot = create_sphere(name=f"robot{i}", width=height, color=COLORS[colors[i]], movable=True)
        collision_scene.add_geometry(robot, transform=pose_from_conf(confs[i]))

    visual_scene = trimesh.Scene([cspace, target0, target1])
    world = World2D(collision_scene, visual_scene=visual_scene)

    return world


def create_region_goal(state: State2D, reverse: bool = False) -> Conjunction:
    world = state.world

    robots = world.get_category_names(categories=["robot"])
    targets = world.get_category_names(categories=["target"])
    if reverse:
        targets.reverse()
    targets = itertools.cycle(targets)

    goals = []
    for robot, target in zip(robots, targets):
        goals.append(Visited(robot, target))
    return Conjunction(*goals)


def create_swap_goal(state: State2D):
    world = state.world
    robots = world.get_category_names(categories=["robot"])
    confs = [state.get_conf(robot) for robot in robots]
    goals = []
    for i, robot in enumerate(robots):
        goals.append(AtConf(robot) == confs[i - 1])
    return Conjunction(*goals)


def create_roadmap(world: World2D, confs: List[np.ndarray]) -> List[Evaluation]:
    edges = compute_degree_edges(confs, degree=5)
    initial = []
    for q1, q2 in edges:
        t1 = compute_traj(q1, q2)
        t2 = t1.reverse()
        initial.extend([Motion(q1, t1, q2), Motion(q2, t2, q1)])
    lines = [list(map(position_from_conf, edge)) for edge in edges]
    world.scene.add_geometry(create_lines(lines, color=apply_alpha(COLORS["black"], alpha=0.05)))
    return initial


def create_initial(
    state: State2D, num_confs: int = 100, collisions: bool = True
) -> List[Evaluation]:
    world = state.world
    robots = world.get_category_names(categories=["robot"])

    cspace = "cspace"
    targets = world.get_category_names(categories=["target"])
    regions = [cspace] + targets

    confs = []
    initial = []
    for robot in robots:
        conf0 = state.get_conf(robot)
        confs.append(conf0)
        initial.append(AtConf(robot) <= conf0)

    confs.extend(itertools.islice(sample_confs(world, cspace), num_confs))
    initial.extend(create_roadmap(world, confs))

    for region in regions:
        for conf in confs:
            if test_contained(world, conf, region):
                initial.append(Contained(conf, region))

    def wrapped_test_collision(robot: str, traj: Traj) -> bool:
        collision = False
        if collisions:
            collision = test_traj_collision(world, robot, traj)
        line = list(map(position_from_conf, traj.edge))
        color = apply_alpha(COLORS["red" if collision else "green"], alpha=0.25)
        world.scene.add_geometry(create_line(line, color=color))
        return collision

    WorldCollision.definition = wrapped_test_collision
    RobotCollision.definition = partial(test_robot_collision, world)

    return initial


def draw_plan(state: State2D, plan: Optional[Plan], **kwargs: Any) -> None:
    world = state.world
    if plan is None:
        return
    robots = world.get_category_names(categories=["robot"])
    state.set()
    paths = {robot: [world.get_conf(robot)] for robot in robots}
    for i, action in enumerate(plan):
        if action.name != "move":
            continue
        robot, q1, q2, t = action.unwrap_arguments()
        paths[robot].append(q2)

    for path in paths.values():
        path3d = list(map(position_from_conf, path))
        world.scene.add_geometry(create_path(path3d, **kwargs))


def animate_plan(state: State2D, plan: Optional[Plan], **kwargs: Any) -> List[bytes]:
    world = state.world
    state.set()
    if plan is None:
        world.show()
        return []
    states = []
    for i, action in enumerate(plan):
        if action.name != "move":
            continue
        robot, q1, q2, t = action.unwrap_arguments()
        for conf in t[1:]:
            world.set_conf(robot, conf)
            states.append(world.current_state())
    return world.animate(states, **kwargs)


def motion(
    problem: str = "region",
    robots: int = 1,
    sequential: bool = True,
    collisions: bool = True,
    unit: bool = False,
    seed: Optional[int] = 0,
    profile: bool = False,
    video: Optional[str] = None,
    visualize: bool = False,
    **kwargs: Any,
) -> TemporalSolution:
    np.set_printoptions(precision=2, suppress=True, threshold=3)
    set_random_seed(seed)
    video_path = get_video_path(video, name=problem)

    world = create_world(num_robots=robots)
    state = world.current_state()

    initial = create_initial(state, collisions=collisions)
    if problem == "region":
        goal = create_region_goal(state)
    elif problem == "reverse_region":
        goal = create_region_goal(state, reverse=True)
    elif problem == "swap":
        goal = create_swap_goal(state)
    else:
        raise ValueError(f"Unknown problem: {problem}")

    if sequential:
        actions = create_sequential_actions(world)
    else:
        actions = create_durative_actions(world)

    problem = Problem(
        initial=initial,
        goal=goal,
        actions=actions,
    )
    if unit:
        problem.set_unit_costs()
    problem.dump()
    print(SEPARATOR)

    with profiler(field="cumtime" if profile else None, num=25):
        solution = solve_finite(problem, **kwargs)
    print(SEPARATOR)
    print("Goal:", problem.goal)
    solution.dump()

    if not visualize and (video_path is None):
        return solution

    draw_plan(state, solution.plan)
    frames = animate_plan(
        state, solution.plan, record=video_path, height=200 if video_path else 480
    )
    save_frames(frames, video_path=video_path)

    return solution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--problem", type=str, default="region", choices=["region", "reverse_region", "swap"]
    )
    parser.add_argument("-r", "--robots", type=int, default=1)
    parser.add_argument("-c", "--cfree", action="store_true", help="Enables collision free")
    parser.add_argument(
        "-a", "--algorithm", type=str, default="lazy", choices=list(FINITE_ALGORITHMS)
    )
    parser.add_argument("-w", "--weight", type=float, default=INF)
    parser.add_argument("-u", "--unit", action="store_true", help="TBD")
    parser.add_argument(
        "-s", "--seed", nargs="?", const=None, default=0, type=int, help="The random seed."
    )
    parser.add_argument("--profile", action="store_true", help="Enables profiling")
    parser.add_argument("-v", "--video", default=None, type=str, help="TBD")
    args = parser.parse_args()
    print("Args:", args)

    motion(
        problem=args.problem,
        robots=args.robots,
        collisions=not args.cfree,
        algorithm=args.algorithm,
        weight=args.weight,
        unit=args.unit,
        seed=args.seed,
        profile=args.profile,
        video=args.video,
        visualize=True,
    )


if __name__ == "__main__":
    main()
