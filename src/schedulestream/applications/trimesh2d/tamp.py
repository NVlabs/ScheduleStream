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
from itertools import cycle, product
from typing import Any, Iterator, List, Optional

# Third Party
import numpy as np

# NVIDIA
from schedulestream.algorithm.solver import solve
from schedulestream.algorithm.stream.common import StreamSolution
from schedulestream.algorithm.stream.satisfier import satisfy_streams
from schedulestream.algorithm.stream.solver import STREAM_ALGORITHMS
from schedulestream.algorithm.temporal import TimedPlan
from schedulestream.applications.blocksworld.visualize import create_table_scene
from schedulestream.applications.trimesh2d.geometry import create_box, pose_from_position
from schedulestream.applications.trimesh2d.samplers import get_distance, test_body_body_collision
from schedulestream.applications.trimesh2d.streams import (
    get_duration,
    get_supporters,
    plan_motion,
    sample_grasps,
    sample_ik,
    sample_placements,
)
from schedulestream.applications.trimesh2d.utils import (
    COLORS,
    get_video_path,
    save_frames,
    set_seed,
)
from schedulestream.applications.trimesh2d.world import State, State2D, World2D
from schedulestream.common.utils import (
    EPSILON,
    INF,
    SEPARATOR,
    get_pairs,
    profiler,
    safe_max,
    safe_min,
    select,
)
from schedulestream.language.action import Action
from schedulestream.language.anonymous import rename_anonymous
from schedulestream.language.connective import Conjunction, CumulativeSequence
from schedulestream.language.constraint import Constraint
from schedulestream.language.durative import DurativeAction
from schedulestream.language.expression import Formula
from schedulestream.language.function import Evaluation, Function, NumericFunction
from schedulestream.language.generator import (
    batch_fn_from_list_gen_fn,
    from_unary_fn,
    from_unary_gen_fn,
)
from schedulestream.language.lazy import LazyOutput
from schedulestream.language.predicate import Predicate, Type
from schedulestream.language.problem import Problem
from schedulestream.language.stream import Stream

Robot = Type()
Body = Type()

Conf = Predicate(["?robot", "?conf"], condition=Robot("?robot"))
Traj = Predicate(["?robot", "?traj"], condition=Robot("?robot"))

Graspable = Predicate(["?obj"], condition=Body("?obj"))
Placeable = Predicate(["?obj", "?obj2"], condition=Body("?obj") & Body("?obj2"))

Grasp = Predicate(["?obj", "?grasp"], condition=Graspable("?obj"))
Placement = Predicate(["?obj", "?placement"], condition=Body("?obj"))

SupportedPlacement = Predicate(
    ["?obj1", "?placement1", "?obj2"],
    condition=Placeable("?obj1", "?obj2") & Placement("?obj1", "?placement1"),
)

Kin = Predicate(
    ["?robot", "?conf", "?obj", "?grasp", "?placement"],
    condition=Grasp("?obj", "?grasp") & Placement("?obj", "?placement") & Conf("?robot", "?conf"),
)

Motion = Predicate(
    ["?robot", "?conf1", "?traj", "?conf2"],
    condition=Conf("?robot", "?conf1") & Conf("?robot", "?conf2") & Traj("?robot", "?traj"),
)


ObjState = Function(["?obj"], condition=Body("?obj"))
Attached = Function(["?obj"], condition=Body("?obj"))

RobotState = Function(["?robot"], condition=Conf("?robot", "?output"))
Holding = Function(["?robot"], condition=Robot("?robot"))
Held = Predicate(["?obj"], condition=Body("?obj"))

rename_anonymous(locals())


def create_world(
    num_robots: int = 1,
    num_objects: int = 1,
    num_regions: int = 1,
    size: float = 0.1,
    width: float = 1.0,
) -> World2D:
    robots = [f"robot{i}" for i in range(num_robots)]
    objects = [f"object{i}" for i in range(num_objects)]

    scene = create_table_scene(robots, objects)

    table_x2 = width - size
    region_width = width / 4
    for i in range(num_regions):
        region = create_box(
            name=f"region{i}", width=region_width, depth=size, height=size, color=COLORS["green"]
        )
        scene.add_geometry(
            region,
            transform=pose_from_position(
                [table_x2 - region_width / 2 - region_width * i, -size, 1e-3]
            ),
        )

    world = World2D(scene)

    return world


def create_initial(state: State2D) -> List[Evaluation]:
    world = state.world
    robots = world.get_category_names(categories=["robot"])
    objects = world.get_category_names(categories=["object"])
    supports = world.get_category_names(categories=["table"])

    initial = []
    for robot in robots:
        conf0 = state.get_conf(robot)
        initial.extend(
            [
                RobotState(robot) <= conf0,
                Holding(robot) <= None,
            ]
        )

    for obj in objects:
        placement0 = state.get_conf(obj)
        initial.extend(
            [
                Graspable(obj),
                Placement(obj, placement0),
                ObjState(obj) <= placement0,
                Attached(obj) <= None,
            ]
        )

    for obj, support in product(objects, supports):
        initial.append(Placeable(obj, support))
    return initial


def initial_from_goal(world: World2D, goal: Optional[Formula]) -> List[Evaluation]:
    initial = []
    if goal is None:
        return initial
    for formula in goal.clause:
        initial.extend(formula.condition.clause)
        if isinstance(formula, Evaluation):
            formula = formula.unwrap()
            if formula.function == Attached:
                (top_obj,) = formula.inputs
                bottom_obj = formula.output
                if bottom_obj in world.object_names:
                    initial.append(Placeable(top_obj, bottom_obj))
    return initial


def create_collision(world: World2D, **kwargs: Any) -> Predicate:
    Collision = Predicate(
        ["?obj1", "?placement1", "?obj2", "?placement2"],
        definition=partial(test_body_body_collision, world),
        **kwargs,
    )
    return Collision.rename(locals())


def create_actions(world: World2D, cost: float = EPSILON, collisions: bool = True) -> List[Action]:
    Distance = NumericFunction(
        ["?robot", "?conf1", "?conf2"],
        condition=Conf("?robot", "?conf1") & Conf("?robot", "?conf2"),
        definition=lambda r, q1, q2: get_distance(q1, q2),
    )
    Duration = NumericFunction(
        ["?robot", "?traj"],
        condition=Traj("?robot", "?traj"),
        definition=lambda r, t: get_duration(t),
        lazy=True,
    )
    move = DurativeAction(
        parameters=["?robot", "?conf1", "?conf2", "?traj"],
        start_condition=RobotState("?robot") == "?conf1",
        start_effect=RobotState("?robot") == "?traj",
        over_condition=Motion("?robot", "?conf1", "?traj", "?conf2"),
        end_effect=RobotState("?robot") <= "?conf2",
        min_duration=Duration("?robot", "?traj"),
    )

    pick = Action(
        parameters=["?robot", "?conf", "?obj", "?placement", "?grasp"],
        precondition=Kin("?robot", "?conf", "?obj", "?grasp", "?placement")
        & (ObjState("?obj") == "?placement")
        & (Holding("?robot") == None)
        & (RobotState("?robot") == "?conf"),
        effect=(Attached("?obj") <= "?robot")
        & (ObjState("?obj") <= "?grasp")
        & (Holding("?robot") <= "?obj")
        & Held("?obj"),
    )

    preconditions = []

    Collision = create_collision(world, lazy=True)
    if collisions:
        objects = world.get_category_names(categories=["obj"])
        preconditions.extend(
            ~Collision("?obj", "?placement", obj, ObjState(obj)) for obj in objects
        )

    place = Action(
        parameters=["?robot", "?conf", "?obj", "?grasp", "?placement", "?obj2"],
        precondition=Kin("?robot", "?conf", "?obj", "?grasp", "?placement")
        & SupportedPlacement("?obj", "?placement", "?obj2")
        & (ObjState("?obj") == "?grasp")
        & (Holding("?robot") == "?obj")
        & (RobotState("?robot") == "?conf")
        & Conjunction(*preconditions),
        effect=(Attached("?obj") <= "?obj2")
        & (ObjState("?obj") <= "?placement")
        & (Holding("?robot") <= None)
        & ~Held("?obj"),
        cost=cost,
    )

    rename_anonymous(locals())

    return [move, pick, place]


def create_placement_stream(world: World2D, **kwargs: Any) -> None:
    return SupportedPlacement.stream(
        conditional_generator=from_unary_gen_fn(partial(sample_placements, world, **kwargs)),
        inputs=["?obj1", "?obj2"],
    )


def create_streams(
    world: World2D,
    batch_size: Optional[int] = 16,
    reachable: bool = False,
    reverse: bool = False,
) -> List[Stream]:
    robots = world.get_category_names(categories=["robot"])
    objects = world.get_category_names(categories=["object"])
    if reverse:
        objects.reverse()
    if reachable:
        reachable_objects = {robot: [obj] for robot, obj in zip(cycle(robots), objects)}
    else:
        reachable_objects = None

    streams = [
        Grasp.stream(conditional_generator=from_unary_gen_fn(sample_grasps), inputs=["?obj"]),
        create_placement_stream(world),
        Motion.stream(
            conditional_generator=from_unary_fn(plan_motion),
            inputs=["?robot", "?conf1", "?conf2"],
            priority=-INF,
        ),
    ]
    if batch_size is None:
        streams.append(
            Kin.stream(
                conditional_generator=from_unary_gen_fn(
                    partial(sample_ik, reachable_objects=reachable_objects)
                ),
            )
        )
    else:
        streams.append(
            Kin.stream(
                conditional_generator=batch_fn_from_list_gen_fn(from_unary_gen_fn(sample_ik)),
                inputs=["?robot", "?obj", "?grasp", "?placement"],
                batch_size=batch_size,
            )
        )
    return streams


def create_goal(
    state: State2D, problem: str, num_goals: Optional[int] = 1, reset: bool = True, **kwargs: Any
) -> Formula:
    world = state.world

    robots = select(world.get_category_names(categories=["robot"]), num=num_goals, **kwargs)
    objects = select(world.get_category_names(categories=["object"]), num=num_goals, **kwargs)

    [table] = select(world.get_category_names(categories=["table"]), num=1, **kwargs)
    [region] = select(world.get_category_names(categories=["region"]), num=1, **kwargs)

    goals = []
    if problem == "holding":
        goals.extend(Holding(robot) == obj for robot, obj in zip(robots, objects))
    elif problem == "held":
        goals.extend(Held(obj) for obj in objects)
    elif problem == "table":
        goals.extend(Attached(obj) == table for obj in objects)
    elif problem == "region":
        goals.extend(Attached(obj) == region for obj in objects)
    elif problem == "stack":
        objects = world.get_category_names(categories=["object"])
        stacks = select(get_pairs(objects), num=num_goals, **kwargs)
        goals.extend(Attached(obj) == obj2 for obj, obj2 in stacks)
    else:
        raise ValueError(problem)
    goal = Conjunction(*goals)

    if not reset:
        return goal

    reset_goals = [RobotState(robot) == state.get_conf(robot) for robot in robots]
    reset_goal = Conjunction(*reset_goals)
    return CumulativeSequence([goal, reset_goal])


def schedule_states(
    state: State2D, timed_plan: Optional[TimedPlan], time_step: float = 1 / 30
) -> Iterator[State]:
    state.set()
    world = state.world
    if timed_plan is None:
        return
    start_time = safe_min(timed_action.start for timed_action in timed_plan)
    end_time = safe_max(timed_action.end for timed_action in timed_plan)
    for t1 in np.arange(start_time, end_time, time_step):
        t2 = t1 + time_step
        for timed_action in timed_plan:
            if (timed_action.end < t1) or (timed_action.start > t2):
                continue
            action = timed_action.action
            arguments = action.unwrap_arguments()
            if action.name == "move":
                robot, q1, q2, traj = arguments
                if isinstance(traj, LazyOutput):
                    traj = traj.substitute().value
                fraction = (t1 - timed_action.start) / timed_action.duration
                if 0.0 < fraction < 1.0:
                    conf = traj.sample(fraction)
                    world.set_conf(robot, conf)
            elif action.name == "pick":
                robot, q, obj, g, p = arguments
                world.attach(obj, robot)
            elif action.name == "place":
                robot, q, obj, g, p, support = arguments
                world.detach(obj)
            else:
                raise ValueError(action.name)
        yield world.current_state()


def satisfy_state(world: World2D, **kwargs: Any) -> State2D:
    placement_stream = create_placement_stream(world, p_valid=None)
    object_supports = get_supporters(world)
    object_placements = {obj: f"p{i}" for i, obj in enumerate(object_supports)}

    streams = []
    for obj, obj2 in object_supports.items():
        streams.append(placement_stream(obj, obj2)(object_placements[obj]))

    Collision = create_collision(world)
    for obj1, obj2 in itertools.combinations(object_placements, r=2):
        collision = Collision(obj1, object_placements[obj1], obj2, object_placements[obj2])
        streams.append(Constraint(~collision))

    mappings = satisfy_streams(streams, **kwargs)
    assert mappings
    mapping = mappings[0]
    mapping = {key.unwrap(): value.unwrap() for key, value in mapping.items()}
    print(f"Mapping:", mapping)
    for obj, placement in object_placements.items():
        placement = mapping[placement]
        world.set_conf(obj, placement)
    return world.current_state()


def tamp(
    problem: str = "holding",
    robots: int = 2,
    objects: int = 2,
    goals: int = 2,
    collisions: bool = True,
    sequential: bool = False,
    anytime: bool = False,
    max_time: float = INF,
    seed: Optional[int] = 0,
    video: Optional[str] = None,
    debug: bool = False,
    profile: bool = False,
    visualize: bool = False,
    **kwargs: Any,
) -> List[StreamSolution]:
    video_path = get_video_path(video, name=problem)
    np.set_printoptions(precision=2, suppress=True, threshold=3)
    set_seed(seed=seed)

    world = create_world(num_robots=robots, num_objects=objects)
    state = satisfy_state(world)

    initial = create_initial(state)
    goal = create_goal(state, problem, num_goals=goals, reset=True)
    initial.extend(initial_from_goal(world, goal))
    problem = Problem(
        initial=initial,
        goal=goal,
        actions=create_actions(world, collisions=collisions),
        streams=create_streams(world),
    )
    if debug:
        problem = problem.lazy_clone()
    print(problem)

    with profiler(field="cumtime" if profile else None, num=25):
        solutions = solve(
            problem,
            sequential=sequential,
            max_solutions=INF if anytime else 1,
            success_cost=0 if anytime else INF,
            max_time=max_time,
            search_time=2.0,
            lazy=True,
            heuristic_fn="relaxed",
            successor_fn="offline",
            weight=5,
            **kwargs,
        )
    if (not visualize and not video_path) or debug:
        return solutions

    if not solutions:
        world.show()
        return solutions

    print(SEPARATOR)
    solution = solutions[0]
    plan = solution.plan
    states = list(schedule_states(state, plan))

    frames = world.animate(states, record=video_path)
    save_frames(frames, video_path)

    return solutions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--problem",
        type=str,
        default="holding",
        choices=["holding", "held", "table", "region", "stack"],
    )
    parser.add_argument("-r", "--robots", type=int, default=2)
    parser.add_argument("-o", "--objects", type=int, default=2)
    parser.add_argument("-g", "--goals", type=int, default=2)
    parser.add_argument("-c", "--cfree", action="store_true", help="Enables collision free")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("-a", "--anytime", action="store_true")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="focused",
        choices=list(STREAM_ALGORITHMS),
        help="The algorithm.",
    )
    parser.add_argument("-t", "--max-time", default=10, type=float)
    parser.add_argument(
        "-s", "--seed", nargs="?", const=None, default=0, type=int, help="The random seed."
    )
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Enables profiling")
    parser.add_argument("-v", "--video", default=None, type=str)
    args = parser.parse_args()
    print("Args:", args)
    tamp(
        problem=args.problem,
        robots=args.robots,
        objects=args.objects,
        goals=args.goals,
        collisions=not args.cfree,
        sequential=args.sequential,
        anytime=args.anytime,
        max_time=args.max_time,
        seed=args.seed,
        debug=args.debug,
        profile=args.profile,
        video=args.video,
        visualize=True,
    )


if __name__ == "__main__":
    main()
