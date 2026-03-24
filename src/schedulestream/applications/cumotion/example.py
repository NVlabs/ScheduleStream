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
from functools import partial
from typing import Any, List, Optional

# Third Party
import numpy as np

# NVIDIA
from schedulestream.algorithm.solver import solve
from schedulestream.algorithm.stream.common import StreamSolution
from schedulestream.applications.cumotion.command import Conf as ConfWrapper
from schedulestream.applications.cumotion.command import animate_commands
from schedulestream.applications.cumotion.generators import (
    grasp_sampler,
    motion_sampler,
    pick_sampler,
    place_sampler,
    placement_sampler,
)
from schedulestream.applications.cumotion.problems import create_franka_problem, sample_placements
from schedulestream.applications.cumotion.world import World
from schedulestream.applications.trimesh2d.utils import get_video_path, set_seed
from schedulestream.common.utils import SEPARATOR, get_pairs
from schedulestream.language.action import Action
from schedulestream.language.anonymous import rename_anonymous
from schedulestream.language.argument import OUTPUT, unwrap_arguments
from schedulestream.language.connective import Conjunction
from schedulestream.language.durative import DurativeAction
from schedulestream.language.function import Function, NumericFunction
from schedulestream.language.generator import false_test, from_gen_fn
from schedulestream.language.predicate import Predicate, Type
from schedulestream.language.problem import Problem
from schedulestream.language.stream import Stream

Conf = Predicate(["?conf"])
Traj = Predicate(["?traj"])
At = Function([], condition=Conf(OUTPUT))
Holding = Function([])
Moved = Predicate([])

Object = Type()
AtPose = Function(["?obj"], condition=Object("?obj"))
Attached = Function(["?obj"], condition=Object("?obj"))
Supporting = Predicate(["?obj"], condition=Object("?obj"))

Graspable = Predicate(["?obj"], condition=Object("?obj"))
Placeable = Predicate(["?obj", "?obj2"], condition=Object("?obj") & Object("?obj2"))

Placement = Predicate(["?obj", "?placement"], condition=Object("?obj"))
Grasp = Predicate(["?obj", "?grasp"], condition=Graspable("?obj"))
Supported = Predicate(
    ["?obj1", "?placement1", "?obj2", "?placement2"],
    condition=Placeable("?obj1", "?obj2")
    & Placement("?obj1", "?placement1")
    & Placement("?obj2", "?placement2"),
)
Motion = Predicate(
    ["?conf1", "?conf2", "?traj"],
    condition=Conf("?conf1") & Conf("?conf2") & Traj("?traj"),
)
Pick = Predicate(
    ["?obj", "?grasp", "?placement", "?conf1", "?conf2", "?traj"],
    condition=Grasp("?obj", "?grasp")
    & Placement("?obj", "?placement")
    & Conf("?conf1")
    & Conf("?conf2")
    & Traj("?traj"),
)
Place = Predicate(
    ["?obj", "?grasp", "?placement", "?conf1", "?conf2", "?traj"],
    condition=Grasp("?obj", "?grasp")
    & Placement("?obj", "?placement")
    & Conf("?conf1")
    & Conf("?conf2")
    & Traj("?traj"),
)

Duration = NumericFunction(
    ["?traj"],
    condition=Traj("?traj"),
    definition=lambda traj: traj.duration,
    lazy=True,
)
ObjCollision = Predicate(
    ["?traj", "?obj", "?pose"],
    condition=Traj("?traj"),
    definition=false_test,
    lazy=True,
)

rename_anonymous(locals())


def create_sequential_actions(world: World, collisions: bool = True) -> List[Action]:
    object_colliders = world.movable_objects if collisions else []
    obj_conditions = [~ObjCollision("?traj", obj, AtPose(obj)) for obj in object_colliders]
    collision_condition = Conjunction(*obj_conditions)

    move = Action(
        parameters="?conf1 ?conf2 ?traj",
        precondition=Motion("?conf1", "?conf2", "?traj")
        & (At() == "?conf1")
        & ~Moved()
        & collision_condition,
        effect=(At() <= "?conf2") & Moved(),
        cost=Duration("?traj"),
    )
    pick = Action(
        parameters="?obj ?grasp ?placement ?conf1 ?conf2 ?traj",
        precondition=Pick("?obj", "?grasp", "?placement", "?conf1", "?conf2", "?traj")
        & (At() == "?conf1")
        & (AtPose("?obj") == "?placement")
        & ~Supporting("?obj")
        & (Holding() == None),
        effect=(At() <= "?conf2")
        & (Holding() <= "?obj")
        & (AtPose("?obj") <= "?grasp")
        & (Attached("?obj") <= None)
        & ~Moved(),
        cost=Duration("?traj"),
    )
    place = Action(
        parameters="?obj ?grasp ?placement ?o2 ?p2 ?conf1 ?conf2 ?traj",
        precondition=Place("?obj", "?grasp", "?placement", "?conf1", "?conf2", "?traj")
        & Supported("?obj", "?placement", "?o2", "?p2")
        & (At() == "?conf1")
        & (AtPose("?obj") == "?grasp")
        & (AtPose("?o2") == "?p2"),
        effect=(At() <= "?conf2")
        & (Holding() <= None)
        & (AtPose("?obj") <= "?placement")
        & Supporting("?o2")
        & (Attached("?obj") <= "?o2")
        & ~Moved(),
        cost=Duration("?traj"),
    )
    rename_anonymous(locals())
    return [move, pick, place]


def create_durative_actions(world: World, collisions: bool = True) -> List[Action]:
    object_colliders = world.movable_objects if collisions else []
    obj_conditions = [~ObjCollision("?traj", obj, AtPose(obj)) for obj in object_colliders]
    collision_condition = Conjunction(*obj_conditions)

    move = DurativeAction(
        parameters="?conf1 ?conf2 ?traj",
        start_condition=(At() == "?conf1") & ~Moved(),
        start_effect=At() <= "?traj",
        over_condition=Motion("?conf1", "?conf2", "?traj") & collision_condition,
        end_effect=(At() <= "?conf2") & Moved(),
        min_duration=Duration("?traj"),
    )
    pick = DurativeAction(
        parameters="?obj ?grasp ?placement ?conf1 ?conf2 ?traj",
        start_condition=(At() == "?conf1") & (AtPose("?obj") == "?placement") & ~Supporting("?obj"),
        start_effect=(At() <= "?traj") & (AtPose("?obj") <= None),
        over_condition=Pick("?obj", "?grasp", "?placement", "?conf1", "?conf2", "?traj")
        & (Holding() == None),
        end_effect=(At() <= "?conf2")
        & (Holding() <= "?obj")
        & (AtPose("?obj") <= "?grasp")
        & (Attached("?obj") <= None)
        & ~Moved(),
        min_duration=Duration("?traj"),
    )
    place = DurativeAction(
        parameters="?obj ?grasp ?placement ?o2 ?p2 ?conf1 ?conf2 ?traj",
        start_condition=(At() == "?conf1")
        & (AtPose("?obj") == "?grasp")
        & (AtPose("?o2") == "?p2"),
        start_effect=(At() <= "?traj") & (AtPose("?obj") <= None) & Supporting("?o2"),
        over_condition=Place("?obj", "?grasp", "?placement", "?conf1", "?conf2", "?traj")
        & Supported("?obj", "?placement", "?o2", "?p2"),
        end_effect=(At() <= "?conf2")
        & (Holding() <= None)
        & (AtPose("?obj") <= "?placement")
        & (Attached("?obj") <= "?o2")
        & ~Moved(),
        min_duration=Duration("?traj"),
    )
    rename_anonymous(locals())
    return [move, pick, place]


def create_streams(
    world: World,
    collisions: bool = True,
) -> List[Stream]:
    return [
        Grasp.stream(
            conditional_generator=from_gen_fn(grasp_sampler),
            inputs=["?obj"],
        ),
        Supported.stream(
            conditional_generator=from_gen_fn(
                partial(placement_sampler, world, collisions=collisions)
            ),
            inputs=["?obj1", "?obj2", "?placement2"],
        ),
        Motion.stream(
            conditional_generator=from_gen_fn(
                partial(motion_sampler, world, collisions=collisions)
            ),
            inputs=["?conf1", "?conf2"],
            context_functions=[ObjCollision],
        ),
        Pick.stream(
            conditional_generator=from_gen_fn(partial(pick_sampler, world, collisions=collisions)),
            inputs=["?obj", "?grasp", "?placement"],
        ),
        Place.stream(
            conditional_generator=from_gen_fn(partial(place_sampler, world, collisions=collisions)),
            inputs=["?obj", "?grasp", "?placement"],
        ),
    ]


def example(
    task: str = "hold",
    num: int = 2,
    durative: bool = False,
    collisions: bool = True,
    debug: bool = False,
    seed: Optional[int] = 0,
    video: Optional[str] = None,
    **kwargs: Any,
) -> List[StreamSolution]:
    np.set_printoptions(precision=2, suppress=True, threshold=3)
    set_seed(seed=seed)

    world = create_franka_problem(num_cubes=num)
    sample_placements(world)
    if not collisions:
        world.disable_objects()

    if task == "hold":
        goals = [Holding() <= obj for arm, obj in zip(world.arms, world.movable_objects)]
    elif task == "stack":
        goals = [Attached(obj1) == obj2 for obj1, obj2 in get_pairs(world.movable_objects)]
    else:
        raise ValueError(task)

    conf = ConfWrapper(world.configuration)
    goals.append(At() == conf)

    initial = [
        At() <= conf,
        Holding() <= None,
    ]
    for obj in world.movable_objects:
        placement = world.get_pose(obj)
        initial.extend(
            [
                Placement(obj, placement),
                AtPose(obj) <= placement,
                Attached(obj) <= None,
                Graspable(obj),
            ]
        )
    for goal in goals:
        if goal.function == Attached:
            (top_obj,) = unwrap_arguments(goal.inputs)
            bottom_obj = goal.output.unwrap()
            initial.append(Placeable(top_obj, bottom_obj))

    if durative:
        actions = create_durative_actions(world, collisions=collisions)
    else:
        actions = create_sequential_actions(world, collisions=collisions)
    problem = Problem(
        initial=initial,
        goal=Conjunction(*goals),
        actions=actions,
        streams=create_streams(world, collisions=collisions),
    )
    if debug:
        problem = problem.lazy_clone(parent=False)

    print(SEPARATOR)

    solutions = solve(
        problem,
        sequential=True,
        lazy=True,
        heuristic_fn="relaxed",
        successor_fn="offline",
        weight=5,
        satisfy_time=1.0,
        num_reschedule=1,
        **kwargs,
    )
    if not solutions or debug:
        return solutions

    solution = solutions[0]
    solution.dump()
    plan = solution.plan
    commands = [action.unwrap_arguments()[-1] for action in plan]

    video_path = get_video_path(video, name=f"{task}{num}")
    animate_commands(world, commands, video_path=video_path)

    return solutions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        default="stack",
        type=str,
        choices=["hold", "stack"],
        help="The task name.",
    )
    parser.add_argument("-n", "--num", default=2, type=int, help="The number of objects.")
    parser.add_argument(
        "-s", "--seed", nargs="?", const=None, default=0, type=int, help="The random seed."
    )
    parser.add_argument(
        "-c", "--cfree", action="store_true", help="Disables collisions (collision free)."
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Run with debug streams")
    parser.add_argument(
        "-v", "--video", type=str, nargs="?", const="", default=None, help="Path to save a video."
    )
    args = parser.parse_args()
    print("Args:", args)
    example(
        task=args.task,
        num=args.num,
        collisions=not args.cfree,
        debug=args.debug,
        seed=args.seed,
        video=args.video,
    )


if __name__ == "__main__":
    main()
