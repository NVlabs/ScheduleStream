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
from typing import Any, Optional

# NVIDIA
from schedulestream.algorithm.finite.solver import FINITE_ALGORITHMS, solve_finite
from schedulestream.algorithm.utils import Solution
from schedulestream.applications.blocksworld.problems import (
    PROBLEMS,
    ArmEmpty,
    Clear,
    Holding,
    On,
    OnTable,
)
from schedulestream.applications.trimesh2d.utils import get_video_path, save_frames
from schedulestream.common.utils import INF, SEPARATOR, profiler
from schedulestream.language.action import Action
from schedulestream.language.anonymous import rename_anonymous

pickup = Action(
    parameters=["?a", "?ob"],
    precondition=Clear("?ob") & OnTable("?ob") & ArmEmpty("?a"),
    effect=Holding("?a", "?ob") & ~Clear("?ob") & ~OnTable("?ob") & ~ArmEmpty("?a"),
    language="Pick block ?ob off the table using arm ?a",
)

putdown = Action(
    parameters=["?a", "?ob"],
    precondition=Holding("?a", "?ob"),
    effect=Clear("?ob") & ArmEmpty("?a") & OnTable("?ob") & ~Holding("?a", "?ob"),
    language="Put block ?ob on the table using arm ?a",
)

stack = Action(
    parameters=["?a", "?ob", "?underob"],
    precondition=Clear("?underob") & Holding("?a", "?ob"),
    effect=ArmEmpty("?a")
    & Clear("?ob")
    & On("?ob", "?underob")
    & ~Clear("?underob")
    & ~Holding("?a", "?ob"),
    language="Stack block ?ob on block ?underob using arm ?a",
)

unstack = Action(
    parameters=["?a", "?ob", "?underob"],
    precondition=On("?ob", "?underob") & Clear("?ob") & ArmEmpty("?a"),
    effect=Holding("?a", "?ob")
    & Clear("?underob")
    & ~On("?ob", "?underob")
    & ~Clear("?ob")
    & ~ArmEmpty("?a"),
    language="Unstack block ?ob from block ?underob using arm ?a",
)

ACTIONS = [pickup, putdown, stack, unstack]

rename_anonymous(locals())


def add_arguments(
    parser: argparse.ArgumentParser, arms: int = 1, blocks: int = 3, weight: float = INF
):
    problem_names = list(PROBLEMS)
    parser.add_argument("-a", "--arms", type=int, default=arms)
    parser.add_argument("-b", "--blocks", type=int, default=blocks)
    parser.add_argument(
        "-p", "--problem", type=str, default=problem_names[0], choices=problem_names
    )
    parser.add_argument("--algorithm", type=str, default="online", choices=list(FINITE_ALGORITHMS))
    parser.add_argument("-s", "--sequential", action="store_true")
    parser.add_argument("-w", "--weight", type=float, default=weight)
    parser.add_argument("--profile", action="store_true", help="Enables profiling")
    parser.add_argument("-v", "--video", type=str, default=None)


def sequential(
    problem: str,
    arms: int,
    blocks: int,
    profile: bool = False,
    video: Optional[str] = None,
    visualize: bool = False,
    **kwargs: Any,
) -> Solution:
    video_path = get_video_path(video, name=problem)
    problem_fn = PROBLEMS[problem]
    problem = problem_fn(num_arms=arms, num_blocks=blocks)
    problem.actions = ACTIONS
    problem.set_unit_costs()
    problem.dump()

    with profiler(field="tottime" if profile else None, num=25):
        solution = solve_finite(problem, **kwargs)
    print(SEPARATOR)
    solution = solution.sequential
    print(solution)
    if solution.success:
        for i, action in enumerate(solution.plan):
            print(f"{i+1}) {action}: '{action.language}'")

    if not visualize and not video_path:
        return solution

    # NVIDIA
    from schedulestream.applications.blocksworld.visualize import visualize_plan

    frames = visualize_plan(
        problem,
        solution.plan,
        record=video_path,
    )
    save_frames(frames, video_path=video_path)
    return solution


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    print("Args:", args)

    sequential(
        args.problem,
        args.arms,
        args.blocks,
        algorithm=args.algorithm,
        sequential=args.sequential,
        weight=args.weight,
        profile=args.profile,
        visualize=True,
        video=args.video,
        verbose=True,
    )


if __name__ == "__main__":
    main()


"""
(define (domain blocksworld)
  (:requirements :strips :equality)
  (:predicates (clear ?x)
               (on-table ?x)
               (arm-empty)
               (holding ?x)
               (on ?x ?y))

  (:action pickup
    :parameters (?ob)
    :precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
    :effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob))
                 (not (arm-empty))))

  (:action putdown
    :parameters  (?ob)
    :precondition (and (holding ?ob))
    :effect (and (clear ?ob) (arm-empty) (on-table ?ob)
                 (not (holding ?ob))))

  (:action stack
    :parameters  (?ob ?underob)
    :precondition (and  (clear ?underob) (holding ?ob))
    :effect (and (arm-empty) (clear ?ob) (on ?ob ?underob)
                 (not (clear ?underob)) (not (holding ?ob))))

  (:action unstack
    :parameters  (?ob ?underob)
    :precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty))
    :effect (and (holding ?ob) (clear ?underob)
                 (not (on ?ob ?underob)) (not (clear ?ob)) (not (arm-empty)))))
"""

"""
(define (problem pb2)
   (:domain blocksworld)
   (:objects a b)
   (:init
     (on-table a)
     (on b a)
     (clear b)
     (arm-empty))
   (:goal (on a b)))
"""
