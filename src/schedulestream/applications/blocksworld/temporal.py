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
from typing import Optional

# NVIDIA
from schedulestream.algorithm.finite.solver import solve_finite
from schedulestream.algorithm.temporal import TemporalSolution
from schedulestream.applications.blocksworld.problems import (
    PROBLEMS,
    ArmEmpty,
    Clear,
    Holding,
    On,
    OnTable,
)
from schedulestream.applications.blocksworld.sequential import add_arguments
from schedulestream.applications.trimesh2d.utils import get_video_path, save_frames
from schedulestream.common.utils import SEPARATOR, Any, profiler
from schedulestream.language.anonymous import rename_anonymous
from schedulestream.language.durative import DurativeAction

pickup = DurativeAction(
    parameters=["?a", "?ob"],
    start_condition=Clear("?ob") & OnTable("?ob") & ArmEmpty("?a"),
    start_effect=~Clear("?ob") & ~OnTable("?ob") & ~ArmEmpty("?a"),
    end_effect=Holding("?a", "?ob"),
    language="Pick block ?ob off the table using arm ?a",
)

putdown = DurativeAction(
    parameters=["?a", "?ob"],
    start_condition=Holding("?a", "?ob"),
    start_effect=~Holding("?a", "?ob"),
    end_effect=Clear("?ob") & ArmEmpty("?a") & OnTable("?ob"),
    language="Put block ?ob on the table using arm ?a",
)

stack = DurativeAction(
    parameters=["?a", "?ob", "?underob"],
    start_condition=Clear("?underob") & Holding("?a", "?ob"),
    start_effect=~Clear("?underob") & ~Holding("?a", "?ob"),
    end_effect=ArmEmpty("?a") & Clear("?ob") & On("?ob", "?underob"),
    language="Stack block ?ob on block ?underob using arm ?a",
)

unstack = DurativeAction(
    parameters=["?a", "?ob", "?underob"],
    start_condition=On("?ob", "?underob") & Clear("?ob") & ArmEmpty("?a"),
    start_effect=~On("?ob", "?underob") & ~Clear("?ob") & ~ArmEmpty("?a"),
    end_effect=Holding("?a", "?ob") & Clear("?underob"),
    language="Unstack block ?ob from block ?underob using arm ?a",
)

ACTIONS = [pickup, putdown, stack, unstack]

rename_anonymous(locals())


def temporal(
    problem: str,
    arms: int,
    blocks: int,
    profile: bool = False,
    video: Optional[str] = None,
    visualize: bool = False,
    **kwargs: Any,
) -> TemporalSolution:
    video_path = get_video_path(video, name=problem)
    problem_fn = PROBLEMS[problem]
    problem = problem_fn(num_arms=arms, num_blocks=blocks)
    problem.actions = ACTIONS
    problem.dump()

    with profiler(field="tottime" if profile else None, num=25):
        solution = solve_finite(problem, **kwargs)
    print(SEPARATOR)
    print(solution)
    if solution.success:
        for timed_action in solution.timed_plan:
            print(
                f"{timed_action.start:.2f}-{timed_action.end:.2f}) {timed_action.action}:"
                f" '{timed_action.language}'"
            )

    if not visualize and (video_path is None):
        return solution

    # NVIDIA
    from schedulestream.applications.blocksworld.visualize import visualize_schedule

    frames = visualize_schedule(problem, solution.timed_plan, record=video_path)
    save_frames(frames, video_path=video_path)

    return solution


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser, arms=2, blocks=3, weight=1)
    args = parser.parse_args()
    print("Args:", args)

    temporal(
        args.problem,
        args.arms,
        args.blocks,
        algorithm=args.algorithm,
        sequential=args.sequential,
        weight=args.weight,
        video=args.video,
        visualize=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
