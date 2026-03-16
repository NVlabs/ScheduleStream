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
from typing import Any, List

# NVIDIA
from schedulestream.algorithm.finite.online import disable_lazy, solve_online
from schedulestream.algorithm.instantiation import instantiate_streams
from schedulestream.algorithm.stream.common import StreamSolution
from schedulestream.algorithm.stream.stream_plan import extract_stream_plan
from schedulestream.common.ordered_set import OrderedSet
from schedulestream.common.utils import INF, current_time, elapsed_time
from schedulestream.language.problem import Problem
from schedulestream.language.stream import StreamOutput


def instantiate_incremental(problem: Problem, max_iterations: int = INF) -> List[StreamOutput]:
    stream_instances = OrderedSet()
    stream_outputs = OrderedSet()
    unchanged = False
    iteration = 0
    while iteration <= max_iterations:
        exhausted = (iteration != 0) and all(instance.exhausted for instance in stream_instances)
        if unchanged and exhausted:
            break
        new_evaluations = []
        stream_instances.update(instantiate_streams(problem.initial, problem.streams))
        for stream_instance in stream_instances:
            for stream_output in stream_instance.next_outputs():
                stream_outputs.add(stream_output)
                for evaluation in stream_output.output_condition.clause:
                    if not evaluation.holds(problem.initial):
                        new_evaluations.append(evaluation)
        problem.initial = problem.initial.new_state(new_evaluations)
        unchanged |= not new_evaluations
        iteration += 1
    return list(stream_outputs)


def solve_incremental(
    problem: Problem,
    max_time: float = INF,
    search_time: float = INF,
    search_iteration: int = 0,
    over_conditions: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> List[StreamSolution]:
    start_time = current_time()
    disable_lazy(problem)
    stream_instances = OrderedSet()
    streams_from_evaluation = {evaluation: None for evaluation in problem.initial}
    solution = StreamSolution()
    unchanged = False
    iteration = 0
    while elapsed_time(start_time) <= max_time:
        exhausted = (iteration != 0) and all(instance.exhausted for instance in stream_instances)
        if (iteration >= search_iteration) or (exhausted and unchanged):
            solution = solve_online(
                problem, max_time=min(max_time - elapsed_time(start_time), search_time), **kwargs
            )
            if solution.success:
                break
        if unchanged and exhausted:
            break

        new_evaluations = []
        stream_instances.update(instantiate_streams(problem.initial, problem.streams))
        for stream_instance in stream_instances:
            for stream_output in stream_instance.next_outputs():
                for evaluation in stream_output.output_condition.clause:
                    if streams_from_evaluation.setdefault(evaluation, []) is not None:
                        streams_from_evaluation[evaluation].append(stream_output)
                    if not evaluation.holds(problem.initial):
                        new_evaluations.append(evaluation)

        if verbose:
            print(
                f"Iteration {iteration}) Evaluations ({(len(new_evaluations))}):"
                f" {new_evaluations} | Elapsed: {elapsed_time(start_time):.3f} sec\n"
            )
        problem.initial = problem.initial.new_state(new_evaluations)
        unchanged |= not new_evaluations
        iteration += 1

    stream_plan = extract_stream_plan(
        problem.initial, solution.plan, streams_from_evaluation, over_conditions=over_conditions
    )
    solution = StreamSolution.from_solution(solution, stream_plan=stream_plan)
    return [solution]
