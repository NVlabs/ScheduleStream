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
from typing import Any, List, Optional

# NVIDIA
from schedulestream.algorithm.stream.common import StreamSolution
from schedulestream.algorithm.stream.incremental import solve_incremental
from schedulestream.algorithm.stream.satisfier import Satisfier
from schedulestream.algorithm.stream.utils import complete_solution, retrace_solution
from schedulestream.common.utils import (
    INF,
    SEPARATOR,
    current_time,
    elapsed_time,
    flatten,
    get_length,
    implies,
)
from schedulestream.language.generator import ConditionalGenerator, Output, from_list_fn
from schedulestream.language.lazy import LazyOutput
from schedulestream.language.problem import Problem
from schedulestream.language.stream import Stream, StreamOutput


def lazy_stream(
    stream: Stream, stream_outputs: List[StreamOutput], **kwargs
) -> ConditionalGenerator:
    lazy_generator = LazyOutput.conditional_generator(stream, **kwargs)
    outputs_from_instance = {}
    for output in stream_outputs:
        if output.stream != stream:
            continue
        outputs_from_instance.setdefault(output.stream_instance, []).append(output)

    def list_fn(*inputs: Any) -> List[Output]:
        instance = stream.instantiate(inputs)
        if instance in outputs_from_instance:
            return [output.outputs for output in outputs_from_instance[instance]]
        if instance.exhausted or instance.called:
            return []
        return list(flatten(lazy_generator(*inputs)))

    return from_list_fn(list_fn)


def solve_lazy(
    problem: Problem, stream_outputs: Optional[List[StreamOutput]] = None, **kwargs: Any
) -> StreamSolution:
    if stream_outputs is None:
        stream_outputs = []
    lazy_streams = []
    for stream in problem.streams:
        lazy_streams.append(stream.clone(conditional_generator=lazy_stream(stream, stream_outputs)))
    abstract_problem = problem.clone(streams=lazy_streams)
    solutions = solve_incremental(abstract_problem, search_iteration=INF, **kwargs)
    if not solutions:
        return StreamSolution()
    return solutions[0]


def solve_focused(
    problem: Problem,
    max_skeletons: int = INF,
    max_solutions: int = 1,
    max_time: float = INF,
    search_time: float = INF,
    satisfy_time: float = 0.0,
    optimize_time: float = 0.0,
    num_reschedule: int = 0,
    success_cost: float = INF,
    greedy: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> List[StreamSolution]:
    start_time = current_time()
    satisfier = Satisfier()
    satisfier.improved = True
    problem.lazily_wrap_functions()

    iterator = 0
    subproblems = {}
    while (
        elapsed_time(start_time) < max_time
        and (satisfier.num_skeletons < max_skeletons)
        and (satisfier.num_solutions < max_solutions)
    ):
        print(SEPARATOR)
        print(
            f"Iteration: {iterator} | Skeletons: {satisfier.num_skeletons} | Solutions:"
            f" {satisfier.num_solutions} | Makespan: {satisfier.min_makespan:.3f}"
        )
        iterator += 1

        solution = None
        for skeleton in satisfier.skeletons:
            binding = skeleton.best_binding
            if (
                binding.bound
                and not binding.complete
                and not binding.rescheduled
                and not satisfier.num_solutions
            ):
                binding.rescheduled = True
                completed_solution = complete_solution(problem, binding.partial_solution)
                if completed_solution.makespan < satisfier.min_makespan:
                    solution = completed_solution
                    break

        if solution is None and satisfier.improved:
            stream_outputs = []
            for skeleton in satisfier.skeletons:
                stream_outputs.extend(skeleton.best_binding.stream_plan)
            subproblem = frozenset(stream_outputs)

            epsilon = 1e-3

            remaining_time = max_time - elapsed_time(start_time)
            if satisfier:
                remaining_time = min(remaining_time, search_time)
            solution = solve_lazy(
                problem,
                stream_outputs,
                max_time=remaining_time,
                max_cost=satisfier.min_makespan - epsilon,
                **kwargs,
            )
            satisfier.improved = False
            subproblems[subproblem] = solution.success

        if solution is not None:
            solution = retrace_solution(solution)
            if verbose:
                print(
                    f"Plan ({get_length(solution.plan)}): {solution.plan}\nStream Plan"
                    f" ({get_length(solution.stream_plan)}): {solution.stream_plan}"
                )
            satisfier.satisfy_skeleton(solution)

        failed = solution is None
        remaining_time = max_time - elapsed_time(start_time)
        if (
            not failed
            and (satisfier.num_skeletons < max_skeletons)
            and implies(greedy, not satisfier.solution_bindings)
        ):
            remaining_time = min(satisfy_time, remaining_time)
        if satisfier.solution_bindings:
            remaining_time = remaining_time - optimize_time

        improved = satisfier.satisfy(
            max_time=remaining_time,
            max_solutions=max_solutions,
            success_cost=success_cost,
            abort=failed,
        )
        if failed and not improved:
            break

    remaining_time = max_time - elapsed_time(start_time)
    satisfier.satisfy(
        max_time=min(optimize_time, remaining_time), max_solutions=INF, success_cost=0.0
    )

    solutions = satisfier.sorted_solutions
    start_time = current_time()
    for i, solution in enumerate(list(solutions)):
        if i >= num_reschedule:
            break
        rescheduled_solution = complete_solution(problem, solution)
        improved = rescheduled_solution.makespan < solution.makespan
        print(
            f"{i}/{len(satisfier.sorted_solutions)} | Improved: {improved} | Original Makespan:"
            f" {solution.makespan:.3f} | Rescheduled Makespan:"
            f" {rescheduled_solution.makespan:.3f} | Elapsed: {elapsed_time(start_time):.3f} sec"
        )
        if improved:
            solutions.append(rescheduled_solution)
    solutions.sort(key=lambda s: s.makespan)

    return solutions
