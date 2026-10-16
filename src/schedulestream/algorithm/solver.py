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
from schedulestream.algorithm.finite.solver import solve_finite
from schedulestream.algorithm.stream.solver import solve_stream
from schedulestream.algorithm.utils import Solution
from schedulestream.common.utils import (
    INF,
    current_time,
    elapsed_time,
    flatten,
    remove_duplicates,
)
from schedulestream.language.action import Action
from schedulestream.language.function import Function
from schedulestream.language.problem import Problem


def solve(problem: Problem, **kwargs: Any) -> List[Solution]:
    if problem.is_stream:
        return solve_stream(problem, **kwargs)
    solution = solve_finite(problem, **kwargs)
    if solution is None:
        return []
    return [solution]


COST_TOLERANCE = 0.1
"""Fraction above the best cost still counted as near-optimal in the solve summary."""


def solve_hierarchical(
    problem: Problem,
    functions: Optional[List[Function]] = None,
    actions: Optional[List[Action]] = None,
    max_time: float = INF,
    strict: bool = True,
    **kwargs: Any,
) -> List[Solution]:
    """Solve without ``functions`` and the effects of ``actions`` (dropping streams/actions that
    become irrelevant) at the high level, then refine each high-level solution in turn at the low
    level on the full problem, seeded with its stream outputs and plan, until one succeeds or
    ``max_time`` runs out. If ``strict``, the low level only uses the deferred streams. Without
    either, just solve."""
    start_time = current_time()
    effect_functions = list(flatten(action.effect.functions for action in actions or []))
    functions = remove_duplicates(list(functions or []) + effect_functions)
    if not functions:
        return solve(problem, max_time=max_time, **kwargs)
    high = problem.remove_conditions(functions, reduce=True)
    high_action_names = {action.name for action in high.actions}
    deferred_actions = [
        action for action in problem.actions if action.name not in high_action_names
    ]
    deferred_streams = [stream for stream in problem.streams if stream not in high.streams]
    print(
        f"{solve_hierarchical.__name__}) Deferred functions ({len(functions)}): {functions} |"
        f" Deferred streams ({len(deferred_streams)}): {deferred_streams} | Deferred actions"
        f" ({len(deferred_actions)}): {[action.name for action in deferred_actions]}"
    )
    high_solutions = solve(high, max_time=max_time, **kwargs)
    costs = [solution.cost for solution in high_solutions]
    best = min(costs, default=INF)
    near_best = (
        sum(cost <= best * (1.0 + COST_TOLERANCE) for cost in costs) if costs and best < INF else 0
    )
    print(
        f"{solve_hierarchical.__name__}) High-level Solutions: {len(high_solutions)} |"
        f" Costs: [{best:.3f}, {max(costs, default=INF):.3f}] |"
        f" Within {COST_TOLERANCE:.0%}: {near_best} |"
        f" Elapsed: {elapsed_time(start_time):.3f} sec"
    )
    full = high.root
    for solution in high_solutions:
        remaining_time = max_time - elapsed_time(start_time)
        if remaining_time <= 0:
            break
        solution.dump()
        initial = full.initial.new_state(evaluations=remove_duplicates(solution.stream_atoms))
        low = full.clone(initial=initial, streams=deferred_streams if strict else full.streams)
        low.add_partial_plan(solution.partial_plan)
        low_solutions = solve(low, max_time=remaining_time, **kwargs)
        if low_solutions:
            return low_solutions
    return []
