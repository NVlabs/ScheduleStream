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
from typing import Any, List, Tuple

# NVIDIA
from schedulestream.algorithm.schedule import schedule
from schedulestream.algorithm.stream.stream_plan import compute_preimage
from schedulestream.algorithm.temporal import TemporalSolution
from schedulestream.common.utils import INF, SEPARATOR, current_time, elapsed_time, partition
from schedulestream.language.constraint import Constraint, Cost
from schedulestream.language.problem import Problem


def filter_costs(costs: List[Cost]) -> List[Cost]:
    return list(filter(lambda c: c.is_procedural, costs))


def partition_costs(costs: List[Cost]) -> Tuple[List[Constraint], List[Cost]]:
    return partition(lambda c: isinstance(c, Constraint), filter_costs(costs))


def evaluate_constraints(costs: List[Cost], greedy: bool = False) -> bool:
    constraints, _ = partition_costs(costs)
    satisfied = True
    for constraint in constraints:
        if not constraint.is_static:
            raise NotImplementedError(constraint)
        satisfied &= constraint.evaluate()
        if greedy and not satisfied:
            break
    return satisfied


def evaluate_costs(costs: List[Cost], max_cost: float = INF) -> float:
    _, costs = partition_costs(costs)
    total_cost = 0.0
    for cost in costs:
        if not cost.is_static:
            raise NotImplementedError(cost)
        total_cost += cost.evaluate()
        if total_cost > max_cost:
            break
    return total_cost


def solve_lazy(
    problem: Problem,
    max_time: float = INF,
    max_cost: float = INF,
    success_cost: float = INF,
    verbose: bool = False,
    **kwargs: Any,
) -> TemporalSolution:
    start_time = current_time()
    assert not problem.streams
    best_solution = TemporalSolution()
    instantiated = problem.instantiate()
    initial = instantiated.initial
    iteration = 0
    while elapsed_time(start_time) <= max_time and (best_solution.cost >= success_cost):
        iteration += 1
        if verbose:
            print(SEPARATOR)
        remaining_time = max_time - elapsed_time(start_time)
        solution = schedule(
            instantiated,
            max_time=remaining_time,
            max_cost=min(best_solution.cost, max_cost),
            verbose=verbose,
            **kwargs,
        )
        if not solution.success:
            break
        dependencies = filter_costs(list(compute_preimage(initial, solution.plan)))
        cost = evaluate_costs(dependencies)
        satisfied = evaluate_constraints(dependencies)
        if verbose:
            print(f"Iteration: {iteration} | Satisfied: {satisfied} | Cost: {cost:.3f}")
        if satisfied and (solution.score < best_solution.score):
            best_solution = solution

    return best_solution
