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
from typing import Any, Optional, Union

# NVIDIA
from schedulestream.algorithm.heuristics import HeuristicFn
from schedulestream.algorithm.search import best_first_search, search
from schedulestream.algorithm.stream.common import StreamSolution
from schedulestream.algorithm.successors import SuccessorFn
from schedulestream.algorithm.temporal import (
    LinearPlan,
    TemporalSolution,
    TimedPlan,
    actions_from_timed,
    get_makespan,
    recover_plan,
    sequential_from_temporal,
    sequential_from_timed,
    serialize_plan,
    time_plan,
)
from schedulestream.algorithm.utils import (
    Plan,
    compute_partial_orders,
    compute_plan_cost,
    visualize_plan,
)
from schedulestream.common.utils import INF, compute_mapping, current_time, elapsed_time, get_length
from schedulestream.language.durative import EndAction
from schedulestream.language.problem import InstantiatedProblem, Problem, ProblemABC


def reschedule_temporal(
    problem: Problem,
    plan: Optional[Plan],
    max_cost: float = INF,
) -> Optional[Plan]:
    if plan is None:
        return plan
    state = problem.initial
    new_plan = []
    remaining_actions = list(plan)
    while remaining_actions and not problem.goal.holds(state):
        action_costs = {action: action.cost.evaluate(state) for action in remaining_actions}
        for action in sorted(action_costs, key=action_costs.get):
            if action.is_applicable(state):
                new_state = action.apply(state)
                if new_state is not None:
                    new_plan.append(action)
                    remaining_actions.remove(action)
                    state = new_state
                    break
        else:
            return plan
    return new_plan


def reschedule_search(
    problem: Problem,
    plan: Plan,
    weight: float = 1.0,
    lazy: bool = False,
    successor_fn: Union[str, SuccessorFn] = "offline",
    heuristic_fn: Union[str, HeuristicFn] = "hmax",
    **kwargs: Any,
) -> Plan:
    reschedule_problem = InstantiatedProblem(
        problem.initial,
        actions=plan,
        goal=problem.goal,
        problem=problem,
        parent=problem.parent,
    )
    solution = best_first_search(
        reschedule_problem,
        weight=weight,
        lazy=lazy,
        successor_fn=successor_fn,
        heuristic_fn=heuristic_fn,
        **kwargs,
    )
    if solution.success is None:
        return plan
    return solution.plan


def reschedule_milp(plan: Plan, **kwargs: Any):
    if not plan:
        return plan
    # NVIDIA
    from schedulestream.common.milp import Constraint, Cost, Variable, solve_milp

    variables = []
    constraints = []
    for idx2, instance in enumerate(plan):
        name = str(instance)
        variables.append(Variable(name, lower=0.0))
        if isinstance(instance.action, EndAction):
            durative_action = instance.action.durative_action
            end_instance = durative_action.start_action.instantiate(instance.arguments)
            for idx1 in reversed(range(0, idx2)):
                if plan[idx1] == end_instance:
                    break
            else:
                raise RuntimeError(end_instance)

            constraints.append(
                Constraint(
                    coefficients={
                        variables[idx2].name: 1.0,
                        variables[idx1].name: -1.0,
                    },
                    lower=durative_action.min_duration,
                    upper=durative_action.max_duration,
                )
            )

    for action1, action2 in compute_partial_orders(plan):
        idx1 = plan.index(action1)
        idx2 = plan.index(action2)
        constraints.append(
            Constraint(
                coefficients={
                    variables[idx2].name: 1.0,
                    variables[idx1].name: -1.0,
                },
                lower=0.0,
            )
        )

    makespan = Variable(name="makespan", lower=0.0)
    for variable in variables:
        constraints.append(
            Constraint(
                coefficients={
                    makespan.name: 1.0,
                    variable.name: -1.0,
                },
                lower=0.0,
            )
        )
    variables.append(makespan)
    costs = [Cost(coefficients={makespan.name: 1.0})]

    solution = solve_milp(variables, constraints=constraints, costs=costs, **kwargs)
    if solution is None:
        return plan
    times = [solution[variable.name] for variable in variables[:-1]]
    action_times = compute_mapping(plan, times)
    plan.sort(key=action_times.get)
    return plan


def reschedule_plan(
    sequential_problem: Problem,
    plan: Optional[LinearPlan],
    algorithm: Optional[Union[bool, str]] = True,
    verbose: bool = True,
) -> Optional[TimedPlan]:
    if plan is None:
        return None
    if algorithm is True:
        algorithm = "search"
    elif algorithm is False:
        algorithm = None
    start_time = current_time()
    timed_plan = time_plan(plan)
    if algorithm is None:
        new_plan = plan
    elif algorithm == "search":
        new_plan = reschedule_temporal(sequential_problem, plan)
    elif algorithm == "milp":
        new_plan = reschedule_milp(plan)
    else:
        raise ValueError(algorithm)
    new_timed_plan = time_plan(new_plan)
    if verbose:
        print(
            f"Reschedule: {algorithm} | Makespan: {get_makespan(timed_plan):.3f} | Length:"
            f" {get_length(timed_plan)} | New Makespan: {get_makespan(new_timed_plan):.3f} | New"
            f" Length: {get_length(new_timed_plan)} | Elapsed: {elapsed_time(start_time):.3f} sec"
        )
    return new_timed_plan


def schedule_solution(
    problem: Problem, solution: StreamSolution, **kwargs: Any
) -> Optional[TimedPlan]:
    initial = problem.initial.new_state(solution.stream_atoms)
    actions = actions_from_timed(solution.plan)
    new_problem = InstantiatedProblem(initial, actions, problem.goal, problem=None)
    new_problem = new_problem.simplify()
    new_problem = sequential_from_temporal(new_problem)
    sequential_plan = reschedule_temporal(new_problem, new_problem.actions, **kwargs)
    return time_plan(sequential_plan)


def schedule(
    problem: ProblemABC,
    sequential: bool = False,
    reschedule: Optional[Union[str, bool]] = None,
    visualize: bool = False,
    **kwargs: Any,
) -> TemporalSolution:
    problem = problem.simplify()
    sequential_problem = sequential_from_temporal(problem)
    if sequential:
        solution = search(problem, **kwargs)
        plan = serialize_plan(solution.plan)
    else:
        solution = search(sequential_problem, **kwargs)
        plan = solution.plan
    if not solution.success:
        return TemporalSolution.from_solution(solution)
    if visualize:
        visualize_plan(plan)

    if reschedule is None:
        reschedule = not sequential
    timed_plan = reschedule_plan(sequential_problem, plan, algorithm=reschedule)
    cost = compute_plan_cost(sequential_from_timed(timed_plan), state=problem.initial)

    timed_plan = recover_plan(timed_plan)
    return TemporalSolution(
        plan=timed_plan,
        cost=cost - get_makespan(timed_plan),
        optimal=False,
    )
