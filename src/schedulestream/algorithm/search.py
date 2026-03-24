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
from dataclasses import dataclass
from functools import cache, partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

# NVIDIA
from schedulestream.algorithm.heuristics import HEURISTICS, HeuristicFn, ReduceFn
from schedulestream.algorithm.successors import SUCCESSORS, SuccessorFn
from schedulestream.algorithm.utils import Solution
from schedulestream.common.queue import StablePriorityQueue
from schedulestream.common.utils import INF, SEPARATOR, current_time, elapsed_time
from schedulestream.language.action import ActionInstance
from schedulestream.language.connective import Sequence
from schedulestream.language.function import Function
from schedulestream.language.problem import Problem, ProblemABC
from schedulestream.language.state import State
from schedulestream.language.utils import INTERNAL_PREFIX


@dataclass
class Priority:
    estimate: float
    cost: float = 0.0

    def __iter__(self) -> Iterator[float]:
        return iter([self.estimate, self.cost])

    def __lt__(self, other: "Priority") -> bool:
        return tuple(self) < tuple(other)


def compute_plan_estimate(cost: float, heuristic: float, weight: float = 1.0) -> float:
    if weight == INF:
        return heuristic
    return cost + weight * heuristic


def compute_priority(cost: float, heuristic: float, weight: float = 1.0) -> Priority:
    estimate = compute_plan_estimate(cost, heuristic, weight)
    return Priority(estimate=estimate, cost=cost)


@dataclass
class Node:
    parent_state: Optional[State] = None
    parent_action: Optional[ActionInstance] = None
    cost: float = 0.0


def retrace_plan(tree: Dict[State, Node], state: State) -> List[ActionInstance]:
    node = tree[state]
    if node.parent_state is None:
        return []
    action = node.parent_action
    return retrace_plan(tree, node.parent_state) + [action]


StateFn = Callable[[State], Optional[State]]


@dataclass
class Heuristic:
    fn: HeuristicFn
    safe: bool = False
    admissible: bool = False
    name: Optional[str] = None

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    __repr__ = __str__


def load_heuristic_fn(
    problem: ProblemABC, heuristic_fn: Union[str, HeuristicFn, Heuristic], lazy: bool = True
) -> Heuristic:
    if isinstance(heuristic_fn, Heuristic):
        return heuristic_fn
    if not isinstance(heuristic_fn, str):
        return Heuristic(heuristic_fn)
    name = heuristic_fn
    assert name in HEURISTICS, (name, list(HEURISTICS))
    admissible = not lazy and (name in {"zero", "hmax"})
    transition = name in {"relaxed"}
    heuristic_fn = HEURISTICS[name](problem)

    if transition:
        _heuristic_fn = heuristic_fn
    else:
        _heuristic_fn = lambda s1, a, s2: heuristic_fn(s1 if lazy and (s1 is not None) else s2)
    return Heuristic(_heuristic_fn, safe=True, admissible=admissible, name=name)


def best_first_search(
    problem: ProblemABC,
    weight: float = 2.0,
    lazy: bool = True,
    operation: ReduceFn = sum,
    max_cost: float = INF,
    success_cost: float = INF,
    max_time: float = INF,
    max_iterations: float = INF,
    successor_fn: Union[str, SuccessorFn] = "relaxed-offline",
    state_fn: StateFn = lambda state: state,
    heuristic_fn: Union[str, HeuristicFn] = "relaxed",
    verbose: bool = False,
) -> Solution:
    start_time = current_time()
    algorithm_name = best_first_search.__name__
    print(
        f"{algorithm_name}) Successor: {successor_fn} | Heuristic: {heuristic_fn} | Weight:"
        f" {weight} | Lazy: {lazy} | Max time: {max_time:.2f} sec"
    )

    if not problem.simplified:
        problem = problem.simplify()
    if isinstance(successor_fn, str):
        assert successor_fn in SUCCESSORS, list(SUCCESSORS)
        successor_fn = SUCCESSORS[successor_fn](problem)
    _successor_fn = cache(successor_fn)

    assert weight >= 0.0, weight
    if weight == 0.0:
        heuristic_fn = "zero"
    heuristic = load_heuristic_fn(problem, heuristic_fn, lazy=lazy)
    heuristic_fn = cache(heuristic.fn)
    optimal = heuristic.admissible and (weight <= 1.0)

    goal = problem.goal
    if goal is None:
        return Solution(optimal=True)

    tree = {}
    queue = StablePriorityQueue()
    min_estimate = INF
    min_estimate_time = current_time()

    initial = problem.initial
    initial = state_fn(initial)
    initial_cost = 0.0
    estimate = heuristic_fn(None, None, initial)
    if estimate < min_estimate:
        min_estimate = estimate
        min_estimate_time = current_time()
    tree[initial] = Node(cost=initial_cost)
    priority = compute_priority(initial_cost, estimate, weight)
    queue.push(priority, (initial, initial_cost))

    best_solution = Solution()
    iteration = 0
    while (
        queue
        and not best_solution.optimal
        and (best_solution.cost >= success_cost)
        and (elapsed_time(start_time) <= max_time)
        and (iteration <= max_iterations)
    ):
        current_state, current_cost = queue.pop()
        if current_cost > tree[current_state].cost:
            continue
        iteration += 1

        successors = list(_successor_fn(current_state))
        satisfies_goal = goal.holds(current_state)
        if False:
            print(
                f"{algorithm_name}) Iteration: {iteration} | Cost: {current_cost:.2e} | Goal:"
                f" {satisfies_goal} | Successors: {len(successors)} | States: {len(tree)} | Best"
                f" Cost: {best_solution.cost:.2e} | Best Heuristic: {min_estimate:.2e} | Elapsed:"
                f" {elapsed_time(start_time):.2f} sec"
            )
        if satisfies_goal and (current_cost < best_solution.cost):
            best_solution = Solution(
                plan=retrace_plan(tree, current_state),
                cost=current_cost,
                optimal=optimal,
            )
            continue

        for plan in _successor_fn(current_state):
            state, cost = current_state, current_cost
            for action in plan:
                if not action.is_applicable(state):
                    break
                action_cost = action.cost.evaluate(state)
                new_cost = operation([cost, action_cost])
                if new_cost >= min(max_cost, best_solution.cost):
                    break
                new_state = action.apply(state)
                if new_state is None:
                    break
                new_state = state_fn(new_state)
                if new_state is None:
                    break
                if (new_state in tree) and (new_cost >= tree[new_state].cost):
                    break
                estimate = heuristic_fn(state, action, new_state)
                if estimate < min_estimate:
                    delta_elapsed = current_time() - min_estimate_time
                    min_estimate = estimate
                    min_estimate_time = current_time()
                    if verbose:
                        print(
                            f"{algorithm_name}) Iteration: {iteration} | State: {len(tree)} | New"
                            f" Heuristic: {min_estimate:.2e} | Cost: {new_cost:.2e} | Last Time:"
                            f" {delta_elapsed:.2f} sec | Elapsed:"
                            f" {elapsed_time(start_time):.2f} sec"
                        )

                if heuristic.safe and (estimate == INF):
                    break
                plan_estimate = compute_plan_estimate(cost, estimate, weight=1.0)
                if heuristic.admissible and (plan_estimate > max_cost):
                    break
                tree[new_state] = Node(parent_state=state, parent_action=action, cost=new_cost)
                priority = compute_priority(new_cost, estimate, weight)
                queue.push(priority, (new_state, new_cost))
                state, cost = new_state, new_cost

    print(
        f"{algorithm_name}) Iteration: {iteration} | States: {len(tree)} | Success:"
        f" {best_solution.success} | Cost: {best_solution.cost:.2e} | Best Heuristic:"
        f" {min_estimate:.2e} | Elapsed: {elapsed_time(start_time):.2f} sec"
    )
    best_solution.optimal |= heuristic.safe and not queue
    return best_solution


lifted_search = partial(best_first_search, successor_fn="online", heuristic_fn="zero")
uniform_cost_search = partial(best_first_search, heuristic_fn="zero", weight=0, lazy=False)
astar_search = partial(best_first_search, heuristic_fn="hmax", weight=1, lazy=False)
weighted_astar_search = partial(best_first_search, weight=1, lazy=True)
weighted_astar3_search = partial(best_first_search, weight=3, lazy=True)
weighted_astar5_search = partial(best_first_search, weight=5, lazy=True)
greedy_best_first_search = partial(best_first_search, weight=INF, lazy=True)


def iterative_search(
    problem: Problem,
    configs: List[Dict[str, Any]],
    max_time: float = INF,
    max_cost: float = INF,
    **kwargs,
) -> Solution:
    start_time = current_time()
    best_solution = Solution()
    for i, config in enumerate(configs):
        print(SEPARATOR)
        remaining_time = max_time - elapsed_time(start_time)
        print(
            f"{i+1}/{len(configs)}) Config: {config} | Best Cost: {best_solution.cost:.3f} |"
            f" Elapsed: {elapsed_time(start_time):.3f} sec"
        )
        solution = best_first_search(
            problem,
            max_time=remaining_time,
            max_cost=min(best_solution.cost, max_cost),
            **config,
            **kwargs,
        )
        if solution.cost < best_solution.cost:
            best_solution = solution
        if solution.optimal:
            break
    return best_solution


iterative_greedy_3_1_search = partial(
    iterative_search, configs=[dict(weight=w) for w in [INF, 3, 1]]
)


Stage = Function(name=f"{INTERNAL_PREFIX}stage")


def serialized_search(
    problem: ProblemABC,
    heuristic_fn: Union[str, HeuristicFn] = "relaxed",
    lazy: bool = True,
    scale: float = 100.0,
    **kwargs: Any,
) -> Solution:
    if isinstance(problem.goal, Sequence):
        goals = list(problem.goal.sequence)
    else:
        goals = [problem.goal]
    if len(goals) <= 1:
        return best_first_search(problem, heuristic_fn=heuristic_fn, lazy=lazy, **kwargs)
    goals.reverse()
    stage_term = Stage()

    def state_fn(state: State) -> Optional[State]:
        stage = state.get_value(stage_term).unwrap()
        if stage is None:
            stage = len(goals) - 1
        goal = goals[stage]
        if (stage != 0) and goal.holds(state):
            stage = max(0, stage - 1)
        evaluations = [Stage() <= stage]
        return state.new_state(evaluations)

    heuristics = {}

    def staged_heuristic_fn(
        state1: Optional[State], action: Optional[ActionInstance], state2: State
    ) -> float:
        stage = state2.get_value(stage_term).unwrap()
        if stage not in heuristics:
            _problem = problem.clone(goal=goals[stage])
            heuristics[stage] = load_heuristic_fn(_problem, heuristic_fn, lazy=lazy)
        estimate = heuristics[stage].fn(state1, action, state2)
        estimate += scale * stage
        return estimate

    heuristic = Heuristic(staged_heuristic_fn, safe=True)
    return best_first_search(problem, state_fn=state_fn, heuristic_fn=heuristic, **kwargs)


def search(*args: Any, algorithm: str = "serialized", **kwargs: Any) -> Solution:
    if algorithm == "bfs":
        return best_first_search(*args, **kwargs)
    if algorithm == "serialized":
        return serialized_search(*args, **kwargs)
    raise ValueError(algorithm)
