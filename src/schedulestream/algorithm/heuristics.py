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
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Callable, Counter, Dict, Iterable, List, Optional, Set, Tuple, Union

# NVIDIA
from schedulestream.algorithm.temporal import get_makespan, time_plan
from schedulestream.algorithm.utils import Plan
from schedulestream.common.ordered_set import OrderedSet
from schedulestream.common.queue import StablePriorityQueue
from schedulestream.common.utils import (
    INF,
    current_time,
    elapsed_time,
    get_length,
    instance_cache,
    safe_max,
)
from schedulestream.language.action import ActionInstance
from schedulestream.language.durative import CLOCK, EndInstance, StartInstance, get_ongoing_actions
from schedulestream.language.expression import Formula
from schedulestream.language.function import Evaluation, Term
from schedulestream.language.predicate import Atom
from schedulestream.language.problem import ProblemABC
from schedulestream.language.state import State

HeuristicFn = Callable[[State], float]
ReduceFn = Callable[[Iterable[float]], float]


def create_zero_heuristic_fn(problem: ProblemABC) -> HeuristicFn:
    return lambda state: 0.0


def create_goal_heuristic_fn(problem: ProblemABC) -> HeuristicFn:
    def heuristic_fn(state: State) -> float:
        return sum(not goal.holds(state) for goal in problem.goal.clause)

    return cache(heuristic_fn)


@dataclass
class RelaxedNode:
    reached: bool = False
    cost: float = INF
    parent_actions: List[ActionInstance] = field(default_factory=list)

    @property
    def parent_action(self) -> Optional[ActionInstance]:
        return self.parent_actions[0] if self.parent_actions else None


@instance_cache
def get_preconditions(action: Union[ActionInstance, Formula]) -> List[Atom]:
    precondition = action.precondition if isinstance(action, ActionInstance) else action
    precondition.assert_conjunction()
    return precondition.simple_clause


def relaxed_cost(
    state: State, action: ActionInstance, unit_weight: Optional[float] = None
) -> float:
    cost = action.cost.evaluate(state)

    start_weight = 0.5
    if isinstance(action, StartInstance):
        duration = action.durative_instance.min_duration.evaluate(state)
        cost = start_weight * duration + CLOCK
    elif isinstance(action, EndInstance):
        remaining = action.cost.evaluate(state)
        cost = (1 - start_weight) * remaining + CLOCK

    if unit_weight is not None:
        cost += unit_weight * 1
    assert cost >= 0.0
    return cost


def compute_costs(
    state: State,
    goal: Formula,
    actions: List[ActionInstance],
    operation: ReduceFn = sum,
    unit_weight: float = 0.0,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[Evaluation, RelaxedNode]:
    start_time = current_time()
    queue = StablePriorityQueue()
    evaluations = OrderedSet(goal.simple_clause)

    ongoing_actions = {}
    for action in actions:
        if isinstance(action, EndInstance) and action.ongoing_atom.holds(state):
            remaining = state.get_value(action.remaining_term)
            ongoing_actions[action] = remaining.unwrap()

    for action in actions:
        evaluations.update(action.precondition.simple_clause)
        evaluations.update(action.effect.simple_clause)

        start_weight = 1.0
        if isinstance(action, StartInstance):
            duration = action.durative_instance.min_duration.evaluate(state)
            cost = start_weight * duration + CLOCK
        elif isinstance(action, EndInstance):
            if action in ongoing_actions:
                duration = ongoing_actions[action]
            else:
                duration = action.durative_instance.min_duration.evaluate(state)
            cost = (1 - start_weight) * duration + CLOCK
        else:
            cost = action.cost.evaluate(state)
        action._cost = 1.0 if unit_weight == INF else cost + unit_weight * 1.0

    for evaluation in evaluations:
        evaluation.processed = False
        evaluation.cost = INF
        evaluation.parents = []
        evaluation.outgoing = []
        if evaluation.holds(state):
            evaluation.cost = 0.0
            queue.push(evaluation.cost, evaluation)

    def process_action(action: ActionInstance) -> bool:
        preconditions = get_preconditions(action)
        if len(preconditions) == 0:
            action_cost = 0.0
        else:
            action_cost = operation(e.cost for e in preconditions)
        if action is goal:
            return True
        effect_cost = action_cost + action._cost
        for effect in action.effect.simple_clause:
            if effect_cost < effect.cost:
                effect.cost = effect_cost
                effect.parents = [action]
                queue.push(effect_cost, effect)
            elif effect.parents and not effect.processed and (effect_cost == effect.cost):
                effect.parents.append(action)
        return False

    reached_goal = False
    for action in actions + [goal]:
        preconditions = get_preconditions(action)
        for evaluation in preconditions:
            evaluation.outgoing.append(action)
        action.num_unsatisfied = len(preconditions)
        if action.num_unsatisfied == 0:
            reached_goal |= process_action(action)

    while queue and not reached_goal:
        evaluation = queue.pop()
        if evaluation.processed:
            continue
        evaluation.processed = True
        for action in evaluation.outgoing:
            action.num_unsatisfied -= 1
            if action.num_unsatisfied == 0:
                reached_goal |= process_action(action)

    reached_nodes = defaultdict(RelaxedNode)
    for evaluation in evaluations:
        reached_nodes[evaluation] = RelaxedNode(
            reached=evaluation.cost < INF,
            cost=evaluation.cost,
            parent_actions=evaluation.parents,
        )

    if verbose:
        reached = [evaluation for evaluation, node in reached_nodes.items() if node.cost < INF]
        frequencies = Counter(evaluation.function for evaluation in reached)
        print(
            f"{compute_costs.__name__}) Reached: {len(reached)}/{len(evaluations)} | Function:"
            f" {dict(frequencies)} | Elapsed: {elapsed_time(start_time):.3f} sec"
        )

    return reached_nodes


def create_reachability_heuristic_fn(
    problem: ProblemABC, operation: ReduceFn, **kwargs: Any
) -> HeuristicFn:
    problem = problem.instantiate()

    def heuristic_fn(state: State) -> float:
        goal = problem.goal
        reached_nodes = compute_costs(state, goal, problem.actions, operation=operation, **kwargs)
        heuristic = operation(reached_nodes[e].cost for e in get_preconditions(goal))
        return heuristic

    return cache(heuristic_fn)


def create_hmax_heuristic_fn(problem: ProblemABC, **kwargs: Any) -> HeuristicFn:
    return create_reachability_heuristic_fn(problem, operation=safe_max, **kwargs)


def create_hadd_heuristic_fn(problem: ProblemABC, **kwargs: Any) -> HeuristicFn:
    return create_reachability_heuristic_fn(problem, operation=sum, **kwargs)


@dataclass
class RetraceContext:
    values: Dict[Term, Any] = field(default_factory=dict)
    selected: Set[ActionInstance] = field(default_factory=set)
    subplans: Dict[ActionInstance, Tuple[ActionInstance, ...]] = field(default_factory=dict)


def default_subplan(
    reached_nodes: Dict[Evaluation, RelaxedNode],
    action: ActionInstance,
    subplans: Dict[ActionInstance, Tuple[ActionInstance, ...]],
) -> Tuple[ActionInstance, ...]:
    if action not in subplans:
        plan = {}
        for evaluation in get_preconditions(action):
            parent_action = reached_nodes[evaluation].parent_action
            if parent_action is not None:
                plan.update(dict.fromkeys(default_subplan(reached_nodes, parent_action, subplans)))
        plan[action] = None
        subplans[action] = tuple(plan)
    return subplans[action]


def commit_action(action: Union[ActionInstance, Formula], context: RetraceContext) -> None:
    context.selected.add(action)
    for evaluation in get_preconditions(action):
        context.values.setdefault(evaluation.term, evaluation.value)


def retrace_relaxed_plan(
    reached_nodes: Dict[Evaluation, RelaxedNode],
    action: ActionInstance,
    achieved_evaluations: Optional[Set[Evaluation]] = None,
    context: Optional[RetraceContext] = None,
) -> List[ActionInstance]:
    relaxed_plan = []
    achieved_evaluations = set(achieved_evaluations or set())
    if context is None:
        context = RetraceContext()
        commit_action(action, context)
    preconditions = get_preconditions(action)
    for evaluation in preconditions:
        if evaluation in achieved_evaluations:
            continue
        assert reached_nodes[evaluation].reached, evaluation
        node = reached_nodes[evaluation]
        parent_action = node.parent_action
        if len(node.parent_actions) > 1:
            parent_action = min(
                node.parent_actions,
                key=lambda a: sum(
                    context.values.get(e.term, e.value) != e.value
                    for b in default_subplan(reached_nodes, a, context.subplans)
                    if b not in context.selected
                    for e in get_preconditions(b) + b.effect.simple_clause
                ),
            )
        if parent_action is not None:
            commit_action(parent_action, context)
            for relaxed_action in retrace_relaxed_plan(
                reached_nodes, parent_action, achieved_evaluations, context
            ):
                relaxed_plan.append(relaxed_action)
                achieved_evaluations.update(relaxed_action.effect.simple_clause)
    return list(OrderedSet(relaxed_plan + [action]))


def create_relaxed_plan_fn(
    problem: ProblemABC,
    operation: ReduceFn = max,
    unit: bool = False,
    makespan: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Callable[[State], Optional[Plan]]:
    problem = problem.instantiate()

    def relaxed_plan_fn(state: State) -> Tuple[Optional[Plan], float]:
        goal = problem.goal
        reached_nodes = compute_costs(state, goal, problem.actions, operation=operation, **kwargs)
        reached = all(reached_nodes[e].reached for e in get_preconditions(goal))
        if not reached:
            return None, INF
        relaxed_plan = retrace_relaxed_plan(reached_nodes, goal)
        relaxed_plan = relaxed_plan[:-1]
        if unit:
            cost = get_length(relaxed_plan)
        elif makespan:
            cost = get_makespan(time_plan(relaxed_plan, ongoing=get_ongoing_actions(state)))
        else:
            cost = compute_relaxed_plan_cost(relaxed_plan)

        if verbose:
            print(
                f"State: {state}\nRelaxed Plan: {relaxed_plan} | Length:"
                f" {get_length(relaxed_plan)} | Cost: {cost:.3e}"
            )
            for i, action in enumerate(relaxed_plan):
                print(f"{i}/{len(relaxed_plan)}) Action: {action} | Cost: {action._cost}")
        return relaxed_plan, cost

    return cache(relaxed_plan_fn)


def compute_relaxed_plan_cost(plan: Optional[Plan] = None, operation: ReduceFn = sum) -> float:
    if plan is None:
        return INF
    if not plan:
        return 0.0
    return operation(action._cost for action in plan)


def create_hff_heuristic_fn(problem: ProblemABC, **kwargs: Any) -> HeuristicFn:
    relaxed_plan_fn = create_relaxed_plan_fn(problem, **kwargs)

    def heuristic_fn(state: State) -> float:
        _, cost = relaxed_plan_fn(state)
        return cost

    return cache(heuristic_fn)


def create_lazy_relaxed_heuristic_fn(
    problem: ProblemABC,
    **kwargs: Any,
) -> HeuristicFn:
    relaxed_plan_fn = cache(create_relaxed_plan_fn(problem, **kwargs))

    def heuristic_fn(
        state1: Optional[State], action: Optional[ActionInstance], state2: State
    ) -> float:
        if action is None:
            _, cost = relaxed_plan_fn(state2)
            return cost
        relaxed_plan, cost = relaxed_plan_fn(state1)
        if (relaxed_plan is None) or (action not in relaxed_plan):
            return cost
        _, cost = relaxed_plan_fn(state2)
        return cost

    return cache(heuristic_fn)


HEURISTICS = {
    "zero": create_zero_heuristic_fn,
    "goal": create_goal_heuristic_fn,
    "hmax": create_hmax_heuristic_fn,
    "hadd": create_hadd_heuristic_fn,
    "hff": create_hff_heuristic_fn,
    "relaxed": create_lazy_relaxed_heuristic_fn,
}
