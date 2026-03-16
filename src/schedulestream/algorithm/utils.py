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
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

# NVIDIA
from schedulestream.common.graph import transitive_reduction, visualize_graph
from schedulestream.common.ordered_set import OrderedSet
from schedulestream.common.utils import INF, current_time, get_dataclass_dict, get_length, get_pairs
from schedulestream.language.action import Action, ActionInstance
from schedulestream.language.argument import Constant
from schedulestream.language.expression import Formula
from schedulestream.language.state import State

Plan = List[ActionInstance]


class PartialPlan:
    def __init__(self, actions: Optional[List[ActionInstance]] = None):
        self.actions = OrderedSet(actions or [])

    def unordered(self) -> "PartialPlan":
        return self.__class__(actions=self.actions)

    def language_plan(self) -> "PartialPlan":
        actions = []
        for action in self.actions:
            if action.language is not None:
                parameters = action.action.language_parameters
                actions.append(action.partially_instantiate(parameters))
        return self.__class__(actions=actions)

    def __bool__(self) -> bool:
        return bool(self.actions)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(actions={self.actions})"

    __repr__ = __str__


def filter_applicable(state: State, actions: List[ActionInstance]) -> List[ActionInstance]:
    return list(filter(lambda action: action.is_applicable(state), actions))


def applicable_actions(state: State, actions: List[Action]) -> List[ActionInstance]:
    # NVIDIA
    from schedulestream.algorithm.instantiation import instantiate_actions

    return filter_applicable(state, instantiate_actions(state, actions))


@dataclass
class Solution:
    plan: Optional[Plan] = None
    cost: Optional[float] = INF
    optimal: float = False
    discovery_time: float = INF

    def __post_init__(self) -> None:
        self.creation_time = current_time()
        if self.cost is None:
            self.cost = compute_plan_cost(self.plan, state=None)

    @property
    def success(self):
        return self.plan is not None

    @property
    def length(self) -> float:
        return get_length(self.plan)

    def as_dict(self) -> dict:
        return get_dataclass_dict(self)

    def dump(self):
        print(
            f"Plan: {self.plan}\nSuccess: {self.success} | Length: {self.length} | Cost:"
            f" {self.cost:.3f} | Optimal: {self.optimal}"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(success={self.success}, cost={self.cost:.3f},"
            f" optimal={self.optimal}, length={self.length},\n  plan={self.plan}))"
        )


def bind_plan(
    plan: List[ActionInstance], mapping: Dict[Constant, Constant]
) -> List[ActionInstance]:
    return [action.bind(mapping) for action in plan]


def is_plan_applicable(state: State, plan: Optional[Plan], goal: Optional[Formula] = None) -> bool:
    if plan is None:
        return False
    for action in plan:
        if not action.is_applicable(state):
            return False
        state = action.apply(state)
    if goal is None:
        return True
    return goal.holds(state)


def compute_plan_cost(
    plan: Optional[Plan], state: Optional[State] = None, operation: Callable[[Any], float] = sum
) -> float:
    if plan is None:
        return INF
    if not plan:
        return 0.0
    costs = []
    for action in plan:
        costs.append(action.cost.evaluate(state))
        if state is not None:
            state = action.apply(state)
    return float(operation(costs))


def apply_actions(
    state: State, actions: Iterable[ActionInstance], initial: bool = False
) -> Iterator[State]:
    if initial:
        yield state
    for action in actions:
        state = action.apply(state)
        yield state


def compute_states(state: State, actions: Iterable[ActionInstance]) -> List[State]:
    return list(apply_actions(state, actions, initial=True))


def compute_supporters(plan: Plan) -> OrderedSet:
    orders = OrderedSet()
    for i, action1 in enumerate(plan):
        for precondition in action1.precondition.simple_clause:
            for j in reversed(range(0, i)):
                action2 = plan[j]
                if precondition in action2.effect.simple_clause:
                    orders.add((action2, action1))
                    break
        for effect in action1.effect.simple_clause:
            for j in reversed(range(0, i)):
                action2 = plan[j]
                if effect in action2.effect.simple_clause:
                    continue
                terms = [evaluation.term for evaluation in action2.effect.simple_clause]
                if effect.term in terms:
                    orders.add((action2, action1))
                    break
    return orders


def compute_threats(plan: Plan) -> OrderedSet:
    orders = OrderedSet()
    for i, action1 in enumerate(plan):
        preimage = {
            evaluation.term: evaluation.value for evaluation in action1.precondition.simple_clause
        }
        for j in range(i + 1, len(plan)):
            action2 = plan[j]
            for effect in action2.effect.simple_clause:
                if preimage.get(effect.term, effect.value) != effect.value:
                    orders.add((action1, action2))
    return orders


def compute_partial_orders(plan: Plan) -> OrderedSet:
    edges = compute_supporters(plan)
    edges.update(compute_threats(plan))
    return transitive_reduction(edges)


def extract_linear_plan(plan: Optional[Plan]) -> Optional[Plan]:
    # NVIDIA
    from schedulestream.language.durative import EndInstance, StartInstance

    if plan is None:
        return None
    index = 0
    linear_plan = []
    while index < len(plan):
        instance = plan[index]
        if isinstance(instance, StartInstance):
            durative_instance = instance.durative_instance
            if (
                (index == len(plan) - 1)
                or not isinstance(plan[index + 1], EndInstance)
                or (plan[index + 1].durative_instance != durative_instance)
            ):
                return None
            linear_plan.append(durative_instance)
            index += 2
        elif isinstance(instance, EndInstance):
            return None
        else:
            raise NotImplementedError(instance)
    return linear_plan


def visualize_plan(plan: Plan, linear: bool = False, **kwargs: Any) -> str:
    # NVIDIA
    from schedulestream.language.durative import EndInstance, OverInstance, StartInstance

    plan = [action for action in plan if not isinstance(action, OverInstance)]
    if linear:
        edges = get_pairs(plan)
    else:
        edges = compute_partial_orders(plan)

    action_colors = {}
    for stream in plan:
        if isinstance(stream, StartInstance):
            action_colors[stream] = "LightYellow"
        elif isinstance(stream, OverInstance):
            action_colors[stream] = "LightBlue"
        elif isinstance(stream, EndInstance):
            action_colors[stream] = "LightSalmon"
    return visualize_graph(edges, vertices=plan, vertex_colors=action_colors, **kwargs)
