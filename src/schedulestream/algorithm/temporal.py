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
import operator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, Optional, Set

# NVIDIA
from schedulestream.algorithm.utils import PartialPlan, Plan, Solution, compute_partial_orders
from schedulestream.common.graph import get_ancestors, get_descendants
from schedulestream.common.utils import INF, flatten, merge_dicts
from schedulestream.language.action import ActionInstance
from schedulestream.language.connective import Conjunction
from schedulestream.language.durative import (
    CLOCK,
    DurativeInstance,
    EndInstance,
    OverInstance,
    StartInstance,
    TimedAction,
)
from schedulestream.language.effect import EffectFunction
from schedulestream.language.function import Evaluation
from schedulestream.language.problem import InstantiatedProblem
from schedulestream.language.state import State
from schedulestream.language.utils import INTERNAL_PREFIX

LinearPlan = List[ActionInstance]
TimedPlan = List[TimedAction]


def sequential_from_temporal(
    problem: InstantiatedProblem, over: bool = True
) -> InstantiatedProblem:
    instance_from_ongoing = {}
    for instance in problem.actions:
        if isinstance(instance, DurativeInstance):
            assert instance.ongoing_atom not in instance_from_ongoing
            instance_from_ongoing[instance.ongoing_atom] = instance
    initial = problem.initial

    def over_fn(state: State) -> Optional[List[Evaluation]]:
        for evaluation in state.evaluations:
            if evaluation in instance_from_ongoing:
                ongoing_instance = instance_from_ongoing[evaluation]
                if not ongoing_instance.over_instance.is_applicable(state):
                    return None
        return []

    over_effect = EffectFunction(
        parameters=[],
        name=f"{INTERNAL_PREFIX}over",
        definition=over_fn,
        fluent=True,
    ).instantiate([])

    actions = list(flatten(action.instances for action in problem.actions))
    if over:
        for action in actions:
            action.effect = Conjunction(action.effect, over_effect)

    return InstantiatedProblem(
        initial, actions, problem.goal, problem=problem.problem, parent=problem
    )


def time_plan(plan: Optional[Plan], epsilon: float = CLOCK) -> Optional[TimedPlan]:
    if plan is None:
        return None
    timed_plan = []
    start_times = {}
    current_time = 0.0
    for i, instance in enumerate(plan):
        if isinstance(instance, StartInstance):
            durative_action = instance.action.durative_action
            durative_instance = durative_action.instantiate(instance.arguments)
            assert durative_instance not in start_times
            start_times[durative_instance] = current_time
        elif isinstance(instance, OverInstance):
            continue
        elif isinstance(instance, EndInstance):
            durative_action = instance.action.durative_action
            durative_instance = durative_action.instantiate(instance.arguments)
            assert durative_instance in start_times
            elapsed_time = current_time - start_times[durative_instance]
            duration = durative_instance.min_duration.evaluate()
            remaining_time = max(0.0, duration - elapsed_time)
            current_time += remaining_time
            timed_plan.append(
                TimedAction(
                    action=durative_instance,
                    start=start_times[durative_instance],
                    end=current_time,
                )
            )
            start_times.pop(durative_instance)
        elif isinstance(instance, DurativeInstance):
            duration = instance.min_duration.evaluate()
            timed_plan.append(
                TimedAction(
                    action=instance,
                    start=current_time,
                    end=current_time + duration,
                )
            )
            current_time += duration
        else:
            timed_plan.append(
                TimedAction(
                    action=instance,
                    start=current_time,
                    end=current_time,
                )
            )
        current_time += epsilon
    return timed_plan


def recover_plan(timed_plan: Optional[TimedPlan]) -> Optional[TimedPlan]:
    if timed_plan is None:
        return timed_plan
    return [timed_action.root for timed_action in timed_plan]


def retime_plan(timed_plan: Optional[TimedPlan]) -> Optional[TimedPlan]:
    sequential_plan = sequential_from_timed(timed_plan)
    return time_plan(sequential_plan)


def serialize_plan(sequential_plan: Optional[Plan]) -> Optional[Plan]:
    if sequential_plan is None:
        return sequential_plan
    return list(flatten(action.instances for action in sequential_plan))


def actions_from_timed(timed_plan: Optional[TimedPlan]) -> Optional[LinearPlan]:
    if timed_plan is None:
        return None
    return [timed_action.action for timed_action in timed_plan]


def sequential_from_timed(
    timed_plan: Optional[TimedPlan], over: bool = True
) -> Optional[LinearPlan]:
    if timed_plan is None:
        return None
    event_plan = []
    for timed_instance in timed_plan:
        instance = timed_instance.action
        if isinstance(instance, DurativeInstance):
            event_plan.append((timed_instance.start, instance.start_instance))
            event_plan.append((timed_instance.end, instance.end_instance))
        else:
            assert timed_instance.start == timed_instance.end
            event_plan.append((timed_instance.start, instance))
    event_plan.sort(key=operator.itemgetter(0))

    sequential_plan = []
    for event, instance in event_plan:
        sequential_plan.append(instance)
        if not over:
            continue
        for timed_instance in timed_plan:
            instance = timed_instance.action
            if isinstance(instance, DurativeInstance):
                if timed_instance.start <= event < timed_instance.end:
                    sequential_plan.append(instance.over_instance)
    return sequential_plan


def compute_unordered_actions(
    timed_plan: TimedPlan, durative_action: DurativeInstance
) -> Set[DurativeInstance]:
    durative_actions = actions_from_timed(timed_plan)
    plan = sequential_from_timed(timed_plan, over=False)
    partial_orders = compute_partial_orders(plan)

    partially_ordered = []
    start_action = durative_action.start_instance
    for end_action2 in get_ancestors(partial_orders, source_vertices=[start_action]):
        if isinstance(end_action2, EndInstance):
            partially_ordered.append(end_action2.durative_instance)
    end_action = durative_action.end_instance
    for start_action2 in get_descendants(partial_orders, source_vertices=[end_action]):
        if isinstance(start_action2, StartInstance):
            partially_ordered.append(start_action2.durative_instance)
    return set(durative_actions) - set(partially_ordered) - {durative_action}


def get_makespan(plan: Optional[TimedPlan]) -> float:
    if plan is None:
        return INF
    if not plan:
        return 0.0
    return max(timed_action.end for timed_action in plan)


@dataclass
class TemporalSolution(Solution):
    plan: Optional[TimedPlan] = None

    @property
    def timed_plan(self) -> Optional[TimedPlan]:
        return self.plan

    @property
    def makespan(self) -> float:
        return get_makespan(self.plan)

    @property
    def duration(self) -> float:
        return self.makespan

    @property
    def score(self) -> float:
        return self.makespan + self.cost

    @property
    def linear_plan(self) -> Optional[LinearPlan]:
        return actions_from_timed(self.timed_plan)

    @property
    def sequential_plan(self) -> Optional[LinearPlan]:
        return sequential_from_timed(self.plan)

    @cached_property
    def partial_plan(self) -> Optional[PartialPlan]:
        if self.plan is None:
            return None
        actions = [timed_action.action for timed_action in self.timed_plan]
        return PartialPlan(actions=actions)

    @cached_property
    def sequential(self) -> Solution:
        return Solution(
            plan=self.linear_plan,
            cost=self.cost,
            optimal=self.optimal,
            discovery_time=self.discovery_time,
        )

    @staticmethod
    def from_solution(solution: Solution, **kwargs: Any) -> "TemporalSolution":
        return TemporalSolution(**merge_dicts(solution.as_dict(), kwargs))

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(success={self.success}, makespan={self.makespan:.3f},"
            f" cost={self.cost:.3f}, optimal={self.optimal},"
            f" length={self.length},\n  plan={self.plan}))"
        )
