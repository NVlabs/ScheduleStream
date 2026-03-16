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
import itertools
from typing import Any, Callable, List

# NVIDIA
from schedulestream.algorithm.instantiation import instantiate_actions
from schedulestream.algorithm.utils import Plan, filter_applicable
from schedulestream.common.ordered_set import OrderedSet
from schedulestream.common.utils import flatten, remove_duplicates
from schedulestream.language.problem import ProblemABC
from schedulestream.language.state import State

SuccessorFn = Callable[[State], List[Plan]]


def create_immediate_successor_fn(problem: ProblemABC, applicable: bool = False) -> SuccessorFn:
    def successor_fn(state: State) -> List[Plan]:
        actions = instantiate_actions(state, problem.actions)
        if applicable:
            actions = filter_applicable(state, actions)
        return [[action] for action in actions]

    return successor_fn


def create_offline_successor_fn(problem: ProblemABC) -> SuccessorFn:
    problem = problem.instantiate()

    free_actions = []
    evaluations = remove_duplicates(
        flatten(action.precondition.simple_clause for action in problem.actions)
    )
    default_evaluations = OrderedSet()
    for evaluation in evaluations:
        evaluation.successors = []
    for action in problem.actions:
        preconditions = action.precondition.simple_clause
        action.num_preconditions = len(preconditions)
        action.num_missing = action.num_preconditions
        if not preconditions:
            free_actions.append(action)
        for evaluation in preconditions:
            evaluation.successors.append(action)
            if evaluation.is_default:
                default_evaluations.add(evaluation)

    def successor_fn(state: State) -> List[Plan]:
        modified = OrderedSet()
        successors = OrderedSet(free_actions)
        evaluations = list(state.evaluations)
        for evaluation in default_evaluations:
            if state.holds(evaluation):
                evaluations.append(evaluation)
        for evaluation in evaluations:
            for action in getattr(evaluation, "successors", []):
                action.num_missing -= 1
                modified.add(action)
                if action.num_missing == 0:
                    successors.add(action)
        for action in modified:
            action.num_missing = action.num_preconditions
        return [[action] for action in successors]

    return successor_fn


def create_relaxed_successor_fn(problem: ProblemABC, **kwargs: Any) -> SuccessorFn:
    # NVIDIA
    from schedulestream.algorithm.heuristics import create_relaxed_plan_fn

    relaxed_plan_fn = create_relaxed_plan_fn(problem, **kwargs)

    def successor_fn(state: State) -> List[Plan]:
        relaxed_plan, _ = relaxed_plan_fn(state)
        if relaxed_plan is None:
            return []
        return [relaxed_plan]

    return successor_fn


def combine_successor_fns(create_successor_fns):
    def create_successor_fn(problem: ProblemABC) -> SuccessorFn:
        successor_fns = [fn(problem) for fn in create_successor_fns]
        successor_fn = lambda state: remove_duplicates(
            itertools.chain.from_iterable(map(tuple, fn(state)) for fn in successor_fns)
        )
        return successor_fn

    return create_successor_fn


SUCCESSORS = {
    "online": create_immediate_successor_fn,
    "offline": create_offline_successor_fn,
    "relaxed": create_relaxed_successor_fn,
    "relaxed-offline": combine_successor_fns(
        [create_relaxed_successor_fn, create_offline_successor_fn]
    ),
}
