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
from typing import List, Optional, Set

# NVIDIA
from schedulestream.common.utils import flatten, remove_duplicates
from schedulestream.language.argument import wrap_argument
from schedulestream.language.connective import Conjunction, Sequence
from schedulestream.language.expression import Formula
from schedulestream.language.function import Function, Term
from schedulestream.language.state import State

INTERNAL_PREFIX = "_"


def is_internal(name: str) -> bool:
    return name.startswith(INTERNAL_PREFIX)


def get_fluent_functions(actions: List["Action"]) -> List[Function]:
    return remove_duplicates(flatten(action.fluents for action in actions))


def infer_state(state: State) -> State:
    evaluations = []
    for evaluation in state:
        evaluations.extend(evaluation.condition.clause)
    return state.new_state(evaluations)


def simplify_expression(expression, state: State, fluents: Optional[Set[Term]] = None):
    fluents = fluents or set()
    if not expression.is_simple:
        return expression
    if set(expression.terms) & fluents:
        return expression
    return wrap_argument(expression.evaluate(state))


def simplify_conjunction(
    condition: Formula, state: State, fluents: Optional[Set[Term]] = None
) -> Optional[Conjunction]:
    if isinstance(condition, Sequence):
        return condition
    fluents = fluents or set()
    conjunction = []
    for evaluation in condition.clause:
        if not evaluation.is_simple and not evaluation.term.is_evaluated():
            conjunction.append(evaluation)
        elif set(evaluation.terms) & fluents:
            conjunction.append(evaluation)
        elif not evaluation.holds(state):
            return None
    return Conjunction(*conjunction)
