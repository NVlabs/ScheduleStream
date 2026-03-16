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
from __future__ import annotations

# Standard Library
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# NVIDIA
from schedulestream.common.utils import remove_duplicates

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.language.argument import Argument, Constant, Parameter, RawConstant
    from schedulestream.language.connective import Conjunction, Sum
    from schedulestream.language.function import Evaluation, Function, Term
    from schedulestream.language.state import State


class Expression:
    @property
    def expressions(self) -> List["Expression"]:
        raise NotImplementedError(self)

    @property
    def parameters(self) -> List[Parameter]:
        # NVIDIA
        from schedulestream.language.argument import Parameter

        return remove_duplicates(arg for arg in self.expressions if isinstance(arg, Parameter))

    @property
    def constants(self) -> List[Constant]:
        # NVIDIA
        from schedulestream.language.argument import Constant

        return remove_duplicates(arg for arg in self.expressions if isinstance(arg, Constant))

    @property
    def terms(self) -> List[Term]:
        # NVIDIA
        from schedulestream.language.function import Term

        return remove_duplicates(
            expression for expression in self.expressions if isinstance(expression, Term)
        )

    @property
    def evaluations(self) -> List[Evaluation]:
        # NVIDIA
        from schedulestream.language.function import Evaluation

        return remove_duplicates(
            expression for expression in self.expressions if isinstance(expression, Evaluation)
        )

    @property
    def functions(self) -> List[Term]:
        return remove_duplicates(term.function for term in self.terms)

    @property
    def grounded(self) -> bool:
        return not self.parameters

    @property
    def is_static(self) -> bool:
        raise NotImplementedError(type(self))

    @cached_property
    def is_simple(self) -> bool:
        return all(function.is_simple for function in self.functions)

    @property
    def clause(self) -> List["Expression"]:
        return [self]

    @property
    def conjunction(self) -> Conjunction:
        # NVIDIA
        from schedulestream.language.connective import Conjunction

        return Conjunction(*self.clause).flatten()

    @cached_property
    def simple_clause(self) -> List[Evaluation]:
        return [evaluation for evaluation in self.clause if evaluation.is_simple]

    @cached_property
    def simple_conjunction(self) -> Conjunction:
        # NVIDIA
        from schedulestream.language.connective import Conjunction

        return Conjunction(*self.simple_clause)

    def remove_functions(self, functions: List[Function]) -> Conjunction:
        # NVIDIA
        from schedulestream.language.connective import Conjunction
        from schedulestream.language.function import Evaluation

        evaluations = []
        for evaluation in self.clause:
            if not isinstance(evaluation, Evaluation) or (evaluation.function not in functions):
                evaluations.append(evaluation)
        return Conjunction(*evaluations)

    def assert_atoms(self) -> None:
        # NVIDIA
        from schedulestream.language.predicate import Atom

        for atom in self.clause:
            if not isinstance(atom, Atom):
                raise NotImplementedError(atom)

    def assert_conjunction(self) -> None:
        # NVIDIA
        from schedulestream.language.function import Evaluation

        for evaluation in self.clause:
            if not isinstance(evaluation, Evaluation):
                raise NotImplementedError(evaluation)

    def flatten(self) -> "Expression":
        return self

    def bind(self, mapping: Dict[Parameter, Argument]) -> "Expression":
        raise NotImplementedError(self)

    def evaluate(self, state: Optional[State] = None) -> RawConstant:
        raise NotImplementedError(self)

    def holds(self, state: State) -> bool:
        value = self.evaluate(state)
        assert (value is True) or (value is False), (self, value)
        return bool(value)

    def support(self, state: State) -> List[Evaluation]:
        raise NotImplementedError(self)


class Formula(Expression):
    def apply(self, state: State) -> Optional[List[Evaluation]]:
        raise NotImplementedError(self)

    def __and__(self, other: "Formula") -> "Formula":
        # NVIDIA
        from schedulestream.language.connective import Conjunction

        return Conjunction(self, other).flatten()

    def __or__(self, other: "Formula") -> "Formula":
        # NVIDIA
        from schedulestream.language.connective import Disjunction

        return Disjunction(self, other).flatten()

    def __invert__(self) -> "Formula":
        # NVIDIA
        from schedulestream.language.connective import Negation

        return Negation(self).flatten()

    def __add__(self, other: "Formula") -> "Formula":
        # NVIDIA
        from schedulestream.language.argument import wrap_argument
        from schedulestream.language.connective import Sum

        if not isinstance(other, Formula):
            other = wrap_argument(other)

        return Sum(self, other).flatten()

    def __radd__(self, other: "Formula") -> "Formula":
        return self + other
