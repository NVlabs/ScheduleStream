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
from typing import Any, Callable, Dict, List, Optional

# NVIDIA
from schedulestream.language.argument import Argument, wrap_argument, wrap_arguments
from schedulestream.language.expression import Expression, Formula
from schedulestream.language.function import Evaluation, Function, Term
from schedulestream.language.state import State


class EffectFunction(Function):
    def __init__(
        self,
        *args: Any,
        definition: Optional[Callable[[Any, State], State]] = None,
        input_functions: Optional[List[Function]] = None,
        output_functions: Optional[List[Function]] = None,
        **kwargs: Any,
    ):
        assert definition is not None
        super().__init__(*args, definition=definition, **kwargs)
        self.input_functions = input_functions
        self.output_functions = output_functions

    def instantiate(self, arguments: List[Argument]) -> "EffectTerm":
        arguments = tuple(wrap_arguments(arguments))
        if arguments not in self.instances:
            self.instances[arguments] = EffectTerm(self, list(arguments))
        return self.instances[arguments]


class EffectTerm(Term):
    def __init__(self, function: EffectFunction, arguments: List[Argument]):
        assert isinstance(function, EffectFunction)
        super().__init__(function, arguments)

    def apply(self, state: "State") -> Optional[List["Evaluation"]]:
        assert self.function.definition is not None
        if not self.function.fluent:
            return self._evaluate_static.value
        return self._evaluate_fluent(state).value


class Assignment(Expression):
    def __init__(self, variable: Term, formula: Formula):
        assert isinstance(variable, Term)
        if not isinstance(formula, Expression):
            formula = wrap_argument(formula)
        self.variable = variable
        self.formula = formula

    @property
    def expressions(self) -> List["Expression"]:
        return [self] + self.variable.expressions + self.formula.expressions

    @property
    def is_simple(self):
        return False

    def bind(self, mapping: Dict[Argument, Argument]) -> "Assignment":
        variable = self.variable.bind(mapping)
        formula = self.formula
        if isinstance(formula, Expression):
            formula = formula.bind(mapping)
        return Assignment(variable, formula)

    def apply(self, state: "State") -> Optional[List["Evaluation"]]:
        value = self.formula.evaluate(state)
        return [self.variable.instantiate(value)]

    def __str__(self):
        return f"{self.variable}<={self.formula}"
