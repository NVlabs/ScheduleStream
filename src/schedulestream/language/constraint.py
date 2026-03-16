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
from functools import cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# NVIDIA
from schedulestream.common.utils import Key
from schedulestream.language.argument import Argument
from schedulestream.language.connective import Conjunction
from schedulestream.language.expression import Expression, Formula
from schedulestream.language.function import Function

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.language.stream import Context


class Cost(Key):
    def __init__(
        self,
        expression: Expression,
        static: bool = False,
    ):
        super().__init__(expression)
        self.expression = expression
        self.static = static
        self.priority = 0
        self.outputs = []
        self.iterations = 1
        self.exhausted = True

    @property
    def term(self) -> Expression:
        return self.expression

    @property
    def function(self) -> Function:
        return self.expression.function

    @property
    def is_procedural(self):
        return self.function.is_procedural

    @property
    def is_static(self):
        return self.function.is_static

    @property
    def input_condition(self) -> Formula:
        return self.expression.condition

    @property
    def output_condition(self) -> Formula:
        return Conjunction()

    @property
    def root(self) -> "Cost":
        return self

    def bind(self, mapping: Dict[Argument, Argument]) -> "Cost":
        expression = self.expression.bind(mapping)
        return self.__class__(expression)

    @cache
    def evaluate(self) -> Any:
        self.expression.term.lazy = False
        return self.expression.term.evaluate()

    def get_iterations(self, context: Optional[Context] = None) -> int:
        assert context is None
        return self.iterations

    def get_outputs(self, iteration: int, **kwargs: Any) -> List["Cost"]:
        if iteration == 0:
            return [self]
        return []

    def __str__(self) -> str:
        return f"+{self.expression}"

    __repr__ = __str__


class Constraint(Cost):
    @property
    def term(self):
        return self.expression.term

    @cache
    def evaluate(self) -> Any:
        self.expression.term.lazy = False
        return self.expression.evaluate()

    def get_outputs(self, iteration: int, **kwargs: Any) -> List["Constraint"]:
        if iteration == 0 and self.evaluate():
            return [self]
        return []

    def __str__(self) -> str:
        return f"{self.expression}"

    __repr__ = __str__
