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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

# NVIDIA
from schedulestream.common.ordered_set import OrderedSet
from schedulestream.common.utils import flatten, join_prefix, remove_duplicates
from schedulestream.language.argument import Argument, Parameter, Parameters, RawConstant
from schedulestream.language.expression import Expression, Formula
from schedulestream.language.function import Evaluation

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.language.state import State


class Negation(Formula):
    def __init__(self, formula):
        self.formula = formula

    def bind(self, mapping: Dict[Parameter, Argument]) -> Formula:
        return self.__class__(self.formula.bind(mapping))

    def evaluate(self, state: Optional[State] = None) -> RawConstant:
        return not self.formula.evaluate(state)

    def flatten(self) -> Formula:
        flattened = self.formula.flatten()
        if isinstance(flattened, Negation):
            return flattened.formula
        return self

    def __str__(self):
        return f"~{self.formula}"

    __repr__ = __str__


class Connective(Formula):
    _operation = None
    _symbol = None
    _english = None

    def __init__(self, *formulas: Formula):
        self.formulas = tuple(formulas)

    @property
    def expressions(self) -> List[Expression]:
        return [self] + list(flatten(formula.expressions for formula in self.formulas))

    def bind(self, mapping: Dict[Parameter, Argument]) -> Formula:
        formulas = [formula.bind(mapping) for formula in self.formulas]
        return self.__class__(*formulas)

    def flatten(self) -> Formula:
        formulas = []
        for formula in self.formulas:
            formula = formula.flatten()
            if type(formula) == type(self):
                formulas.extend(formula.formulas)
            else:
                formulas.append(formula)
        if len(formulas) == 1:
            return formulas[0]
        return self.__class__(*formulas)

    def evaluate(self, state: Optional[State] = None) -> RawConstant:
        return self._operation(formula.evaluate(state) for formula in self.formulas)

    @property
    def language(self) -> Optional[str]:
        languages = [formula.language for formula in self.formulas if formula.language is not None]
        if not languages:
            return None
        return f" {self._english} ".join(languages)

    def __str__(self):
        return f" {self._symbol} ".join(map(str, self.formulas))

    __repr__ = __str__


class Conjunction(Connective):
    _operation = all
    _symbol = "&"
    _english = "and"

    @property
    def clause(self) -> List[Formula]:
        clause = []
        for formula in self.formulas:
            clause.extend(formula.clause)
        return clause

    @property
    def condition(self) -> "Conjunction":
        evaluations = []
        for formula in self.formulas:
            evaluations.append(formula.condition)
        return self.__class__(*evaluations).flatten()

    def support(self, state: State) -> List[Evaluation]:
        return remove_duplicates(flatten(formula.support(state) for formula in self.formulas))

    def apply(self, state: State) -> Optional[List[Evaluation]]:
        evaluations = []
        for formula in self.formulas:
            new_evaluations = formula.apply(state)
            if new_evaluations is None:
                return None
            evaluations.extend(new_evaluations)
        return remove_duplicates(evaluations)


And = Conjunction


class Sequence(Conjunction):
    def __init__(self, sequence: List[Formula]):
        assert sequence
        super().__init__(sequence[-1])
        self.sequence = sequence

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.sequence})"

    __repr__ = __str__


class CumulativeSequence(Sequence):
    def __init__(self, parts: List[Formula]):
        assert parts
        formulas = [parts[0]]
        for formula in parts[1:]:
            formulas.append(formulas[-1] & formula)
        super().__init__(formulas)
        self.parts = parts

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.parts})"

    __repr__ = __str__


class Disjunction(Connective):
    _operation = any
    _symbol = "|"
    _english = "or"


Or = Disjunction


class Implication(Disjunction):
    def __init__(self, formula1: Formula, formula2: Formula):
        super().__init__(~formula1, formula2)
        self.formula1 = formula1
        self.formula2 = formula2

    @property
    def antecedent(self) -> Formula:
        return self.formula1

    @property
    def consequent(self) -> Formula:
        return self.formula2


Implies = Implication


class Sum(Connective):
    _operation = sum
    _symbol = "+"
    _english = "plus"

    def support(self, state: State) -> List[Evaluation]:
        return remove_duplicates(flatten(formula.support(state) for formula in self.formulas))


def conjunction_difference(condition1: Conjunction, condition2: Conjunction) -> Conjunction:
    return Conjunction(*OrderedSet(condition1.clause).difference(condition2.clause))


def partition_condition(
    evaluation: Evaluation, inputs: Parameters
) -> Tuple[Conjunction, Conjunction]:
    input_conditions = []
    output_conditions = [evaluation]
    for atom in evaluation.condition.clause:
        if set(atom.parameters) <= set(inputs):
            input_conditions.append(atom)
        else:
            output_conditions.append(atom)
    input_condition = Conjunction(*input_conditions)
    output_condition = Conjunction(*output_conditions)
    return input_condition, output_condition
