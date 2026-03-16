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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

# NVIDIA
from schedulestream.common.utils import (
    apply_binding,
    compute_mapping,
    implies,
    remove_duplicates,
    str_from_sequence,
)
from schedulestream.language.anonymous import Anonymous, rename_anonymous
from schedulestream.language.argument import (
    FUNCTION_PARAMETER,
    Argument,
    InputParameter,
    InputParameters,
    Parameter,
    Parameters,
    RawConstant,
    TypedParameters,
    bind_language,
    create_typed_parameters,
    unwrap_argument,
    unwrap_arguments,
    wrap_argument,
    wrap_arguments,
)
from schedulestream.language.expression import Expression, Formula

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.language.state import State


InputFormula = Optional[Union[List["Formula"], "Formula"]]


def create_type_formula(typed_parameters: TypedParameters, object_type: bool = True) -> Formula:
    type_atoms = {ty(parameter) for parameter, ty in typed_parameters.items() if ty is not None}
    if object_type:
        # NVIDIA
        from schedulestream.language.predicate import Value

        type_atoms.update(Value(parameter) for parameter in typed_parameters.keys())
    # NVIDIA
    from schedulestream.language.connective import Conjunction

    return Conjunction(*type_atoms)


def create_formula(formula: InputFormula) -> Formula:
    # NVIDIA
    from schedulestream.language.connective import Conjunction

    if formula is None:
        formula = []
    if isinstance(formula, list):
        return Conjunction(*formula)
    return formula


class Function(Anonymous):
    def __init__(
        self,
        parameters: InputParameters = None,
        condition: InputFormula = None,
        definition: Optional[Union[Formula, Callable[[Any], Any]]] = None,
        fluents: Optional[List["Function"]] = None,
        fluent: bool = False,
        language: Optional[str] = None,
        default: Any = None,
        lazy: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_definition(definition)
        fluents = list(fluents or [])
        implies(not self.is_procedural, not fluents)
        self.fluents = fluents
        self.fluent = fluent
        self.typed_parameters = create_typed_parameters(parameters)
        self.language = language
        self.instances = {}

        self.default_value = wrap_argument(default)
        self.lazy = lazy

        condition = create_formula(condition)
        condition.assert_atoms()
        relation_parameters = set(self.parameters) | {self.output_parameter}
        assert set(condition.parameters) <= relation_parameters, (
            set(condition.parameters) - relation_parameters
        )
        self._condition = condition

    @property
    def parameters(self) -> Parameters:
        return list(self.typed_parameters.keys())

    @property
    def output_parameter(self) -> Parameter:
        return Parameter(FUNCTION_PARAMETER)

    @property
    def relation_parameters(self) -> Parameters:
        return list(self.parameters) + [self.output_parameter]

    @cached_property
    def condition(self) -> Formula:
        # NVIDIA
        from schedulestream.language.connective import Conjunction
        from schedulestream.language.predicate import Value

        if self == Value:
            return self._condition

        condition = self._condition & create_type_formula(self.typed_parameters)
        formulas = []
        for atom in condition.clause:
            formulas.append(atom)
            formulas.extend(atom.condition.clause)
        return Conjunction(*formulas).flatten()

    @cached_property
    def relation(self) -> "Relation":
        # NVIDIA
        from schedulestream.language.predicate import Relation

        return Relation(self)

    @property
    def is_procedural(self):
        return self.definition is not None

    @property
    def is_simple(
        self,
    ) -> bool:
        return not self.is_procedural

    @property
    def is_static(self):
        return self.is_procedural and not self.fluent

    def set_definition(self, definition: Optional[Union[Formula, Callable[[Any], Any]]]) -> None:
        self.definition = definition

    def instantiate(self, arguments: List[Argument]) -> "Term":
        arguments = tuple(wrap_arguments(arguments))
        if arguments not in self.instances:
            self.instances[arguments] = Term(self, list(arguments))
        return self.instances[arguments]

    def __call__(self, *arguments: Argument) -> "Term":
        return self.instantiate(tuple(arguments))

    def rename(self, variables: Dict[str, Any]) -> "Function":
        rename_anonymous(variables, values=[self])
        return self

    def __lt__(self, other: "Function") -> bool:
        return self.name < other.name

    def __str__(self):
        return f"{self.name}({str_from_sequence(self.parameters)})"

    __repr__ = __str__


class NumericFunction(Function):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, default=0.0, **kwargs)


class Term(Expression):
    def __init__(self, function: Function, arguments: List[Argument]):
        assert isinstance(function, Function), function
        arguments = list(arguments)
        if len(function.parameters) != len(arguments):
            raise ValueError(
                f"Unequal number of arguments\n\tFunction: {function}\n\tArguments: {arguments}"
            )
        self.function = function
        self.arguments = arguments
        self.instances = {}
        self._static_value = None
        self._fluent_values = {}
        self.lazy = self.function.lazy

    @property
    def term(self) -> "Term":
        return self

    @property
    def name(self) -> str:
        return self.function.name

    @property
    def expressions(self) -> List[Expression]:
        return [self] + self.arguments

    @property
    def inputs(self):
        return self.arguments

    @property
    def parameters(self) -> List[Parameter]:
        return [arg for arg in self.arguments if isinstance(arg, Parameter)]

    @cached_property
    def is_recursive(self) -> bool:
        return any(isinstance(arg, Term) for arg in self.arguments)

    @cached_property
    def mapping(self) -> Dict[Parameter, Argument]:
        return compute_mapping(self.function.parameters, self.arguments)

    def get_argument(self, parameter: InputParameter) -> Any:
        return unwrap_argument(self.mapping[parameter])

    def get_arguments(self, parameters: Optional[List[InputParameter]] = None) -> List[Any]:
        if parameters is None:
            parameters = self.parameters
        return list(map(self.get_argument, parameters))

    @property
    def language(self) -> str:
        return bind_language(self.function.language, self.mapping)

    @cached_property
    def condition(self):
        return self.function.condition.bind(self.mapping)

    def unwrap_arguments(self) -> List[Any]:
        return unwrap_arguments(self.arguments)

    def unwrap(self) -> "Term":
        return Term(self.function, self.unwrap_arguments())

    def bind(self, mapping: Dict[Parameter, Argument]) -> "Term":
        arguments = apply_binding(mapping, self.arguments)
        return self.function.instantiate(tuple(arguments))

    def instantiate(self, argument: Argument) -> "Evaluation":
        argument = wrap_argument(argument)
        if argument not in self.instances:
            self.instances[argument] = Evaluation(self, argument)
        return self.instances[argument]

    def ground(self, state: State) -> "Term":
        if not self.is_recursive:
            return self
        arguments = [arg.evaluate(state) for arg in self.arguments]
        return self.function.instantiate(arguments)

    @property
    def is_static(self) -> bool:
        if not self.function.is_static:
            return False
        return all(arg.is_static for arg in self.arguments)

    def is_evaluated(self) -> bool:
        return self.is_static and (self._static_value is not None)

    @property
    def _evaluate_static(self, verbose: bool = False) -> RawConstant:
        assert not self.function.fluent
        self.lazy = False
        if self._static_value is None:
            inputs = unwrap_arguments(self.arguments)
            output = wrap_argument(self.function.definition(*inputs))
            if verbose:
                print(f"{self}={output}")
            self._static_value = output
        return self._static_value

    def _evaluate_fluent(self, state: State, verbose: bool = False) -> RawConstant:
        assert self.function.fluent
        self.lazy = False
        if state not in self._fluent_values:
            inputs = unwrap_arguments(self.arguments)
            output = wrap_argument(self.function.definition(state, *inputs))
            if verbose:
                print(f"{self}={output}")
            self._fluent_values[state] = output
        return self._fluent_values[state]

    def _evaluate(self, state: Optional[State]) -> RawConstant:
        if self.is_recursive:
            return self.ground(state)._evaluate(state)
        if not self.function.is_procedural:
            assert state is not None, self
            value = state.get_value(self)
        elif self.lazy:
            assert not self.function.fluent
            value = self.function.default_value
        elif not self.function.fluent:
            value = self._evaluate_static
        else:
            assert state is not None, self
            value = self._evaluate_fluent(state)
        return value

    def evaluate(self, state: Optional[State] = None) -> RawConstant:
        if self.is_recursive:
            return self.ground(state).evaluate(state)
        return self._evaluate(state).value

    def equals(self, value: Any) -> "Evaluation":
        return self.instantiate(value)

    @cached_property
    def default(self) -> "Evaluation":
        return self.equals(self.function.default_value)

    def evaluation(self, state: State) -> "Evaluation":
        value = self.evaluate(state)
        return self.equals(value)

    def support(self, state: State) -> List["Evaluation"]:
        if self.is_recursive:
            support = list(self.ground(state).support(state))
            for arg in self.arguments:
                support.extend(arg.support(state))
            return support
        return [self.evaluation(state)]

    def __le__(self, other: Any) -> "Evaluation":
        return self.equals(other)

    def __eq__(self, other: Any) -> Union[bool, "Evaluation"]:
        if any(isinstance(other, cls) for cls in [Term, Formula]):
            return super().__eq__(other)
        return self.equals(other)

    def __hash__(self) -> int:
        return super().__hash__()

    def __lt__(self, other: "Term") -> bool:
        return self.function < other.function

    def __str__(self) -> str:
        return f"{self.function.name}({str_from_sequence(self.arguments)})"

    __repr__ = __str__


class Equation(Formula):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    @property
    def expressions(self) -> List[Expression]:
        return [self] + self.left.expressions + self.right.expressions

    def evaluate(self, state: Optional[State] = None) -> RawConstant:
        left_value = self.left.evaluate(state)
        right_value = self.right.evaluate(state)
        return wrap_argument(left_value) == wrap_argument(right_value)

    def support(self, state: State) -> List["Evaluation"]:
        return remove_duplicates(self.left.support(state) + self.right.support(state))


class Evaluation(Equation):
    def __init__(self, term: Term, value: Any):
        assert isinstance(term, Term), term
        super().__init__(term, value)

    @property
    def term(self) -> Term:
        return self.left

    @property
    def value(self) -> Any:
        return self.right

    @property
    def function(self) -> Function:
        return self.term.function

    @property
    def name(self) -> str:
        return self.function.name

    @property
    def is_default(self) -> bool:
        return self.value == self.function.default_value

    @property
    def is_static(self) -> bool:
        return self.left.is_static and self.right.is_static

    @property
    def inputs(self) -> List[Argument]:
        return self.term.arguments

    @property
    def output(self) -> Argument:
        return self.value

    @property
    def outputs(self) -> List[Argument]:
        return [self.output]

    @cached_property
    def literal(self):
        arguments = self.inputs + self.outputs
        term = self.function.relation.instantiate(arguments)
        return term.atom()

    @cached_property
    def mapping(self) -> Dict[Parameter, Argument]:
        mapping = dict(self.term.mapping)
        mapping[self.function.output_parameter] = self.value
        return mapping

    @cached_property
    def condition(self) -> Formula:
        return self.function.condition.bind(self.mapping)

    @property
    def language(self) -> Optional[str]:
        return bind_language(self.function.language, self.mapping)

    def unwrap(self) -> "Evaluation":
        return Evaluation(self.term.unwrap(), self.value.unwrap())

    def bind(self, mapping: Dict[Argument, Argument]) -> "Evaluation":
        term = self.term.bind(mapping)
        value = mapping.get(self.value, self.value)
        return term.equals(value)

    def evaluate(self, state: Optional[State] = None) -> RawConstant:
        return self.term._evaluate(state) == self.value

    def apply(self, state: State) -> Optional[List["Evaluation"]]:
        return self.evaluations

    def __lt__(self, other: "Evaluation") -> bool:
        return self.term < other.term

    def __str__(self):
        return f"{self.term}={str(self.value)}"

    __repr__ = __str__
