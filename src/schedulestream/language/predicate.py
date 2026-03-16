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
from schedulestream.language.argument import (
    PARAMETER_PREFIX,
    Argument,
    Constant,
    InputParameters,
    Parameter,
    create_typed_parameters,
    is_parameter,
    wrap_argument,
    wrap_arguments,
)
from schedulestream.language.function import Evaluation, Function, Term

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.language.generator import ConditionalGenerator
    from schedulestream.language.stream import PredicateStream


class Predicate(Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, default=False, **kwargs)
        self.parent = None

    @cached_property
    def relation(self) -> "Predicate":
        return self

    def instantiate(self, arguments: List[Argument]) -> "PredicateTerm":
        arguments = tuple(wrap_arguments(arguments))
        if arguments not in self.instances:
            self.instances[arguments] = PredicateTerm(self, list(arguments))
        return self.instances[arguments]

    def atom(self, arguments: List[Argument]) -> "Atom":
        term = self.instantiate(arguments)
        return term.atom()

    def __call__(self, *arguments: Argument) -> "Atom":
        if (
            (len(arguments) == 1)
            and (len(arguments) != self.parameters)
            and is_parameter(arguments[0])
        ):
            arguments = arguments[0].split(" ")
        return self.atom(list(arguments))

    def stream(
        self,
        conditional_generator: Optional[ConditionalGenerator],
        inputs: InputParameters,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> PredicateStream:
        # NVIDIA
        from schedulestream.language.stream import BatchPredicateStream, PredicateStream

        if is_parameter(inputs):
            inputs = inputs.split(" ")
        inputs = create_typed_parameters(inputs)
        if (batch_size is None) or (conditional_generator is None):
            return PredicateStream(conditional_generator, predicate=self, inputs=inputs, **kwargs)
        return BatchPredicateStream(
            conditional_generator, predicate=self, inputs=inputs, batch_size=batch_size, **kwargs
        )


class Type(Predicate):
    def __init__(self, parameter=f"{PARAMETER_PREFIX}value", **kwargs):
        super().__init__(parameters={parameter: None}, **kwargs)


class Proposition(Predicate):
    def __init__(self, **kwargs):
        super().__init__(parameters=[], **kwargs)


class Relation(Predicate):
    def __init__(self, function: Function):
        super().__init__(
            function.relation_parameters, name=function._name, condition=function._condition
        )
        self.parent = function


class PredicateTerm(Term):
    def __init__(self, function: Predicate, arguments: List[Argument]):
        assert isinstance(function, Predicate)
        super().__init__(function, arguments)

    @property
    def predicate(self) -> Predicate:
        return self.function

    def instantiate(self, argument: Constant) -> "Literal":
        argument = wrap_argument(argument)
        if argument not in self.instances:
            if argument.value is True:
                self.instances[argument] = Atom(self)
            elif argument.value is False:
                self.instances[argument] = NegatedAtom(self)
            else:
                raise ValueError(
                    f"Argument is not True or False\n\tPredicate term: {self}\n\tConstant:"
                    f" {argument}"
                )
        return self.instances[argument]

    def atom(self) -> Evaluation:
        return self.equals(value=True)


class Literal(Evaluation):
    def __init__(self, term: PredicateTerm):
        assert isinstance(term, PredicateTerm)
        assert isinstance(self._value, Constant), self._value
        super().__init__(term, self._value)

    @cached_property
    def literal(self):
        return self

    @cached_property
    def mapping(self) -> Dict[Parameter, Argument]:
        return self.term.mapping

    def bind(self, mapping: Dict[Argument, Argument]) -> "Literal":
        term = self.term.bind(mapping)
        return term.equals(self._value)

    def __invert__(self) -> "Literal":
        return self.term.equals(not self._value.value)


class Atom(Literal):
    _value = wrap_argument(True)

    def __str__(self):
        return str(self.term)

    __repr__ = __str__


class NegatedAtom(Literal):
    _value = wrap_argument(False)

    def __str__(self):
        return f"~{self.term}"

    __repr__ = __str__


Value = Type(name="Value")
