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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# NVIDIA
from schedulestream.common.utils import is_hashable
from schedulestream.language.expression import Expression, Formula

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.language.function import Evaluation
    from schedulestream.language.state import State

PARAMETER_PREFIX = "?"
FUNCTION_PARAMETER = f"{PARAMETER_PREFIX}output"
OUTPUT = FUNCTION_PARAMETER
RawConstant = Any


def is_parameter(value: Union[Any, str]) -> bool:
    return isinstance(value, str) and value.startswith(PARAMETER_PREFIX)


def assert_parameter(parameter: Union[Any, str]) -> None:
    assert is_parameter(parameter), parameter


def assert_parameters(parameters: List[Union[Any, str]]):
    for parameter in parameters:
        assert_parameter(parameter)


class Parameter(str, Expression):
    def __new__(cls, name, *args, **kwargs):
        if isinstance(name, Parameter):
            return name
        assert not isinstance(name, Constant), name
        parameter = str.__new__(cls, name, *args, **kwargs)
        assert is_parameter(parameter), parameter
        return parameter

    @property
    def expressions(self) -> List[Expression]:
        return [self]

    def bind(self, mapping: Dict["Parameter", "Argument"]) -> Formula:
        return mapping.get(self, self)

    def evaluate(self, state: Optional[State] = None) -> RawConstant:
        raise RuntimeError()

    def support(self, state: State) -> List[Evaluation]:
        raise RuntimeError()

    def __repr__(self):
        return str(self)


InputParameter = Union[Parameter, str]
Parameters = List[str]
TypedParameters = Dict[str, "Type"]
InputParameters = Optional[Union[str, Parameters, TypedParameters]]


def create_typed_parameters(
    parameters: InputParameters,
) -> TypedParameters:
    if parameters is None:
        parameters = []
    elif isinstance(parameters, str):
        parameters = parameters.split(" ")
    if isinstance(parameters, dict):
        typed_parameters = parameters
    else:
        typed_parameters = {parameter: None for parameter in parameters}
    return {Parameter(parameter): ty for parameter, ty in typed_parameters.items()}


class Constant(Expression):
    _from_id = {}
    _from_value = {}

    def __init__(self, entity: Any):
        assert not isinstance(entity, self.__class__)
        value = Constant.get_value(entity)
        if value is not None:
            return
        self.value = entity
        self.num = len(self._from_id)
        self._from_id[type(entity), id(entity)] = self
        if is_hashable(entity):
            self._from_value[type(entity), entity] = self

    def __new__(cls, entity):
        value = cls.get_value(entity)
        if value is not None:
            return value
        return super().__new__(cls)

    @property
    def expressions(self) -> List[Expression]:
        return [self]

    def unwrap(self) -> Any:
        return self.value

    def bind(self, mapping: Dict[Parameter, "Argument"]) -> Formula:
        return self

    @property
    def is_static(self) -> bool:
        return True

    def evaluate(self, state: Optional[State] = None) -> RawConstant:
        return self.value

    def support(self, state: State) -> List[Evaluation]:
        return []

    @staticmethod
    def get_value(entity: Any) -> Optional["Constant"]:
        if is_hashable(entity):
            key = (type(entity), entity)
            if key in Constant._from_value:
                return Constant._from_value[key]
        else:
            key = (type(entity), id(entity))
            if key in Constant._from_id:
                return Constant._from_id[key]
        return None

    @staticmethod
    def to_value(entity: Any) -> "Constant":
        return Constant(entity)

    def __lt__(self, other: "Constant") -> bool:
        return self.num < other.num

    def __repr__(self):
        return str(self.value)

    __str__ = __repr__


WrappedArgument = Union[Parameter, Constant]
Argument = Union[str, Any, WrappedArgument]


def wrap_argument(argument: Argument) -> WrappedArgument:
    if isinstance(argument, Expression):
        return argument
    if is_parameter(argument):
        return Parameter(argument)
    return Constant(argument)


def wrap_arguments(arguments: List[Argument]) -> List[WrappedArgument]:
    return list(map(wrap_argument, arguments))


def unwrap_argument(argument: WrappedArgument) -> Any:
    return argument.unwrap()


def unwrap_arguments(arguments: List[WrappedArgument]) -> List[Any]:
    return list(map(unwrap_argument, arguments))


def bind_language(
    language: Optional[str], parameter_values: Dict[Parameter, Constant]
) -> Optional[str]:
    if language is None:
        return language
    for parameter in sorted(parameter_values, key=len, reverse=True):
        if parameter in language:
            value = parameter_values[parameter].value
            language = language.replace(parameter, value)
    return language
