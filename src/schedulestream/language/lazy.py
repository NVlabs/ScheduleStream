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
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, Dict, Hashable, Optional

# NVIDIA
from schedulestream.common.utils import compute_mapping, key_from_kwargs
from schedulestream.language.action import DEFAULT_COST
from schedulestream.language.argument import (
    PARAMETER_PREFIX,
    Constant,
    Parameter,
    Parameters,
    unwrap_arguments,
    wrap_argument,
    wrap_arguments,
)
from schedulestream.language.function import Function
from schedulestream.language.generator import ConditionalGenerator, Output, from_fn

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.language.stream import Stream, StreamInstance


def lazy_function(function: Function) -> None:
    procedural = function.definition

    def lazy_fn(*args: Any, **kwargs: Any) -> Any:
        if any(isinstance(arg, LazyOutput) for arg in args):
            return function.default_value
        return procedural(*args, **kwargs)

    function.definition = lazy_fn


class LazyOutput:
    _cache = {}

    def __init__(
        self,
        stream: Stream,
        input_values: Dict[Parameter, Any],
        output: Parameter,
        value: Optional[Any] = None,
    ) -> None:
        self.stream = stream
        self.input_values = input_values
        self.output = output
        self.value = value

    def reset(self) -> None:
        self._cache.clear()

    @cached_property
    def output_index(self) -> int:
        return list(self.stream.outputs).index(self.output)

    @property
    def depth(self) -> int:
        depths = [
            input_value.depth
            for input_value in self.input_values.values()
            if isinstance(input_value, LazyOutput)
        ]
        if depths:
            depth = max(depths)
        else:
            depth = 0
        return depth + 1

    @staticmethod
    def hashable_key(
        stream: Stream, input_values: Dict[Parameter, Any], output: Parameter
    ) -> Hashable:
        input_key = key_from_kwargs(input_values)
        key = (stream, input_key, output)
        return key

    @staticmethod
    def lookup(
        stream: Stream, input_values: Dict[Parameter, Any], output: Parameter
    ) -> "LazyOutput":
        key = LazyOutput.hashable_key(stream, input_values, output)
        if key not in LazyOutput._cache:
            LazyOutput._cache[key] = LazyOutput(stream, input_values, output)
        return LazyOutput._cache[key]

    @staticmethod
    def from_mapping(stream: Stream, mapping: Dict[Parameter, Any]) -> Output:
        outputs = [LazyOutput.lookup(stream, mapping, output) for output in stream.outputs]
        return Output(*outputs)

    @staticmethod
    def from_instance(stream_instance: StreamInstance) -> Output:
        return LazyOutput.from_mapping(stream_instance.stream, stream_instance.input_values)

    @staticmethod
    def conditional_generator(
        stream: Stream, inputs: Optional[Parameters] = None
    ) -> ConditionalGenerator:
        if inputs is None:
            inputs = list(stream.inputs)

        def fn(*args: Any) -> Output:
            args = wrap_arguments(args)
            mapping = compute_mapping(stream.inputs, args)
            partial_mapping = {inp: mapping[inp] for inp in inputs}
            return LazyOutput.from_mapping(stream, partial_mapping)

        return from_fn(fn)

    @cache
    def substitute(self) -> Constant:
        assert set(self.input_values) == set(self.stream.inputs)
        inputs = list(map(self.input_values.get, self.stream.inputs))
        inputs = [
            inp.value.substitute() if isinstance(inp.value, LazyOutput) else inp for inp in inputs
        ]
        inputs = unwrap_arguments(inputs)

        conditional_generator = self.stream.original_conditional_generator
        assert conditional_generator is not None
        generator = conditional_generator(*inputs)

        output_list = next(generator)
        assert output_list
        outputs = wrap_arguments(output_list[0])
        assert len(self.stream.outputs) == 1
        output = outputs[self.output_index]
        return output

    def __str__(self) -> str:
        prefix = self.output.strip(PARAMETER_PREFIX)
        prefix = prefix[:1]
        index = id(self) % 1000
        return f"@{prefix}{index}"

    __repr__ = __str__
