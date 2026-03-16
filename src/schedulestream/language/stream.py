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
from functools import cache, cached_property
from typing import Any, Dict, List, Optional, Tuple

# NVIDIA
from schedulestream.common.graph import is_acyclic, visualize_graph
from schedulestream.common.utils import (
    Key,
    apply_binding,
    compute_mapping,
    not_null,
    str_from_sequence,
)
from schedulestream.language.anonymous import Anonymous
from schedulestream.language.argument import (
    Argument,
    Constant,
    InputParameters,
    Parameter,
    Parameters,
    TypedParameters,
    create_typed_parameters,
    unwrap_arguments,
    wrap_arguments,
)
from schedulestream.language.connective import (
    Conjunction,
    conjunction_difference,
    partition_condition,
)
from schedulestream.language.constraint import Constraint, Cost
from schedulestream.language.function import Function, create_type_formula
from schedulestream.language.generator import (
    ConditionalGenerator,
    Generator,
    Output,
    WrappedGenerator,
    batch_fn_from_list_gen_fn,
)
from schedulestream.language.lazy import LazyOutput
from schedulestream.language.predicate import Predicate


class Context(Key):
    def __init__(
        self,
        outputs: Optional[List[Constant]] = None,
        constraints: Optional[List[Constraint]] = None,
        costs: Optional[List[Cost]] = None,
    ) -> None:
        self.outputs = tuple(outputs)
        self.constraints = frozenset(constraints or [])
        self.costs = frozenset(costs or [])
        super().__init__(self.outputs, self.constraints, self.costs)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(outputs={list(self.outputs)},"
            f" constraints={list(self.constraints)}, costs={list(self.costs)})"
        )

    __repr__ = __str__


class Stream(Anonymous):
    def __init__(
        self,
        conditional_generator: Optional[ConditionalGenerator],
        inputs: Optional[InputParameters] = None,
        input_condition: Optional[Conjunction] = None,
        outputs: Optional[InputParameters] = None,
        output_condition: Optional[Conjunction] = None,
        context_functions: Optional[List[Function]] = None,
        priority: float = 0,
        parent: Optional["Stream"] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.conditional_generator = conditional_generator
        self.original_conditional_generator = conditional_generator
        if input_condition is None:
            input_condition = Conjunction()
        if output_condition is None:
            output_condition = Conjunction()
        self.inputs = create_typed_parameters(inputs)
        self.outputs = create_typed_parameters(outputs)

        input_condition = input_condition & create_type_formula(self.inputs)
        output_condition = output_condition & output_condition.condition
        self.input_condition = conjunction_difference(input_condition, input_condition.condition)
        self.output_condition = conjunction_difference(
            output_condition, input_condition & input_condition.condition
        )
        self.context_functions = list(context_functions or [])
        self.priority = priority
        self.parent = parent
        self.instances = {}
        self.reset()
        if self.conditional_generator is None:
            self.set_lazy()

    def reset(self) -> None:
        self.instances.clear()

    @property
    def has_context(self):
        return bool(self.context_functions)

    def instantiate(self, inputs: List[Any]) -> "StreamInstance":
        inputs = tuple(wrap_arguments(inputs))
        if inputs not in self.instances:
            self.instances[inputs] = StreamInstance(self, list(inputs))
        return self.instances[inputs]

    def __call__(self, *inputs: Any) -> "StreamInstance":
        return self.instantiate(list(inputs))

    @property
    def root(self) -> "Stream":
        if self.parent is None:
            return self
        return self.parent.root

    def clone(
        self,
        conditional_generator: Optional[ConditionalGenerator] = None,
        name: Optional[str] = None,
        parent: bool = True,
    ) -> "Stream":
        return Stream(
            conditional_generator=not_null(conditional_generator, self.conditional_generator),
            inputs=self.inputs,
            input_condition=self.input_condition,
            outputs=self.outputs,
            output_condition=self.output_condition,
            name=not_null(name, self.name),
            priority=self.priority,
            parent=self if parent else None,
        )

    def lazy_conditional_generator(self, **kwargs) -> ConditionalGenerator:
        return LazyOutput.conditional_generator(self, **kwargs)

    def set_lazy(self, **kwargs: Any) -> None:
        self.conditional_generator = self.lazy_conditional_generator(**kwargs)

    def lazy_clone(self, parent: bool = True, **kwargs: Any) -> "Stream":
        name = None
        return self.clone(
            conditional_generator=self.lazy_conditional_generator(**kwargs),
            name=name,
            parent=parent,
        )

    def dump(self):
        print(
            f"{self.__class__.__name__}(\n"
            f" name={self.name},\n"
            f" inputs={self.inputs},\n"
            f" input_condition={self.input_condition},\n"
            f" outputs={self.outputs},\n"
            f" output_condition={self.output_condition})"
        )

    def __str__(self) -> str:
        return f"{self.name}({str_from_sequence(self.inputs)})->({str_from_sequence(self.outputs)})"

    __repr__ = __str__


class TestStream(Stream):
    def __init__(
        self,
        conditional_generator: Optional[ConditionalGenerator],
        inputs: Optional[InputParameters] = None,
        input_condition: Optional[Conjunction] = None,
        output_condition: Optional[Conjunction] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            conditional_generator=conditional_generator,
            inputs=inputs,
            input_condition=input_condition,
            outputs=None,
            output_condition=output_condition,
            **kwargs,
        )


class PredicateStream(Stream):
    def __init__(
        self,
        conditional_generator: Optional[ConditionalGenerator],
        predicate: Predicate,
        inputs: Parameters,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.predicate = predicate
        assert set(inputs) <= set(self.parameters), set(inputs) - set(self.parameters)
        outputs = [parameter for parameter in self.parameters if parameter not in inputs]
        predicate_atom = self.predicate.atom(self.parameters)
        input_condition, output_condition = partition_condition(predicate_atom, inputs)
        if name is None:
            name = self.predicate.name.lower()
        super().__init__(
            conditional_generator=conditional_generator,
            inputs=inputs,
            input_condition=input_condition,
            outputs=outputs,
            output_condition=output_condition,
            name=name,
            **kwargs,
        )

    @property
    def parameters(self) -> Parameters:
        return self.predicate.parameters

    def clone(
        self,
        conditional_generator: Optional[ConditionalGenerator] = None,
        name: Optional[str] = None,
        parent: bool = True,
    ) -> "PredicateStream":
        return PredicateStream(
            conditional_generator=not_null(conditional_generator, self.conditional_generator),
            predicate=self.predicate,
            inputs=list(self.inputs),
            name=not_null(name, self.name),
            priority=self.priority,
            parent=self if parent else None,
        )


class BatchPredicateStream(PredicateStream):
    def __init__(self, *args: Any, batch_size: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, context_functions=None, **kwargs)
        self.batch_size = batch_size

    def next_batch(self, instances: List["StreamInstance"]) -> None:
        inputs_list = [unwrap_arguments(instance.inputs) for instance in instances]
        outs_from_inp = {}
        for inp, out in self.conditional_generator(inputs_list):
            outs_from_inp.setdefault(tuple(wrap_arguments(inp)), []).append(tuple(out))
        for inp, outs in outs_from_inp.items():
            instance = self.instantiate(inp)
            generator = instance.get_generator()
            generator.add(outs)
            instance.exhausted = False


class StreamInstance:
    def __init__(
        self,
        stream: Stream,
        inputs: List[Any],
    ) -> None:
        self.stream = stream
        inputs = list(inputs)
        for constant in inputs:
            assert isinstance(constant, Constant), constant
        assert len(self.stream.inputs) == len(inputs), (self.stream.inputs, inputs)
        self.inputs = inputs
        self.input_condition = self.stream.input_condition.bind(self.input_values)
        self.output_condition = self.stream.output_condition.bind(self.input_values)
        self.instances = {}
        self.generators = {}
        self.reset()

    def reset(self):
        self.generators.clear()
        self.iterations = 0
        self.exhausted = False

    @property
    def called(self) -> bool:
        return bool(self.iterations)

    @property
    def num_outputs(self) -> int:
        return len(self.instances)

    @property
    def name(self) -> str:
        return self.stream.name

    @property
    def conditional_generator(self) -> ConditionalGenerator:
        return self.stream.conditional_generator

    @cached_property
    def input_values(self) -> Dict[Parameter, Constant]:
        return compute_mapping(self.stream.inputs, self.inputs)

    @property
    def outputs(self) -> TypedParameters:
        return self.stream.outputs

    def create_generator(self, context: Optional[Context] = None) -> WrappedGenerator:
        if isinstance(self.stream, BatchPredicateStream):
            generator = WrappedGenerator(generator=iter([]))
            return generator
        inputs = unwrap_arguments(self.inputs)
        if context is None:
            generator = self.conditional_generator(*inputs)
        else:
            generator = self.conditional_generator(*inputs, context=context)
        if not isinstance(generator, WrappedGenerator):
            generator = WrappedGenerator(generator)
        return generator

    def get_generator(self, context: Optional[Context] = None) -> WrappedGenerator:
        if context not in self.generators:
            self.generators[context] = self.create_generator(context)
        return self.generators[context]

    def get_iterations(self, context: Optional[Context] = None) -> int:
        if context not in self.generators:
            return 0
        generator = self.get_generator(context)
        return generator.num

    def instantiate(self, outputs: List[Any]) -> "StreamOutput":
        outputs = tuple(wrap_arguments(outputs))
        if outputs not in self.instances:
            self.instances[outputs] = StreamOutput(self, list(outputs))
        return self.instances[outputs]

    def __call__(self, *outputs: Any) -> "StreamOutput":
        return self.instantiate(list(outputs))

    def bind(self, mapping: Dict[Argument, Argument]) -> "StreamInstance":
        inputs = apply_binding(mapping, self.inputs)
        return self.stream.instantiate(inputs)

    def get_outputs(
        self,
        iteration: Optional[int] = None,
        context: Optional[Context] = None,
        verbose: bool = False,
    ) -> List["StreamOutput"]:
        generator = self.get_generator(context)
        try:
            output_list = generator.get_index(iteration)
        except StopIteration:
            output_list = []
        self.iterations = max(self.iterations, generator.num)
        self.exhausted |= generator.exhausted
        stream_outputs = list(map(self.instantiate, output_list))
        if verbose:
            self.dump()
        return stream_outputs

    def next_outputs(self, verbose: bool = False) -> List["StreamOutput"]:
        return self.get_outputs(verbose=verbose)

    @property
    def root(self) -> "StreamInstance":
        return self.stream.root.instantiate(self.inputs)

    def unwrap(self) -> "StreamInstance":
        return StreamInstance(self.stream, unwrap_arguments(self.inputs))

    def dump(self) -> None:
        print(
            f"Stream: {self} | Exhausted: {self.exhausted} |"
            f" Attempts: {self.iterations} | Outputs:"
            f" {self.num_outputs}"
        )

    def __str__(self) -> str:
        return f"{self.name}({str_from_sequence(self.inputs)})->({str_from_sequence(self.outputs)})"

    __repr__ = __str__


class StreamOutput:
    def __init__(
        self,
        stream_instance: StreamInstance,
        outputs: List[Any],
    ) -> None:
        self.stream_instance = stream_instance
        outputs = list(outputs)
        assert len(self.stream.outputs) == len(outputs), (self.stream.outputs, outputs)
        self.outputs = outputs
        self.output_condition = self.stream_instance.output_condition.bind(self.output_values)

    @property
    def stream(self) -> Stream:
        return self.stream_instance.stream

    @property
    def name(self) -> str:
        return self.stream.name

    @property
    def priority(self) -> float:
        return self.stream.priority

    @property
    def inputs(self) -> Any:
        return self.stream_instance.inputs

    @property
    def input_values(self) -> Dict[Parameter, Constant]:
        return self.stream_instance.input_values

    @property
    def input_condition(self) -> Conjunction:
        return self.stream_instance.input_condition

    @cached_property
    def output_values(self) -> Dict[Parameter, Constant]:
        return compute_mapping(self.stream.outputs, self.outputs)

    @cached_property
    def parameter_values(self) -> Dict[Parameter, Constant]:
        parameter_values = dict(self.input_values)
        parameter_values.update(self.output_values)
        return parameter_values

    @property
    def root(self) -> "StreamOutput":
        return self.stream_instance.root.instantiate(self.outputs)

    def bind(self, mapping: Dict[Argument, Argument]) -> "StreamOutput":
        stream_instance = self.stream_instance.bind(mapping)
        outputs = apply_binding(mapping, self.outputs)
        return StreamOutput(stream_instance, outputs)

    def unwrap(self) -> "StreamOutput":
        return StreamOutput(self.stream_instance, unwrap_arguments(self.outputs))

    def __str__(self) -> str:
        return f"{self.name}({str_from_sequence(self.inputs)})->({str_from_sequence(self.outputs)})"

    __repr__ = __str__


def get_stream_orders(streams: List[Stream]) -> List[Tuple[Stream, Stream]]:
    stream_orders = []
    for stream1, stream2 in itertools.product(streams, repeat=2):
        output_predicates = stream1.output_condition.functions
        input_predicates = stream2.input_condition.functions
        if set(output_predicates) & set(input_predicates):
            stream_orders.append((stream1, stream2))
    return stream_orders


def are_streams_acyclic(streams: List["Stream"]) -> bool:
    stream_orders = get_stream_orders(streams)
    return is_acyclic(stream_orders, streams)


def visualize_streams(streams: List[Stream], **kwargs: Any) -> str:
    stream_orders = get_stream_orders(streams)
    return visualize_graph(stream_orders, streams, **kwargs)
