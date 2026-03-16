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
from itertools import chain, count, islice
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sized, Tuple

# NVIDIA
from schedulestream.common.utils import INF, str_from_sequence

Input = Tuple[Any]


class Output(tuple):
    def __new__(cls, *values):
        return tuple.__new__(cls, values)

    def __str__(self) -> str:
        return f"({str_from_sequence(self)})"

    __repr__ = __str__


Generator = Iterator[List[Output]]
ConditionalGenerator = Callable[[Any], Generator]


class WrappedGenerator(Iterator):
    def __init__(self, generator: Iterator[Any], stop: Optional[int] = None):
        if isinstance(generator, Sized) and stop is None:
            stop = len(generator)
        self.generator = islice(generator, stop)
        self.stop = stop
        self.stopped = False
        self.history = []

    @property
    def num(self) -> int:
        return len(self.history)

    @property
    def exhausted(self) -> bool:
        if self.stopped:
            return True
        return (self.stop is not None) and (self.num >= self.stop)

    def update(self, generator: Iterator[Any]) -> None:
        self.generator = chain(self.generator, generator)
        self.stopped = False

    def add(self, item: Any) -> None:
        self.update(generator=iter([item]))

    def next(self) -> Any:
        if self.exhausted:
            raise StopIteration()
        try:
            self.history.append(next(self.generator))
        except StopIteration:
            self.stopped = True
            raise StopIteration()
        return self.history[-1]

    def get_index(self, index: Optional[int] = None) -> Any:
        if index is None:
            index = self.num
        while self.num <= index:
            self.next()
        return self.history[index]

    def __getitem__(self, item: int) -> Any:
        return self.get_index(item)

    __next__ = next

    def __str__(self):
        return f"{self.__class__.__name__}(stop={self.stop})"

    __repr__ = __str__


def from_list_gen_fn(
    list_gen_fn: Callable[[Any], Iterable[List[Output]]], stop: Optional[int] = None
) -> ConditionalGenerator:
    if stop is None:
        return list_gen_fn
    return lambda *args, **kwargs: WrappedGenerator(list_gen_fn(*args, **kwargs), stop=stop)


def from_list_fn(list_fn: Callable[[Any], List[Output]]) -> ConditionalGenerator:
    return from_list_gen_fn(lambda *args, **kwargs: WrappedGenerator([list_fn(*args, **kwargs)]))


def output_from_value(value: Any) -> Optional[Output]:
    if value is None:
        return value
    return Output(value)


def list_from_output(output: Optional[Output]) -> List[Output]:
    if output is None:
        return []
    return [Output(*output)]


def from_fn(fn: Callable[[Any], Optional[Output]]) -> ConditionalGenerator:
    return from_list_fn(lambda *args, **kwargs: list_from_output(fn(*args, **kwargs)))


def from_test(test: Callable[[Any], bool]) -> ConditionalGenerator:
    return from_fn(lambda *args, **kwargs: Output() if test(*args, **kwargs) else None)


def from_gen_fn(gen_fn: Callable[[Any], Iterable[Optional[Output]]]) -> ConditionalGenerator:
    return from_list_gen_fn(lambda *args, **kwargs: map(list_from_output, gen_fn(*args, **kwargs)))


def from_unary_fn(unary_fn: Callable[[Any], Any]) -> ConditionalGenerator:
    return from_fn(lambda *args, **kwargs: output_from_value(unary_fn(*args, **kwargs)))


def from_unary_gen_fn(unary_gen_fn: Callable[[Any], Iterable[Any]]) -> ConditionalGenerator:
    return from_gen_fn(
        lambda *args, **kwargs: map(output_from_value, unary_gen_fn(*args, **kwargs))
    )


def select_indices(sequence: Tuple[Any], indices: Optional[List[int]]) -> Tuple[Any]:
    if indices is None:
        return sequence
    return tuple(sequence[index] for index in indices)


def select_list_gen_fn_indices(
    list_gen_fn: ConditionalGenerator,
    input_indices: Optional[List[int]] = None,
    output_indices: Optional[List[int]] = None,
) -> ConditionalGenerator:
    def new_list_gen_fn(*inputs: Any) -> Generator:
        new_inputs = select_indices(inputs, input_indices)
        for output_list in list_gen_fn(*new_inputs):
            new_output_list = [select_indices(outputs, output_indices) for outputs in output_list]
            yield new_output_list

    return new_list_gen_fn


def batch_list_gen_fn(
    list_gen_fn: ConditionalGenerator, batch_size: Optional[int] = None, verbose: bool = False
):
    batch_size = batch_size or INF

    def new_list_gen_fn(*args: Any, **kwargs: Any) -> Generator:
        list_gen = list_gen_fn(*args, **kwargs)
        batch = []
        for i, outs in enumerate(list_gen):
            empty_outs = not outs
            batch.extend(outs)
            while len(batch) >= batch_size or empty_outs:
                num = min(batch_size, len(batch))
                if verbose:
                    print(f"Iteration: {i}) Pool: {len(batch)} | New: {len(outs)} | Batch: {num}")
                yield batch[:num].copy()
                del batch[:num]
                empty_outs = False
        if batch:
            yield batch.copy()

    return new_list_gen_fn


def constant_fn(value: Any) -> Callable[[Any], Any]:
    return lambda *args, **kwargs: value


true_test = constant_fn(True)
false_test = constant_fn(True)
empty_list_gen_fn = from_fn(constant_fn(None))


def diagonal_product(iterables: List[Iterable[Any]]) -> Iterator[Tuple[Any]]:
    iterators = [iter(iterable) for iterable in iterables]
    sequences = [[] for _ in range(len(iterators))]

    def fn(remaining: int, index: int = 0) -> Iterator[Tuple[Any]]:
        if index == len(sequences):
            yield tuple()
            return
        max_remaining = sum(map(len, sequences[index + 1 :]))
        nums = range(max(0, remaining - max_remaining), min(len(sequences[index]), remaining + 1))
        for n in nums:
            item = sequences[index][n]
            for combo in fn(remaining - n, index + 1):
                yield (item,) + combo

    for diagonal in count():
        for i, iterator in enumerate(iterators):
            try:
                sequences[i].append(next(iterator))
            except StopIteration:
                pass
        exhausted = True
        for combo in fn(diagonal):
            yield combo
            exhausted = False
        if exhausted:
            break


Pair = Tuple[Input, Output]
ConditionalBatch = Callable[[List[Input]], List[Pair]]


def list_gen_fn_from_batch_fn(batch_fn: ConditionalBatch) -> ConditionalGenerator:
    def list_fn(*inp: Any) -> List[Output]:
        inp_list = [inp]
        pair_list = batch_fn(inp_list)
        out_list = [out for _, out in pair_list]
        return out_list

    return from_list_fn(list_fn)


def batch_fn_from_list_gen_fn(list_gen_fn: ConditionalGenerator) -> ConditionalBatch:
    def batch_fn(batch_inputs: List[Input]) -> List[Pair]:
        batch_pairs = []
        for inp in batch_inputs:
            out_list = next(list_gen_fn(*inp), [])
            for out in out_list:
                pair = (inp, out)
                batch_pairs.append(pair)
        return batch_pairs

    return batch_fn


def select_batch_fn_indices(
    batch_fn: ConditionalBatch,
    input_indices: Optional[List[int]] = None,
    output_indices: Optional[List[int]] = None,
) -> ConditionalBatch:
    def new_batch_fn(batch_inputs: List[Input]) -> List[Pair]:
        new_batch_inputs = [select_indices(inputs, input_indices) for inputs in batch_inputs]
        batch_pairs = batch_fn(new_batch_inputs)
        new_batch_pairs = [
            (inputs, select_indices(outputs, output_indices)) for inputs, outputs in batch_pairs
        ]
        return new_batch_pairs

    return new_batch_fn
