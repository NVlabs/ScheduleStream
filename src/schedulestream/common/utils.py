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
import cProfile
import math
import os
import pickle
import pstats
import random
import shutil
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import fields
from functools import cached_property, wraps
from itertools import cycle, islice
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

INF = float("inf")
EPSILON = 1e-6
SEPARATOR = "\n" + 80 * "-" + "\n"


def current_time() -> float:
    return time.perf_counter()


def elapsed_time(start_time: float) -> float:
    return current_time() - start_time


def remaining_time(start_time: float, max_time: float = INF) -> float:
    return max_time - elapsed_time(start_time)


@contextmanager
def timeout_context(timeout: float):
    if timeout == math.inf:
        yield
        return
    timeout = int(math.ceil(timeout))

    def raise_timeout(*args: Any) -> None:
        raise TimeoutError(f"Timed out after {timeout:d} seconds.")

    if timeout <= 0:
        raise_timeout()

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(timeout)
    try:
        yield
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


@contextmanager
def timer(message: str = "Elapsed time: {:.6f} sec"):
    start_time = current_time()
    yield
    print(message.format(elapsed_time(start_time)))


@contextmanager
def profiler(field: Optional[str] = "tottime", num: int = 10):
    if field is None:
        yield
        return
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    pstats.Stats(pr).sort_stats(field).print_stats(num)


def create_seed(seed: Optional[int] = None, max_seed: Optional[int] = 10**3 - 1) -> int:
    if seed is None:
        seed = random.randint(0, max_seed)
    return seed


def random_seeds(
    seed: Optional[int] = None, max_seed: Optional[int] = 10**3 - 1
) -> Iterator[int]:
    seed = create_seed(seed, max_seed=max_seed)
    random_generator = random.Random(seed)
    yield seed
    while True:
        seed = random_generator.randint(0, max_seed)
        yield seed


@contextmanager
def random_context(seed: Optional[int] = None):
    state = random.getstate()
    if seed is not None:
        random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def remove_path(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
    return True


def negate_test(test):
    return lambda *args, **kwargs: not test(*args, **kwargs)


true_test = lambda *args, **kwargs: True


def implies(p1: bool, p2: bool) -> bool:
    return not p1 or p2


def not_null(*args: Any) -> Optional[Any]:
    null = None
    for arg in args:
        if arg is not null:
            return arg
    return null


def is_hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def value_or_id(value: Any) -> Union[Any, int]:
    if is_hashable(value):
        return value
    return id(value)


def key_from_value(value: Any) -> Hashable:
    if is_hashable(value):
        return value
    key = (type(value), id(value))
    return key


def key_from_args(args: Iterable[Any]) -> Tuple[Hashable, ...]:
    return tuple(map(key_from_value, args))


def key_from_kwargs(kwargs: Any) -> FrozenSet[Tuple[str, Hashable]]:
    return frozenset((key, key_from_value(value)) for key, value in kwargs.items())


def key_from_arguments(
    *args: Any, **kwargs: Any
) -> Tuple[Tuple[Hashable, ...], FrozenSet[Tuple[str, Hashable]]]:
    key = (key_from_args(args), key_from_kwargs(kwargs))
    return key


def key_cache(function: Callable) -> Callable:
    cache = {}

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = key_from_arguments(*args, **kwargs)
        if key not in cache:
            cache[key] = function(*args, **kwargs)
        return cache[key]

    return wrapper


def filter_kwargs(filter_value: Optional[Any] = None, **kwargs: Any) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not filter_value}


def join_prefix(sequence: List[Any], prefix: str = " ") -> str:
    return "".join(f"{prefix}{value}" for value in sequence)


def str_from_sequence(sequence: List[Any]) -> str:
    return ", ".join(map(str, sequence))


def safe_zip(*sequences: List[Any]) -> List[Any]:
    if not sequences:
        return []
    sequences = list(map(list, sequences))
    for sequence in sequences:
        assert len(sequence) == len(sequences[0]), (len(sequence), len(sequences[0]))
    return list(zip(*sequences))


def safe_min(iterable: Iterable) -> float:
    return min(iterable, default=0.0)


def safe_max(iterable: Iterable) -> float:
    return max(iterable, default=0.0)


def compute_mapping(sequence1: Iterable[Hashable], sequence2: Iterable[Any]) -> Dict[Hashable, Any]:
    return dict(safe_zip(sequence1, sequence2))


def apply_mapping(mapping: Dict[Hashable, Any], sequence: Iterable[Hashable]) -> List[Any]:
    sequence = list(sequence)
    assert all(element in mapping for element in sequence)
    return list(map(mapping.get, sequence))


def apply_binding(mapping: Dict[Hashable, Any], sequence: Iterable[Hashable]) -> List[Any]:
    return [mapping.get(element, element) for element in sequence]


def remove_duplicates(iterable: Iterable[Any]) -> List[Any]:
    duplicates = set()
    sequence = []
    for element in iterable:
        if element not in duplicates:
            sequence.append(element)
            duplicates.add(element)
    return sequence


filter_duplicates = remove_duplicates


def merge_dicts(*args: dict) -> dict:
    result = dict()
    for d in args:
        result.update(d)
    return result


def item(iterable: Iterable[Any]) -> List[Any]:
    [element] = iterable
    return element


def flatten(iterable_of_iterables: Iterable[Iterable[Any]]) -> Iterator[Any]:
    return (element for iterables in iterable_of_iterables for element in iterables)


def get_pairs(iterable: Iterable[Any]) -> List[Tuple[Any, Any]]:
    sequence = list(iterable)
    return list(zip(sequence[:-1], sequence[1:]))


def get_length(sequence: Optional[List[Any]]) -> float:
    if sequence is None:
        return INF
    return len(sequence)


def partition(test: Callable[[Any], bool], iterable: Iterable[Any]) -> Tuple[List[Any], List[Any]]:
    true_elements = []
    false_elements = []
    for element in iterable:
        if test(element):
            true_elements.append(element)
        else:
            false_elements.append(element)
    return true_elements, false_elements


def randomize(iterable: Iterable[Any]) -> List[Any]:
    sequence = list(iterable)
    random.shuffle(sequence)
    return sequence


def randomly_cycle(iterable: Iterable[Any]) -> Iterator[Any]:
    sequence = list(iterable)
    while True:
        yield random.choice(sequence)


def select(
    sequence: Iterable[Any], reverse: bool = False, shuffle: bool = False, num: Optional[int] = 1
) -> List[Any]:
    sequence = list(sequence)
    if reverse:
        sequence.reverse()
    if shuffle:
        random.shuffle(sequence)
    if (num is None) or (num >= len(sequence)):
        return sequence
    return sequence[:num]


def downsample(iterator: Iterator[Any], frequency: Optional[int] = None) -> Iterator[Any]:
    if frequency is None:
        yield from iterator
        return
    for i, element in enumerate(iterator):
        if i % frequency == 0:
            yield element


def repeat_first(iterator: Iterator[Any], num: int = 0) -> Iterator[Any]:
    try:
        element = next(iterator)
    except StopIteration:
        return
    yield element
    for _ in range(num):
        yield element
    for element in iterator:
        yield element


def repeat_last(iterator: Iterator[Any], num: int = 0) -> Iterator[Any]:
    try:
        element = next(iterator)
    except StopIteration:
        return
    yield element
    for element in iterator:
        yield element
    for _ in range(num):
        yield element


def assert_subset(iterable1, iterable2) -> None:
    difference = set(iterable1) - set(iterable2)
    assert not difference, difference


def take(iterable: Iterable[Any], n: Optional[int] = None) -> Iterator[Any]:
    if n == INF:
        n = None
    return islice(iterable, n)


def batched(iterable: Iterable[Any], batch_size: int) -> Iterator[List[Any]]:
    assert batch_size >= 1, batch_size
    iterator = iter(iterable)
    while True:
        batch = list(take(iterator, batch_size))
        if not batch:
            break
        yield batch


def fill_batch(
    iterable: Iterable[Any], sample: bool = False, batch_size: Optional[int] = None
) -> Iterator[Any]:
    if sample:
        iterable = randomly_cycle(iterable)
    else:
        iterable = cycle(iterable)
    return take(iterable, batch_size)


def batched_filled(iterable: Iterable[Any], batch_size: int, **kwargs: Any) -> Iterator[List[Any]]:
    for batch in batched(iterable, batch_size):
        yield list(fill_batch(batch, batch_size=batch_size, **kwargs))


def get_dataclass_dict(data: Any) -> dict:
    return {field.name: getattr(data, field.name) for field in fields(data)}


def read_pickle(path: str, verbose: bool = True) -> Any:
    start_time = current_time()
    with open(path, "rb") as f:
        data = pickle.load(f)
    if verbose:
        print(f"Loaded {os.path.abspath(path)} in {elapsed_time(start_time):.3f} sec")
    return data


def write_pickle(path: str, data: Any, verbose: bool = True) -> None:
    start_time = current_time()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    if verbose:
        print(f"Saved {os.path.abspath(path)} in {elapsed_time(start_time):.3f} sec")


class Key:
    def __init__(self, *args: Any):
        self.key = tuple(args)

    @cached_property
    def type_key(self) -> Tuple[Hashable]:
        return (type(self),) + tuple(self.key)

    def __eq__(self, other: "Key") -> bool:
        if type(self) != type(other):
            return False
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.type_key)

    def __lt__(self, other: "Key") -> bool:
        return self.key < other.key

    def __str__(self):
        return f"{str_from_sequence(self.key)}"

    __repr__ = __str__


class Context:
    def save(self) -> None:
        pass

    def set(self) -> None:
        raise NotImplementedError()

    def __enter__(self) -> None:
        self.save()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.set()


class Contexts(Context):
    def __init__(self, contexts: List[Context]):
        self.contexts = contexts

    def set(self) -> None:
        for context in self.contexts:
            context.set()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.contexts})"

    __repr__ = __str__


class Silence(Context):
    def __init__(self, enable: bool = True):
        self.enable = enable

    def save(self) -> None:
        if not self.enable:
            return
        self.stdout = sys.stdout
        self.devnull = open(os.devnull, "w")
        sys.stdout = self.devnull

    def set(self) -> None:
        if not self.enable:
            return
        sys.stdout = self.stdout
        self.devnull.close()
