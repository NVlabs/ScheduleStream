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
from collections import deque
from collections.abc import Sized
from heapq import heappop, heappush
from typing import Any, Iterable, Iterator, List, Optional, Tuple


class Queue(deque):
    def peek(self) -> Any:
        assert self
        return self[0]

    def push(self, value: Any) -> None:
        self.append(value)

    def pop(self) -> Any:
        assert self
        return self.popleft()


class Stack(Queue):
    def push(self, value: Any) -> None:
        self.appendleft(value)


class HeapElement:
    def __init__(self, key: Any, value: Any) -> None:
        self.key = key
        self.value = value

    def __lt__(self, other: "HeapElement") -> bool:
        return self.key < other.key

    def __iter__(self) -> Iterator[Any]:
        return iter([self.key, self.value])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.key}, {self.value})"


class PriorityQueue(Sized):
    def __init__(self, pairs: Optional[Iterable[Tuple[Any, Any]]] = None) -> None:
        self.queue = []
        if pairs is not None:
            for priority, value in pairs:
                self.push(priority, value)

    def peek(self) -> Any:
        assert self.queue
        element = self.queue[0]
        return element.value

    def push(self, priority: Any, value: Any) -> None:
        element = HeapElement(priority, value)
        heappush(self.queue, element)

    def pop(self) -> Any:
        assert self.queue
        element = heappop(self.queue)
        return element.value

    def __len__(self) -> int:
        return len(self.queue)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={(len(self.queue))})"


class StablePriorityQueue(PriorityQueue):
    def __init__(self, pairs: Optional[Iterable[Tuple[Any, Any]]] = None) -> None:
        self.num_pushes = 0
        super().__init__(pairs)

    def push(self, priority: Any, value: Any) -> None:
        priority = (priority, self.num_pushes)
        super().push(priority, value)
        self.num_pushes += 1
