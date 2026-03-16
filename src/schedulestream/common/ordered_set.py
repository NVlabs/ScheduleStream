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
from collections import OrderedDict
from typing import Any, Iterable, Optional


class OrderedSet(OrderedDict):
    _value = None

    def __init__(self, elements: Optional[Iterable[Any]] = None):
        super().__init__()
        self.update(elements)

    def add(self, element: Any) -> None:
        self[element] = self._value

    def remove(self, element: Any) -> None:
        self.pop(element)

    def discard(self, element: Any) -> None:
        if element in self:
            self.remove(element)

    def update(self, elements: Optional[Iterable[Any]] = None) -> None:
        if elements is not None:
            for element in elements:
                self.add(element)

    def union(self, *others: Iterable["OrderedSet"]) -> "OrderedSet":
        union = OrderedSet(self)
        for other in others:
            union.update(other)
        return union

    def intersection(self, *others: Iterable["OrderedSet"]) -> "OrderedSet":
        union = OrderedSet()
        for element in self:
            if all(element in other for other in others):
                union.add(element)
        return union

    def intersect(self, *others: Iterable["OrderedSet"]) -> bool:
        sets = sorted([self, *others], key=len)
        for element in sets[0]:
            if all(element in s for s in sets):
                return True
        return False

    def difference(self, *others: Iterable["OrderedSet"]) -> "OrderedSet":
        difference = OrderedSet(self)
        for other in others:
            for element in other:
                difference.discard(element)
        return difference

    def __str__(self):
        return f"{{{', '.join(map(str, self))}}}"
