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
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Union


class Anonymous:
    counter = Counter()
    prefix = None

    def __init__(self, name: Optional[str] = None):
        self._name = name
        self.index = self.counter[self.__class__]
        self.counter[self.__class__] += 1

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        prefix = self.prefix
        if prefix is None:
            prefix = self.__class__.__name__
        return f"{prefix}{self.index}"

    def __lt__(self, other: "Anonymous") -> bool:
        return self.name < other.name

    @staticmethod
    def reset() -> None:
        Anonymous.counter.clear()


def rename_anonymous(
    variables: Dict[str, Any], values: Optional[Set[Any]] = None
) -> List[Union["Function", "Action"]]:
    # NVIDIA
    from schedulestream.language.action import Action
    from schedulestream.language.function import Function

    renamed_values = []
    for variable, value in variables.items():
        if (isinstance(value, Function) or isinstance(value, Action)) and (value._name is None):
            if (values is None) or (value in values):
                value._name = variable
                renamed_values.append(value)
    return renamed_values
