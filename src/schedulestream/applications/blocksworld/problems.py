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
import string
from typing import List

# NVIDIA
from schedulestream.language.anonymous import rename_anonymous
from schedulestream.language.connective import Conjunction
from schedulestream.language.predicate import Atom, Predicate
from schedulestream.language.problem import Problem

Clear = Predicate(["?x"], language="Block ?x is clear")
OnTable = Predicate(["?x"], language="Block ?x is on the table")
ArmEmpty = Predicate(["?a"], language="Arm ?a is empty")
Holding = Predicate(["?a", "?x"], language="Arm ?a is holding block ?x")
On = Predicate(["?x", "?y"], language="Block ?x is on block ?y")

ACTIONS = []

rename_anonymous(locals())


def create_arms(num: int = 1) -> List[str]:
    return [f"Arm{i}" for i in range(num)]


def set_empty(arms: List[str]) -> List[Atom]:
    return list(map(ArmEmpty, arms))


def create_blocks(num: int) -> List[str]:
    blocks = list(string.ascii_uppercase)
    assert 0 <= num <= len(blocks)
    blocks = blocks[:num]
    return blocks


def place_blocks(blocks: List[str], clear: bool = True) -> List[Atom]:
    state = []
    for block in blocks:
        state.append(OnTable(block))
        if clear:
            state.append(Clear(block))
    return state


def stack_blocks(
    blocks: List[str],
    clear: bool = True,
) -> List[Atom]:
    state = []
    if not blocks:
        return state
    state.append(OnTable(blocks[0]))
    if clear:
        state.append(Clear(blocks[-1]))
    for block1, block2 in zip(blocks[:-1], blocks[1:]):
        state.append(On(block2, block1))
    return state


def create_line(num_arms: int = 1, num_blocks: int = 2) -> Problem:
    arms = create_arms(num_arms)
    initial = set_empty(arms)

    blocks = create_blocks(num_blocks)
    initial.extend(stack_blocks(blocks))

    goal = Conjunction(*place_blocks(blocks, clear=False))

    return Problem(
        initial=initial,
        goal=goal,
        actions=ACTIONS,
    )


def create_tower(num_arms: int = 1, num_blocks: int = 2) -> Problem:
    arms = create_arms(num_arms)
    initial = set_empty(arms)

    blocks = create_blocks(num_blocks)
    initial.extend(place_blocks(blocks))

    goal = Conjunction(*stack_blocks(blocks, clear=False))

    return Problem(
        initial=initial,
        goal=goal,
        actions=ACTIONS,
    )


def create_reverse(num_arms: int = 1, num_blocks: int = 2) -> Problem:
    arms = create_arms(num_arms)
    initial = set_empty(arms)

    blocks = create_blocks(num_blocks)
    initial.extend(stack_blocks(blocks[::-1]))

    goal = Conjunction(*stack_blocks(blocks, clear=False))

    return Problem(
        initial=initial,
        goal=goal,
        actions=ACTIONS,
    )


def create_repair(num_arms: int = 1, num_blocks: int = 2) -> Problem:
    arms = create_arms(num_arms)
    initial = set_empty(arms)

    blocks = create_blocks(num_blocks)
    initial.extend(stack_blocks(blocks[-1:] + blocks[:-1]))

    goal = Conjunction(*stack_blocks(blocks, clear=False))

    return Problem(
        initial=initial,
        goal=goal,
        actions=ACTIONS,
    )


def create_sussman(num_arms: int = 1, num_blocks: int = 2) -> Problem:
    arms = create_arms(num_arms)
    initial = [
        OnTable("a"),
        On("b", "a"),
        Clear("b"),
    ]
    initial.extend(set_empty(arms))

    blocks = create_blocks(num_blocks)[2:]
    initial.extend(place_blocks(blocks))

    return Problem(
        initial=initial,
        goal=On("a", "b"),
        actions=ACTIONS,
    )


PROBLEMS = [
    create_line,
    create_tower,
    create_reverse,
    create_repair,
    create_sussman,
]
PROBLEMS = {fn.__name__.removeprefix("create_"): fn for fn in PROBLEMS}
