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
import math
from operator import itemgetter
from typing import Any, Iterator, List, Optional, Tuple

# NVIDIA
from schedulestream.algorithm.temporal import TimedPlan, retime_plan
from schedulestream.applications.custream.command import Command, Commands, Composite
from schedulestream.applications.custream.state import State
from schedulestream.applications.custream.world import World
from schedulestream.applications.trimesh2d.utils import animate_scene
from schedulestream.common.utils import downsample, repeat_first, repeat_last, str_from_sequence
from schedulestream.language.durative import TimedAction


def str_from_action(timed_action: TimedAction) -> str:
    arguments = [arg for arg in timed_action.unwrap_arguments() if isinstance(arg, str)]
    return f"{timed_action.name}({str_from_sequence(arguments)})"


def extract_sequential_commands(
    state: State, timed_plan: Optional[TimedPlan]
) -> Optional[Commands]:
    if timed_plan is None:
        return None
    world = state.world
    commands = []
    for timed_action in timed_plan:
        for command in Commands.flatten([timed_action.unwrap_arguments()[-1]]):
            command.metadata["action"] = str_from_action(timed_action)
            commands.append(command)
    return Commands(world, commands)


def combine_timed_commands(
    world: World, timed_commands: List[Tuple[float, Command]], time_step: Optional[float] = None
) -> Commands:
    if not timed_commands:
        return Commands(world, [])
    if time_step is None:
        time_step = world.time_step
    timed_commands.sort(key=itemgetter(0))
    times, commands = zip(*timed_commands)
    makespan = times[-1]
    num_steps = int(math.ceil(makespan / time_step))
    discretized_commands = [[] for _ in range(num_steps)]
    print(f"Commands: {len(timed_commands)} | Makespan: {makespan:.3f} | Steps: {num_steps}")
    for time, command in timed_commands:
        step = int(time / time_step)
        while step >= len(discretized_commands):
            discretized_commands.append([])
        discretized_commands[step].append(command)
    composite_commands = [Composite(world, commands) for commands in discretized_commands]
    return Commands(world, composite_commands)


def decompose_timed_action(
    timed_action: TimedAction, rule: str = "middle"
) -> Iterator[Tuple[float, Command]]:
    arguments = timed_action.unwrap_arguments()
    commands = arguments[-1]
    extra_duration = timed_action.duration - commands.duration
    if rule == "start":
        current_t = timed_action.start
    elif rule == "middle":
        current_t = timed_action.start + extra_duration / 2
    elif rule == "end":
        current_t = timed_action.end - extra_duration
    else:
        raise ValueError(rule)
    for command in commands.decompose():
        command.metadata["action"] = str_from_action(timed_action)
        yield current_t, command
        current_t += command.duration


def extract_timed_commands(
    state: State, timed_plan: Optional[TimedPlan], verbose: bool = True
) -> Optional[Commands]:
    if timed_plan is None:
        return None
    world = state.world
    timed_plan = retime_plan(timed_plan)

    timed_commands = []
    for i, timed_action in enumerate(timed_plan):
        timed_commands.extend(decompose_timed_action(timed_action))
        if verbose:
            print(
                f"{i}/{len(timed_plan)}) Action: {timed_action.action} | Start Time:"
                f" {timed_action.start:.3f} | End Time: {timed_action.end:.3f}"
            )
    return combine_timed_commands(world, timed_commands)


def animate_commands(
    state: State,
    commands: Optional[Commands],
    frequency: Optional[int] = None,
    start_steps: Optional[int] = 10,
    end_steps: Optional[int] = 10,
    **kwargs: Any,
) -> List[bytes]:
    world = state.world
    state.set()
    if commands is None:
        return []
    iterator = downsample(commands.execute(), frequency=frequency)
    if start_steps is not None:
        iterator = repeat_first(iterator, num=start_steps)
    if end_steps is not None:
        iterator = repeat_last(iterator, num=end_steps)
    return animate_scene(world.scene, iterator, **kwargs)
