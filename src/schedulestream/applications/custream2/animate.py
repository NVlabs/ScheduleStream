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
import operator
from itertools import chain, repeat
from typing import Any, Iterator, List, Optional, Tuple

# NVIDIA
from schedulestream.algorithm.temporal import retime_plan
from schedulestream.applications.custream2.command import Command, Commands, Composite, Trajectory
from schedulestream.applications.custream2.state import State
from schedulestream.applications.custream2.utils import INF
from schedulestream.applications.custream2.world import World
from schedulestream.applications.trimesh2d.utils import animate_scene
from schedulestream.common.utils import repeat_first, repeat_last, str_from_sequence
from schedulestream.language.durative import TimedAction


def animate_states(
    states: List[Any],
    time_step: Optional[float] = None,
    state_duration: float = 0.5,
    record: bool = False,
    **kwargs: Any,
) -> List[bytes]:
    """Animate a sequence of world states, yielding frames per state."""
    if not states:
        return []
    world = states[0].world
    if time_step is None:
        time_step = world.time_step
    frequency = 1 / time_step
    steps_per_state = round(state_duration * frequency)
    iterator = (s.set() for s in chain.from_iterable(repeat(t, steps_per_state) for t in states))
    return animate_scene(world.scene, iterator, frequency=frequency, record=record, **kwargs)


def animate_state(state: Any, state_duration: float = 0.5, **kwargs: Any) -> List[bytes]:
    """Animate a single world state, holding it for ``state_duration`` seconds."""
    if state_duration == INF:
        state.show()
        return []
    return animate_states([state], state_duration=state_duration, **kwargs)


def animate_commands(
    state: State,
    commands: Optional[Commands],
    time_step: Optional[float] = None,
    record: bool = False,
    start_duration: float = 0.5,
    end_duration: float = 0.5,
) -> List[bytes]:
    """Animate a Commands sequence, padding the start and end with frozen frames."""
    state.set()
    world = state.world
    if commands is None:
        if not record:
            world.show()
        return []
    if time_step is None:
        time_step = world.time_step
    frequency = 1 / time_step

    iterator = commands.execute(target_fps=frequency)
    iterator = repeat_first(iterator, num=round(start_duration * frequency))
    iterator = repeat_last(iterator, num=round(end_duration * frequency))
    iterator = (world.set() for _ in iterator)
    return animate_scene(world.scene, iterator, frequency=frequency, record=record)


def flatten_commands(commands: Optional[Command | List[Command]]) -> List[Command]:
    """Flatten an optional plan into a sequential list of leaf commands.

    Unwraps Commands containers and single-command Composites; raises when a Composite
    holds truly simultaneous commands.
    """
    if commands is None:
        return []
    if isinstance(commands, Command):
        commands = [commands]
    flattened = []
    for command in Commands.flatten(commands):
        if isinstance(command, Composite):
            leaves = list(Composite.flatten(command.commands))
            if len(leaves) > 1:
                raise NotImplementedError(f"simultaneous commands are unsupported: {command}")
            flattened.extend(leaves)
        else:
            flattened.append(command)
    return flattened


def process_task_commands(commands: List[Command], task_space: bool = False) -> List[Command]:
    """Once a command is task-space, replace every subsequent command with its
    task-space equivalent (after an IK-tracked segment, execution can no longer
    assume the planned joint states). ``task_space`` seeds the initial value,
    converting the whole plan when True."""
    new_commands = []
    for command in Commands.flatten(commands):
        if task_space:
            command = command.get_task_command()
        task_space |= command.is_task_space
        new_commands.append(command)
    return new_commands


def has_task_space(commands: Optional[Command | List[Command]]) -> bool:
    """Return True if any leaf command in ``commands`` is task-space."""
    return any(command.is_task_space for command in flatten_commands(commands))


def solve_joint_commands(
    state: State, commands: Optional[Command | List[Command]], **kwargs: Any
) -> Optional[Commands]:
    """Convert each leaf command to its joint-space equivalent in sequence,
    advancing ``state``'s world so each solve sees the outcome of the previous one.
    Returns None if any command fails to solve."""
    flattened = flatten_commands(commands)
    if not has_task_space(commands):
        return Commands(state.world, flattened)
    state.set()
    solved_commands = []
    for command in flattened:
        solved_command = command.solve_joint_command(**kwargs)
        if solved_command is None:
            return None
        solved_commands.append(solved_command)
    return Commands(state.world, solved_commands)


def extract_sequential_commands(
    state: State, timed_plan: Any, joint_space: bool = False
) -> Optional[Commands]:
    """Extract commands from a timed plan in sequential order, ignoring timing.
    If ``joint_space``, convert the commands to their joint-space equivalents
    via :func:`solve_joint_commands`."""
    world = state.world
    commands: List[Command] = []
    for timed_action in timed_plan:
        command = timed_action.unwrap_arguments()[-1]
        commands.append(command)
    commands = process_task_commands(commands)
    if joint_space:
        return solve_joint_commands(state, commands)
    return Commands(world, commands)


def decompose_timed_action(
    timed_action: TimedAction, rule: str = "middle", task_space: bool = False, **kwargs: Any
) -> Iterator[Tuple[float, Command]]:
    """Yield (absolute_start_time, command) pairs for each leaf command in a timed action.

    ``task_space`` first converts the commands to their task-space equivalents (see
    :func:`process_task_commands`). Either way, each command is then solved back to its
    joint-space equivalent (:meth:`Command.solve_joint_command`, forwarding ``kwargs``) before
    decomposing.
    """
    arguments = timed_action.unwrap_arguments()
    commands: Command = arguments[-1]
    extra_duration = timed_action.duration - commands.duration
    if rule == "start":
        current_t = timed_action.start
    elif rule == "middle":
        current_t = timed_action.start + extra_duration / 2
    elif rule == "end":
        current_t = timed_action.end - commands.duration
    else:
        raise ValueError(rule)
    for command in Commands.flatten([commands]):
        task_space |= command.is_task_space
        if task_space:
            command = command.get_task_command()
            command = command.solve_joint_command(**kwargs)
            if command is None:
                raise RuntimeError(f"failed to solve joint-space command for {command}")
        for item in command.decompose():
            yield current_t, item
            current_t += command.time_step


def combine_timed_commands(
    world: World,
    timed_commands: List[Tuple[float, Command]],
    time_step: Optional[float] = None,
    verbose: bool = True,
) -> Commands:
    """Group commands by their discretized start time into Composite steps."""
    if not timed_commands:
        return Commands(world, [])
    if time_step is None:
        time_step = world.time_step
    timed_commands = sorted(timed_commands, key=operator.itemgetter(0))
    times, cmds = zip(*timed_commands)
    makespan = times[-1]
    num_steps = int(math.ceil(makespan / time_step)) + 1
    discretized: List[List[Command]] = [[] for _ in range(num_steps)]
    for time, command in timed_commands:
        step = min(int(time / time_step), len(discretized) - 1)
        discretized[step].append(command)
    if verbose:
        print(
            f"combine_timed_commands) Commands: {len(timed_commands)} |"
            f" Makespan: {makespan:.3f} | Steps: {num_steps}"
        )
    composite_commands = [Composite(world, bucket) for bucket in discretized if bucket]
    return Commands(world, composite_commands)


def extract_timed_commands(state: State, timed_plan: Any, verbose: bool = False) -> Commands:
    """Extract parallel commands by retiming the plan and grouping by start time."""
    world = state.world
    timed_plan = retime_plan(timed_plan)
    timed_commands: List[Tuple[float, Command]] = []
    for i, timed_action in enumerate(timed_plan):
        entries = list(decompose_timed_action(timed_action))
        timed_commands.extend(entries)
        if verbose:
            print(
                f"{i}/{len(timed_plan)}) Action: {timed_action.action} | Commands: {len(entries)}"
                f" Start: {timed_action.start:.3f} | End: {timed_action.end:.3f}"
            )
    return combine_timed_commands(world, timed_commands)
