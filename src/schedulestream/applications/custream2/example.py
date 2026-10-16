#!/usr/bin/env python3
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
import sys

# NVIDIA
from schedulestream.common.utils import get_curobo_major

if get_curobo_major() != 2:
    print("custream2 requires cuRoboV2 (see the README for installation).", file=sys.stderr)
    sys.exit(1)

# Standard Library
import argparse
from typing import Any, Dict, List, Optional, Tuple

# Third Party
import numpy as np
import torch
from curobo.types import JointState

# NVIDIA
from schedulestream.applications.custream2.animate import animate_commands
from schedulestream.applications.custream2.command import Commands
from schedulestream.applications.custream2.costs import COSTS
from schedulestream.applications.custream2.initialize import generate_states, get_stacked_objects
from schedulestream.applications.custream2.motion import parse_motion_planner
from schedulestream.applications.custream2.scene import (
    FRANKA_POSITIONS,
    PRIMITIVES,
    SCENES,
)
from schedulestream.applications.custream2.state import Configuration, State
from schedulestream.applications.custream2.tamp import (
    At,
    AtPose,
    Attached,
    Holding,
    ObjObjRelation,
    solve_tamp,
)
from schedulestream.applications.custream2.utils import set_seed
from schedulestream.applications.custream2.world import World
from schedulestream.applications.trimesh2d.utils import get_video_path, save_frames
from schedulestream.common.utils import SEPARATOR, get_pairs, remove_duplicates
from schedulestream.language.connective import Conjunction
from schedulestream.language.expression import Formula

TASK_SCENES = {"hold": "line", "sort": "platforms"}


def create_world(
    task: str,
    num: int = 2,
    single: bool = False,
    batch: int = 1,
    **kwargs: Any,
) -> World:
    scene = TASK_SCENES.get(task, task)
    if scene not in SCENES:
        raise NotImplementedError(f"Task {task!r} is not yet supported in custream2.")
    kwargs.update(num_objects=num, ik_batch=batch)
    if task in ["hold", "motion", "ycb"]:
        kwargs["num_robots"] = 1 if single else num
    elif task in ["pack", "tight", "blocked", "unpack", "stack"]:
        kwargs["num_robots"] = 1 if single else 2
    elif task == "sort":
        kwargs.update(num_platforms=num, floor_size=0.8)
    world = SCENES[scene](**kwargs)
    world.set_open_positions()
    return world


def create_stacked_objects(world: World, task: str) -> Optional[Dict[str, str]]:
    """Return the initial {object: support} override, or None to infer it from the world poses."""
    if task != "stack":
        return None
    floor = max(world.fixed_objects, key=lambda o: o.area)
    return {name: floor.name for name in world.movable_names}


def create_goal(
    state: State,
    task: Optional[str] = None,
    max_goals: Optional[int] = None,
    reset: bool = True,
) -> Formula:
    world = state.world
    stacked_objects = get_stacked_objects(world, names=world.movable_names)
    supporting_objects = remove_duplicates(stacked_objects.values())
    free_objects = [surface for surface in world.fixed_names if surface not in supporting_objects]

    if task is None:
        goals = []
    elif task in ["floor"]:
        surface = world.fixed_names[0]
        goals = [Attached(obj) == surface for obj in world.movable_names[:-1]]
        if len(world.movable_names) >= 2:
            obj2 = world.movable_names[-1]
            relation = "proximity"
            goals.extend(
                ~ObjObjRelation(relation, obj, AtPose(obj), obj2, AtPose(obj2))
                for obj in world.movable_names[:-1]
            )
    elif task in ["hold", "ycb"]:
        goals = [Holding(arm) <= obj for arm, obj in zip(world.arm_names, world.movable_names)]
    elif task in ["pack", "tight", "unpack"]:
        surface = free_objects[0]
        goals = [Attached(obj) == surface for obj in world.movable_names]
    elif task == "blocked":
        surface = min(world.fixed_objects, key=lambda o: o.area).name
        goals = [Attached(world.movable_names[0]) == surface]
    elif task == "stack":
        goals = [Attached(obj1) == obj2 for obj1, obj2 in get_pairs(world.movable_names)]
        base = world.movable_names[-1]
        surface = min(world.fixed_objects, key=lambda o: o.area).name
        goals.append(Attached(base) == surface)
    elif task == "sort":
        goals = [Attached(obj1) == obj2 for obj1, obj2 in zip(world.movable_names, free_objects)]
    elif task == "stand":
        goals = [Attached(obj) == "stand" for obj in world.movable_names]
    elif task == "relay":
        goals = [Attached(obj) == "mid_platform" for obj in world.movable_names]
    elif task == "motion":
        goals = []
        for arm in world.arms:
            arm_joints = world.get_arm_joints(arm)
            pos = torch.tensor(
                FRANKA_POSITIONS[: len(arm_joints)], dtype=torch.float32, device=world.device
            )
            target = Configuration(
                world, JointState.from_position(pos.unsqueeze(0), joint_names=arm_joints)
            )
            goals.append(At(arm) <= target)
        reset = False
    else:
        raise ValueError(task)

    if (max_goals is not None) and (max_goals < len(goals)):
        goals = goals[:max_goals]

    if reset:
        for arm in world.arms:
            conf = state.arm_configuration(arm)
            goals.append(At(arm) <= conf)
    return Conjunction(*goals)


def plan_example(
    task: str,
    num: int = 2,
    single: bool = False,
    primitive: Optional[str] = None,
    prompt: Optional[str] = None,
    reset: bool = True,
    batch: int = 1,
    motion: str = "joint",
    cost: str = "joint",
    seed: Optional[int] = 0,
    debug: bool = False,
    max_time: float = 60.0,
    **kwargs: Any,
) -> Tuple[State, Optional[Commands]]:
    """Create the task's world, sample an initial state, and plan; commands are None on failure."""
    np.set_printoptions(precision=2, suppress=True, threshold=3)
    set_seed(seed=seed)

    create_world_kwargs = {} if primitive is None else {"primitive": primitive}
    world = create_world(task, num=num, single=single, batch=batch, **create_world_kwargs)
    world.warmup(motion_planner=(motion.split(":", 1)[0] == "curobo"))
    state = next(generate_states(world, stacked_objects=create_stacked_objects(world, task)), None)
    assert state is not None, "Failed to generate a valid state"
    state.set()

    goal = create_goal(state, task=task if prompt is None else None, reset=reset)
    commands = solve_tamp(
        state,
        goal,
        prompt=prompt,
        motion=motion,
        cost=cost,
        debug=debug,
        max_time=max_time,
        **kwargs,
    )
    return state, commands


def example(
    visualize: bool = False,
    record: bool = False,
    **kwargs: Any,
) -> Optional[List[bytes]]:
    """Plan a task (see :func:`plan_example`) and optionally animate it; None if planning fails."""
    state, commands = plan_example(**kwargs)
    world = state.world
    if commands is None:
        if visualize:
            world.show()
        return None

    if not visualize and not record:
        return []

    world.reset()
    state.set()

    return animate_commands(state, commands, record=record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        default="hold",
        type=str,
        choices=[
            "hold",
            "motion",
            "pack",
            "tight",
            "blocked",
            "unpack",
            "stack",
            "sort",
            "stand",
            "relay",
            "ycb",
            "floor",
        ],
        help="The task name.",
    )
    parser.add_argument("-n", "--num", default=2, type=int, help="The number of objects.")
    parser.add_argument(
        "--primitive",
        type=str,
        default=None,
        choices=PRIMITIVES,
        help="Movable object primitive (e.g. cuboid, ycb).",
    )
    parser.add_argument("--scenes", default=1, type=int, help="The number of scenes to run.")
    parser.add_argument("--single", action="store_true", help="Use a single robot when applicable.")
    parser.add_argument("--no-reset", action="store_true", help="Disables adding a reset goal.")
    parser.add_argument("--prompt", type=str, default=None, help="The task prompt.")
    parser.add_argument(
        "--interpret",
        action="store_true",
        help=(
            "Use interpret_llm (goal conjunction) instead of solve_llm (partial plan) for --prompt."
        ),
    )
    parser.add_argument("-b", "--batch", type=int, default=128, help="The stream batch size.")
    parser.add_argument("-c", "--cfree", action="store_true", help="Enables collision free.")
    parser.add_argument(
        "--defer",
        action="store_true",
        help="Defer motion planning to execution time (DeferredTrajectory).",
    )
    parser.add_argument(
        "-m",
        "--motion",
        type=parse_motion_planner,
        default="joint",
        help=(
            "Motion planner: curobo (plan_cspace), joint (cspace PRM), task (SE(3) tool-pose"
            " PRM), or linear (straight-line interpolation, no collision checking)."
            " joint accepts a distance-cost suffix, e.g. joint:sphere or"
            " joint:task-position."
        ),
    )
    parser.add_argument(
        "--cost",
        type=str,
        default="joint",
        choices=COSTS,
        help="Distance function for the action costs.",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Run with debug streams")
    parser.add_argument(
        "-s", "--seed", nargs="?", const=None, default=0, type=int, help="The random seed."
    )
    parser.add_argument("-p", "--profile", action="store_true", help="Enables profiling")
    parser.add_argument("-v", "--video", type=str, nargs="?", const="", default=None)
    parser.add_argument("--headless", action="store_true", help="Disables visualization.")
    parser.add_argument(
        "--max-time", type=float, default=60.0, help="Solver time limit in seconds."
    )
    args = parser.parse_args()
    print("Args:", args)

    seed = args.seed
    video_path = get_video_path(args.video, name=f"{args.task}{args.num}")
    all_frames = []
    for scene in range(args.scenes):
        print(SEPARATOR)
        frames = example(
            task=args.task,
            num=args.num,
            single=args.single,
            primitive=args.primitive,
            prompt=args.prompt,
            interpret=args.interpret,
            reset=not args.no_reset,
            batch=args.batch,
            collisions=not args.cfree,
            motion=args.motion,
            cost=args.cost,
            defer=args.defer,
            debug=args.debug,
            seed=seed,
            profile=args.profile,
            record=video_path is not None,
            visualize=not args.headless,
            verbose=True,
            max_time=args.max_time,
        )
        if frames is not None:
            all_frames.extend(frames)
        seed = int(np.random.RandomState(seed).randint(10**3))
    if video_path is not None:
        save_frames(all_frames, video_path=video_path)


if __name__ == "__main__":
    main()
