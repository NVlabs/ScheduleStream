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
# Standard Library
import argparse
from functools import partial
from typing import Any, List, Optional

# Third Party
import numpy as np

# NVIDIA
from schedulestream.algorithm.solver import solve
from schedulestream.applications.custream.animate import animate_commands, extract_timed_commands
from schedulestream.applications.custream.collision import (
    commands_conf_collision,
    commands_placement_collision,
)
from schedulestream.applications.custream.command import Commands
from schedulestream.applications.custream.initialize import generate_states
from schedulestream.applications.custream.placement import Placement as PlacementPose
from schedulestream.applications.custream.placement import get_ordered_stacks, get_stacked_objects
from schedulestream.applications.custream.postprocess import postprocess_plan
from schedulestream.applications.custream.scene import create_franka_line_world, create_franka_world
from schedulestream.applications.custream.state import State
from schedulestream.applications.custream.streams import (
    grasp_stream,
    motion_stream,
    pick_stream,
    place_stream,
    placement_stream,
)
from schedulestream.applications.custream.utils import set_seed
from schedulestream.applications.custream.world import World
from schedulestream.applications.trimesh2d.utils import get_video_path, save_frames
from schedulestream.common.utils import SEPARATOR, get_pairs, profiler, remove_duplicates
from schedulestream.language.action import Action
from schedulestream.language.anonymous import rename_anonymous
from schedulestream.language.argument import OUTPUT, unwrap_arguments
from schedulestream.language.connective import Conjunction
from schedulestream.language.durative import DurativeAction
from schedulestream.language.expression import Formula
from schedulestream.language.function import Evaluation, Function, NumericFunction
from schedulestream.language.generator import (
    batch_list_gen_fn,
    from_unary_fn,
    from_unary_gen_fn,
    list_gen_fn_from_batch_fn,
)
from schedulestream.language.predicate import Predicate, Type
from schedulestream.language.problem import Problem
from schedulestream.language.stream import PredicateStream, Stream

Arm = Type()
Conf = Predicate(["?arm", "?conf"], condition=Arm("?arm"))
Traj = Predicate(["?arm", "?traj"], condition=Arm("?arm"))
At = Function(["?arm"], condition=Conf("?arm", OUTPUT))
Holding = Function(["?arm"], condition=Arm("?arm"))
Moved = Predicate(["?arm"], condition=Arm("?arm"))

Object = Type()
AtPose = Function(["?obj"], condition=Object("?obj"))
Attached = Function(["?obj"], condition=Object("?obj"))
Supporting = Predicate(["?obj"], condition=Object("?obj"))

Graspable = Predicate(["?arm", "?obj"], condition=Object("?arm") & Object("?obj"))
Placeable = Predicate(["?obj", "?obj2"], condition=Object("?obj") & Object("?obj2"))

Placement = Predicate(["?obj", "?placement"], condition=Object("?obj"))
Grasp = Predicate(["?arm", "?obj", "?grasp"], condition=Graspable("?arm", "?obj"))
Supported = Predicate(
    ["?obj1", "?placement1", "?obj2", "?placement2"],
    condition=Placeable("?obj1", "?obj2")
    & Placement("?obj1", "?placement1")
    & Placement("?obj2", "?placement2"),
)
Motion = Predicate(
    ["?arm", "?conf1", "?conf2", "?traj"],
    condition=Conf("?arm", "?conf1") & Conf("?arm", "?conf2") & Traj("?arm", "?traj"),
)
Pick = Predicate(
    ["?arm", "?obj", "?grasp", "?placement", "?conf1", "?conf2", "?traj"],
    condition=Grasp("?arm", "?obj", "?grasp")
    & Placement("?obj", "?placement")
    & Conf("?arm", "?conf1")
    & Conf("?arm", "?conf2")
    & Traj("?arm", "?traj"),
)
Place = Predicate(
    ["?arm", "?obj", "?grasp", "?placement", "?conf1", "?conf2", "?traj"],
    condition=Grasp("?arm", "?obj", "?grasp")
    & Placement("?obj", "?placement")
    & Conf("?arm", "?conf1")
    & Conf("?arm", "?conf2")
    & Traj("?arm", "?traj"),
)

Duration = NumericFunction(
    ["?arm", "?traj"],
    condition=Traj("?arm", "?traj"),
    definition=None,
    lazy=True,
)
ArmArmCollision = Predicate(
    ["?arm1", "?traj1", "?arm2", "?conf2"],
    definition=None,
    lazy=True,
)
ArmObjCollision = Predicate(
    ["?arm1", "?traj1", "?obj", "?pose"],
    definition=None,
    lazy=True,
)

rename_anonymous(locals())


def create_actions(world: World, collisions: bool = True) -> List[Action]:
    Duration.set_definition(lambda *pair: pair[1].duration)
    ArmArmCollision.set_definition(partial(commands_conf_collision, world, conf_collisions=True))
    ArmObjCollision.set_definition(partial(commands_placement_collision, world))

    arm_colliders = world.arm_names if collisions else []
    arm_conditions = [~ArmArmCollision("?arm", "?traj", arm, At(arm)) for arm in arm_colliders]
    object_colliders = world.movable_names if collisions else []
    obj_conditions = [
        ~ArmObjCollision("?arm", "?traj", obj, AtPose(obj)) for obj in object_colliders
    ]
    collision_condition = Conjunction(*arm_conditions, *obj_conditions)

    move = DurativeAction(
        parameters="?arm ?conf1 ?conf2 ?traj",
        start_condition=(At("?arm") == "?conf1"),
        start_effect=At("?arm") <= "?traj",
        over_condition=Motion("?arm", "?conf1", "?conf2", "?traj") & collision_condition,
        end_effect=(At("?arm") <= "?conf2") & Moved("?arm"),
        min_duration=Duration("?arm", "?traj"),
    )
    pick = DurativeAction(
        parameters="?arm ?obj ?grasp ?placement ?conf1 ?conf2 ?traj",
        start_condition=(At("?arm") == "?conf1")
        & (AtPose("?obj") == "?placement")
        & ~Supporting("?obj"),
        start_effect=(At("?arm") <= "?traj") & (AtPose("?obj") <= None),
        over_condition=Pick("?arm", "?obj", "?grasp", "?placement", "?conf1", "?conf2", "?traj")
        & (Holding("?arm") == None)
        & collision_condition,
        end_effect=(At("?arm") <= "?conf2")
        & (Holding("?arm") <= "?obj")
        & (AtPose("?obj") <= "?grasp")
        & (Attached("?obj") <= "?arm")
        & ~Moved("?arm"),
        min_duration=Duration("?arm", "?traj"),
        cost=1e-6,
    )
    place = DurativeAction(
        parameters="?arm ?obj ?grasp ?placement ?o2 ?p2 ?conf1 ?conf2 ?traj",
        start_condition=(At("?arm") == "?conf1")
        & (AtPose("?obj") == "?grasp")
        & (AtPose("?o2") == "?p2"),
        start_effect=(At("?arm") <= "?traj") & (AtPose("?obj") <= None) & Supporting("?o2"),
        over_condition=Place("?arm", "?obj", "?grasp", "?placement", "?conf1", "?conf2", "?traj")
        & Supported("?obj", "?placement", "?o2", "?p2")
        & collision_condition,
        end_effect=(At("?arm") <= "?conf2")
        & (Holding("?arm") <= None)
        & (AtPose("?obj") <= "?placement")
        & (Attached("?obj") <= "?o2")
        & ~Moved("?arm"),
        min_duration=Duration("?arm", "?traj"),
        cost=1e-6,
    )
    rename_anonymous(locals())
    return [move, pick, place]


def create_streams(
    world: World,
    collisions: bool = True,
) -> List[Stream]:
    batch_size = world.ik_batch
    pick_cond_gen = partial(pick_stream, world, collisions=collisions)
    place_cond_gen = partial(place_stream, world, collisions=collisions)
    if batch_size == 1:
        batch_size = None
        pick_cond_gen = list_gen_fn_from_batch_fn(pick_cond_gen)
        place_cond_gen = list_gen_fn_from_batch_fn(place_cond_gen)

    streams = [
        Grasp.stream(
            conditional_generator=batch_list_gen_fn(
                from_unary_gen_fn(partial(grasp_stream, world)),
                batch_size=world.ik_batch,
            ),
            inputs=["?obj", "?arm"],
        ),
        Supported.stream(
            conditional_generator=batch_list_gen_fn(
                from_unary_gen_fn(
                    partial(placement_stream, world, object_proximity=0.0 if collisions else None)
                ),
                batch_size=world.ik_batch,
            ),
            inputs=["?obj1", "?obj2", "?placement2"],
        ),
        Motion.stream(
            conditional_generator=from_unary_fn(
                partial(motion_stream, world, collisions=collisions)
            ),
            inputs=["?arm", "?conf1", "?conf2"],
            context_functions=[ArmObjCollision],
        ),
        Pick.stream(
            conditional_generator=pick_cond_gen,
            inputs=["?obj", "?grasp", "?placement", "?arm"],
            batch_size=batch_size,
        ),
        Place.stream(
            conditional_generator=place_cond_gen,
            inputs=["?obj", "?grasp", "?placement", "?arm"],
            batch_size=batch_size,
        ),
    ]
    return streams


def create_initial(state: State) -> List[Evaluation]:
    world = state.world
    initial = []

    for arm in world.arms:
        conf = state.arm_configuration(arm)
        initial.extend(
            [
                At(arm) <= conf,
                Holding(arm) <= None,
            ]
        )

    placements = {}
    ordered_stacks = get_ordered_stacks(world, names=world.object_names)
    for obj, parent in ordered_stacks.items():
        parent_placement = placements.get(parent, None)
        placement = PlacementPose(world, obj, placement=parent_placement)
        placements[obj] = placement
        initial.extend(
            [
                Placement(obj, placement),
                AtPose(obj) <= placement,
                Attached(obj) <= None,
            ]
        )

    for obj in world.movable_objects:
        initial.extend(Graspable(arm, obj.name) for arm in world.arms)
        for support in world.objects:
            if support.stackable:
                initial.append(Placeable(obj.name, support.name))
    return initial


def create_goal(
    state: State, task: str, max_goals: Optional[int] = None, reset: bool = True, **kwargs: Any
) -> Formula:
    world = state.world
    stacked_objects = get_stacked_objects(world, names=world.movable_names)
    supporting_objects = remove_duplicates(stacked_objects.values())
    if task == "hold":
        goals = [Holding(arm) <= obj for arm, obj in zip(world.arm_names, world.movable_names)]
    elif task == "pack":
        surfaces = [surface for surface in world.fixed_names if surface not in supporting_objects]
        surface = surfaces[0]
        goals = [Attached(obj) == surface for obj in world.movable_names]
    elif task == "stack":
        goals = [Attached(obj1) == obj2 for obj1, obj2 in get_pairs(world.movable_names)]
        base = world.movable_names[-1]
        surface = min(world.fixed_objects, key=lambda o: o.area).name
        goals.append(Attached(base) == surface)
    else:
        raise ValueError(task)

    if (max_goals is not None) and (max_goals > len(goals)):
        goals = goals[:max_goals]

    if reset:
        for arm in world.arms:
            conf = state.arm_configuration(arm)
            goals.append(At(arm) <= conf)
    return Conjunction(*goals)


def initial_from_goal(world: World, goal: Formula) -> List[Evaluation]:
    initial = []
    if goal is None:
        return initial
    for formula in goal.clause:
        initial.extend(formula.condition.clause)
        if isinstance(formula, Evaluation):
            if formula.function == Attached:
                (top_obj,) = unwrap_arguments(formula.inputs)
                bottom_obj = formula.output.unwrap()
                if bottom_obj in world.object_names:
                    initial.append(Placeable(top_obj, bottom_obj))
    return initial


def create_world(
    task: str,
    num: int = 2,
    **kwargs: Any,
) -> World:
    if task == "hold":
        world = create_franka_line_world(
            num_robots=num,
            num_objects=num,
            **kwargs,
        )
    elif task == "pack":
        world = create_franka_world(
            num_robots=2,
            num_objects=num,
            floor_width=0.6,
            floor_depth=1.0,
            platform_width=0.4,
            **kwargs,
        )
        surface = max(world.fixed_objects, key=lambda o: o.area)
        for obj in world.movable_objects:
            obj.stack(surface)
    elif task == "stack":
        world = create_franka_world(
            num_robots=2,
            num_objects=num,
            floor_width=0.6,
            floor_depth=1.0,
            platform_width=0.1,
            **kwargs,
        )
        surface = max(world.fixed_objects, key=lambda o: o.area)
        for obj in world.movable_objects:
            obj.stack(surface)
    else:
        raise ValueError(task)

    closed_conf, opened_conf = world.get_gripper_limits()
    world.set_joint_positions(world.gripper_joints, opened_conf)
    world.set_retract_conf()
    return world


def solve_tamp(
    state: State,
    goal: Formula,
    hierarchical: bool = True,
    sequential: Optional[bool] = None,
    collisions: bool = True,
    debug: bool = False,
    profile: bool = False,
    **kwargs,
) -> Optional[Commands]:
    world = state.world
    if sequential is None:
        sequential = len(world.arms) <= 1
    initial = create_initial(state)
    initial.extend(initial_from_goal(world, goal))
    problem = Problem(
        initial=initial,
        goal=goal,
        actions=create_actions(world, collisions=collisions),
        streams=create_streams(world, collisions=collisions),
    )
    if debug:
        problem = problem.lazy_clone(parent=False)
    motion_streams = [
        stream
        for stream in problem.streams
        if isinstance(stream, PredicateStream) and stream.predicate == Motion
    ]
    if hierarchical:
        hierarchical_functions = [At]
        if problem.has_function("ArmArmCollision"):
            hierarchical_functions.append(problem.get_function("ArmArmCollision"))
        problem = problem.remove_conditions(hierarchical_functions)
        problem.streams = [stream for stream in problem.streams if stream not in motion_streams]

    print(SEPARATOR)

    with profiler(field="tottime" if profile else None, num=25):
        solutions = solve(
            problem,
            sequential=sequential,
            lazy=False,
            heuristic_fn="hmax",
            successor_fn="offline",
            weight=1,
            satisfy_time=1.0,
            optimize_time=1.0,
            num_reschedule=1,
            **kwargs,
        )
    print(SEPARATOR)
    if not solutions:
        return None

    solution = solutions[0]
    solution.dump()
    problem = problem.root
    problem.initial = problem.initial.new_state(evaluations=solution.stream_atoms)
    problem.streams = motion_streams

    with profiler(field="tottime" if profile else None, num=25):
        solutions = solve(
            problem,
            sequential=True,
            reschedule=True,
            lazy=True,
            heuristic_fn="relaxed",
            successor_fn="offline",
            weight=5,
            num_reschedule=1,
            **kwargs,
        )
    if not solutions:
        return None

    print(SEPARATOR)
    solution = solutions[0]
    solution.dump()
    if debug:
        return Commands(world, commands=[])

    if sequential:
        return extract_timed_commands(state, solution.plan)
    with state:
        return postprocess_plan(state, solution.plan, collisions=collisions)


def example(
    task: str = "hold",
    num: int = 2,
    batch: int = 1,
    seed: Optional[int] = 0,
    visualize: bool = False,
    video: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Commands]:
    np.set_printoptions(precision=2, suppress=True, threshold=3)
    set_seed(seed=seed)

    world = create_world(task, num=num)
    world.initialize(batch)
    state = next(generate_states(world))
    assert state is not None

    goal = create_goal(state, task)
    commands = solve_tamp(state, goal, **kwargs)
    if commands is None:
        return commands

    video_path = get_video_path(video, name=f"{task}{num}")
    if not visualize and (video_path is None):
        return commands

    frames = animate_commands(state, commands, frequency=1, record=(video_path is not None))
    save_frames(frames, video_path=video_path)

    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        default="hold",
        type=str,
        choices=["hold", "pack", "stack"],
        help="The task name.",
    )
    parser.add_argument(
        "-n", "--num", default=2, type=int, help="The number of robots and objects."
    )
    parser.add_argument("-b", "--batch", type=int, default=1, help="The stream batch size.")
    parser.add_argument("-c", "--cfree", action="store_true", help="Enables collision free.")
    parser.add_argument("-d", "--debug", action="store_true", help="Run with debug streams")
    parser.add_argument(
        "-s", "--seed", nargs="?", const=None, default=0, type=int, help="The random seed."
    )
    parser.add_argument("-p", "--profile", action="store_true", help="Enables profiling")
    parser.add_argument("-v", "--video", type=str, nargs="?", const="", default=None)
    args = parser.parse_args()
    print("Args:", args)

    example(
        task=args.task,
        num=args.num,
        batch=args.batch,
        collisions=not args.cfree,
        debug=args.debug,
        seed=args.seed,
        profile=args.profile,
        video=args.video,
        visualize=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
