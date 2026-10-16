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
import sys

# NVIDIA
from schedulestream.common.utils import get_curobo_major

if get_curobo_major() != 1:
    print("custream requires cuRoboV1 (see the README for installation).", file=sys.stderr)
    sys.exit(1)

# Standard Library
import argparse
import math
from functools import partial
from typing import Any, List, Optional

# Third Party
import numpy as np

# NVIDIA
from schedulestream.algorithm.llm import interpret_llm, solve_llm
from schedulestream.algorithm.solver import solve
from schedulestream.applications.custream.animate import (
    animate_commands,
    extract_sequential_commands,
    extract_timed_commands,
)
from schedulestream.applications.custream.collision import (
    commands_conf_collision,
    commands_placement_collision,
)
from schedulestream.applications.custream.command import Commands
from schedulestream.applications.custream.initialize import generate_states
from schedulestream.applications.custream.placement import Placement as PlacementPose
from schedulestream.applications.custream.placement import get_ordered_stacks, get_stacked_objects
from schedulestream.applications.custream.postprocess import postprocess_plan
from schedulestream.applications.custream.scene import (
    create_franka_line_world,
    create_franka_platforms_world,
    create_franka_relay_world,
    create_franka_shelf_world,
    create_franka_world,
)
from schedulestream.applications.custream.state import State
from schedulestream.applications.custream.streams import (
    distance_function,
    grasp_stream,
    motion_stream,
    pick_stream,
    place_stream,
    placement_stream,
)
from schedulestream.applications.custream.utils import set_seed
from schedulestream.applications.custream.world import World
from schedulestream.applications.trimesh2d.utils import get_video_path, save_frames
from schedulestream.common.utils import (
    INF,
    SEPARATOR,
    get_pairs,
    profiler,
    remove_duplicates,
)
from schedulestream.language.action import Action, AppliedPredicate
from schedulestream.language.anonymous import rename_anonymous
from schedulestream.language.argument import OUTPUT, unwrap_arguments
from schedulestream.language.connective import Conjunction
from schedulestream.language.durative import DurativeAction
from schedulestream.language.expression import Formula
from schedulestream.language.function import Evaluation, Function, NumericFunction
from schedulestream.language.generator import (
    batch_fn_from_fn,
    batch_list_gen_fn,
    false_test,
    from_unary_fn,
    from_unary_gen_fn,
    list_gen_fn_from_batch_fn,
    scale_fn,
    truncate_batch_fn,
)
from schedulestream.language.predicate import Predicate, Type
from schedulestream.language.problem import Problem
from schedulestream.language.stream import PredicateStream, Stream

Arm = Type()
Conf = Predicate(["?arm", "?conf"], condition=Arm("?arm"))
Traj = Predicate(["?arm", "?traj"], condition=Arm("?arm"))
At = Function(["?arm"], condition=Conf("?arm", OUTPUT))
Holding = Function(["?arm"], condition=Arm("?arm"), language=f"Arm ?arm is holding object {OUTPUT}")
Moved = Predicate(["?arm"], condition=Arm("?arm"))

Object = Type()
AtPose = Function(["?obj"], condition=Object("?obj"))
Attached = Function(
    ["?obj"], condition=Object("?obj"), language=f"Object ?obj is attached to {OUTPUT}"
)
Supporting = Predicate(["?obj"], condition=Object("?obj"))

Graspable = Predicate(["?arm", "?obj"], condition=Arm("?arm") & Object("?obj"))
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
)
Distance = NumericFunction(
    ["?arm", "?conf1", "?conf2"],
    condition=Conf("?arm", "?conf1") & Conf("?arm", "?conf2"),
    definition=None,
    batch_size=INF,
)
ArmArmCollision = Predicate(
    ["?arm1", "?traj1", "?arm2", "?conf2"],
    definition=None,
)
ArmObjCollision = Predicate(
    ["?arm1", "?traj1", "?obj", "?pose"],
    definition=None,
)
ObjObjCollision = Predicate(
    ["?obj", "?placement", "?obj2", "?pose2"],
    definition=false_test,
)

rename_anonymous(locals())


def create_actions(world: World, collisions: bool = True) -> List[Action]:
    Duration.set_definition(lambda *pair: pair[1].duration)
    Distance.set_definition(batch_fn_from_fn(scale_fn(partial(distance_function, world), scale=10)))
    ArmArmCollision.set_definition(partial(commands_conf_collision, world, conf_collisions=True))
    ArmObjCollision.set_definition(partial(commands_placement_collision, world))

    arm_colliders = world.arm_names if collisions else []
    if len(arm_colliders) <= 1:
        arm_colliders.clear()
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
        cost=Distance("?arm", At("?arm"), "?conf1"),
        language=f"Arm ?arm picks object ?obj",
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
        cost=Distance("?arm", At("?arm"), "?conf1"),
        language=f"Arm ?arm places object ?obj on object ?o2",
    )
    rename_anonymous(locals())
    return [move, pick, place]


def create_streams(
    world: World,
    collisions: bool = True,
    greedy_outputs: int = 10,
) -> List[Stream]:
    batch_size = world.ik_batch
    greedy_outputs = min(int(math.sqrt(batch_size)), greedy_outputs)
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
                batch_size=world.ik_batch // greedy_outputs,
                max_iterations=1,
            ),
            inputs=["?obj", "?arm"],
        ),
        Supported.stream(
            conditional_generator=batch_list_gen_fn(
                from_unary_gen_fn(
                    partial(placement_stream, world, object_proximity=0.0 if collisions else None)
                ),
                batch_size=world.ik_batch // greedy_outputs,
                max_iterations=1,
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
            greedy_outputs=greedy_outputs,
        ),
        Place.stream(
            conditional_generator=place_cond_gen,
            inputs=["?obj", "?grasp", "?placement", "?arm"],
            batch_size=batch_size,
            greedy_outputs=greedy_outputs,
        ),
    ]
    return streams


def create_placeable(world: World, stackable: bool = False) -> List[Evaluation]:
    initial = []
    for obj in world.movable_objects:
        initial.extend(Graspable(arm, obj.name) for arm in world.arms)
        for support in world.objects:
            if support.stackable or (stackable and support.movable):
                initial.append(Placeable(obj.name, support.name))
    return initial


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
    initial.extend(create_placeable(world))

    return initial


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
    elif task == "hold":
        goals = [Holding(arm) <= obj for arm, obj in zip(world.arm_names, world.movable_names)]
    elif task in ["pack", "unpack"]:
        surface = free_objects[0]
        goals = [Attached(obj) == surface for obj in world.movable_names]
    elif task == "stack":
        goals = [Attached(obj1) == obj2 for obj1, obj2 in get_pairs(world.movable_names)]
        base = world.movable_names[-1]
        surface = min(world.fixed_objects, key=lambda o: o.area).name
        goals.append(Attached(base) == surface)
    elif task == "sort":
        goals = [Attached(obj1) == obj2 for obj1, obj2 in zip(world.movable_names, free_objects)]
    elif task == "shelf":
        goals = [Attached(obj) == "shelf" for obj in world.movable_names]
    elif task == "relay":
        goals = [Attached(obj) == "mid_platform" for obj in world.movable_names]
    else:
        raise ValueError(task)

    if (max_goals is not None) and (len(goals) > max_goals):
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
            if isinstance(formula.function, AppliedPredicate):
                action = formula.function.action
                parameter_values = formula.term.parameter_values
                if action.name == "place":
                    top_obj = parameter_values["?obj"].unwrap()
                    bottom_obj = parameter_values["?o2"].unwrap()
                    initial.append(Placeable(top_obj, bottom_obj))
            elif formula.function == Attached:
                (top_obj,) = unwrap_arguments(formula.inputs)
                bottom_obj = formula.output.unwrap()
                if bottom_obj in world.object_names:
                    initial.append(Placeable(top_obj, bottom_obj))
    return initial


def create_world(
    task: str,
    num: int = 2,
    single: bool = False,
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
            num_robots=1 if single else 2,
            num_objects=num,
            floor_width=0.6,
            floor_depth=1.0,
            platform_width=0.4,
            **kwargs,
        )
        floor = max(world.fixed_objects, key=lambda o: o.area)
        for obj in world.movable_objects:
            obj.stack(floor)
    elif task == "unpack":
        world = create_franka_world(
            num_robots=1 if single else 2,
            num_objects=num,
            floor_width=0.6,
            floor_depth=1.0,
            platform_width=0.2,
            **kwargs,
        )
        platform = min(world.fixed_objects, key=lambda o: o.area)
        for obj in world.movable_objects:
            obj.stack(platform)
    elif task == "stack":
        world = create_franka_world(
            num_robots=1 if single else 2,
            num_objects=num,
            floor_width=0.6,
            floor_depth=1.0,
            platform_width=0.1,
            **kwargs,
        )
        floor = max(world.fixed_objects, key=lambda o: o.area)
        for obj in world.movable_objects:
            obj.stack(floor)
    elif task == "sort":
        world = create_franka_platforms_world(num_platforms=num, num_objects=num, **kwargs)
    elif task == "shelf":
        world = create_franka_shelf_world(num_objects=num, **kwargs)
    elif task == "relay":
        world = create_franka_relay_world(num_objects=num, **kwargs)
    else:
        raise ValueError(task)

    closed_conf, opened_conf = world.get_gripper_limits()
    world.set_joint_positions(world.gripper_joints, opened_conf)
    world.set_retract_conf()
    return world


def solve_tamp(
    state: State,
    goal: Formula,
    prompt: Optional[str] = None,
    hierarchical: bool = True,
    sequential: Optional[bool] = None,
    collisions: bool = True,
    debug: bool = False,
    profile: bool = False,
    **kwargs,
) -> Optional[Commands]:
    world = state.world
    world.dump()
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
    if prompt is not None:
        print(SEPARATOR)
        llm_problem = problem.clone(
            initial=problem.initial.new_state(create_placeable(world, stackable=True))
        )
        context = world.description()
        if True:
            partial_plan = solve_llm(llm_problem, prompt, context)
            problem.add_partial_plan(partial_plan)
        else:
            problem.goal &= interpret_llm(llm_problem, prompt, context)
        problem.initial = problem.initial.new_state(initial_from_goal(world, problem.goal))

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
        ArmObjCollision.set_definition(false_test)

    print(SEPARATOR)

    with profiler(field="tottime" if profile else None, num=25):
        solutions = solve(
            problem,
            sequential=sequential,
            lazy=True,
            heuristic_fn="relaxed",
            successor_fn="offline",
            weight=5,
            satisfy_time=1.0,
            optimize_time=1.0,
            num_reschedule=1,
            **kwargs,
        )
    print(SEPARATOR)
    if not solutions:
        return None

    if hierarchical:
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

    if len(world.arms) <= 1:
        return extract_sequential_commands(state, solution.plan)
    if sequential:
        return extract_timed_commands(state, solution.plan)
    with state:
        return postprocess_plan(state, solution.plan, collisions=collisions)


def example(
    task: str = "hold",
    num: int = 2,
    single: bool = False,
    prompt: Optional[str] = None,
    goals: Optional[int] = None,
    reset: bool = True,
    batch: int = 1,
    seed: Optional[int] = 0,
    visualize: bool = False,
    video: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Commands]:
    np.set_printoptions(precision=2, suppress=True, threshold=3)
    set_seed(seed=seed)

    world = create_world(task, num=num, single=single)
    world.initialize(batch)
    state = next(generate_states(world))
    assert state is not None

    goal = create_goal(state, task=task if prompt is None else None, max_goals=goals, reset=reset)
    commands = solve_tamp(state, goal, prompt=prompt, **kwargs)
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
        choices=["hold", "pack", "unpack", "stack", "sort", "shelf", "relay"],
        help="The task name.",
    )
    parser.add_argument("-n", "--num", default=2, type=int, help="The number of objects.")
    parser.add_argument("--single", action="store_true", help="Use a single robot when applicable.")
    parser.add_argument("-g", "--goals", default=None, type=int, help="The number of goals.")
    parser.add_argument("--no-reset", action="store_true", help="Disables adding a reset goal.")
    parser.add_argument("--prompt", type=str, default=None, help="The task prompt.")
    parser.add_argument("-b", "--batch", type=int, default=1, help="The stream batch size.")
    parser.add_argument("-c", "--cfree", action="store_true", help="Enables collision free.")
    parser.add_argument("-d", "--debug", action="store_true", help="Run with debug streams")
    parser.add_argument(
        "-s", "--seed", nargs="?", const=None, default=0, type=int, help="The random seed."
    )
    parser.add_argument("-p", "--profile", action="store_true", help="Enables profiling")
    parser.add_argument("-v", "--video", type=str, nargs="?", const="", default=None)
    parser.add_argument("--headless", action="store_true", help="Disables visualization.")
    args = parser.parse_args()
    print("Args:", args)

    example(
        task=args.task,
        num=args.num,
        single=args.single,
        prompt=args.prompt,
        goals=args.goals,
        reset=not args.no_reset,
        batch=args.batch,
        collisions=not args.cfree,
        debug=args.debug,
        seed=args.seed,
        profile=args.profile,
        video=args.video,
        visualize=not args.headless,
        verbose=True,
    )


if __name__ == "__main__":
    main()
