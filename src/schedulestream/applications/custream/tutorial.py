#! /usr/bin/env python3
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
from schedulestream.algorithm.stream.common import StreamSolution
from schedulestream.algorithm.stream.solver import STREAM_ALGORITHMS
from schedulestream.applications.custream.animate import animate_commands, combine_timed_commands
from schedulestream.applications.custream.collision import (
    commands_conf_collision,
    trajectory_placement_collision,
)
from schedulestream.applications.custream.command import Attach, Trajectory
from schedulestream.applications.custream.grasp import Grasp, grasp_generator
from schedulestream.applications.custream.initialize import generate_states
from schedulestream.applications.custream.placement import Placement as PlacementPose
from schedulestream.applications.custream.scene import create_franka_line_world
from schedulestream.applications.custream.streams import ik_stream, motion_stream
from schedulestream.applications.custream.utils import set_seed
from schedulestream.applications.custream.world import World
from schedulestream.common.utils import current_time, elapsed_time
from schedulestream.language.action import Action
from schedulestream.language.anonymous import rename_anonymous
from schedulestream.language.argument import OUTPUT
from schedulestream.language.durative import DurativeAction
from schedulestream.language.function import NumericFunction
from schedulestream.language.generator import from_unary_fn, from_unary_gen_fn
from schedulestream.language.predicate import Function, Predicate
from schedulestream.language.problem import Problem


def check_obj_collision(world: World, traj: Trajectory, placement: Optional[PlacementPose]) -> bool:
    if isinstance(placement, Grasp):
        return False
    return trajectory_placement_collision(world, traj.arm, traj, placement.obj, placement)


def check_arm_collision(
    world: World,
    traj1: Trajectory,
    traj2: Trajectory,
) -> bool:
    if traj1.arm == traj2.arm:
        return False
    return commands_conf_collision(world, traj1.arm, traj1, traj2.arm, traj2)


def tutorial(
    num: int = 2,
    seed: Optional[int] = 0,
    reset: bool = False,
    collisions: bool = True,
    debug: bool = False,
    visualize: bool = False,
    **kwargs: Any,
) -> List[StreamSolution]:
    set_seed(seed=seed)
    world = create_franka_line_world(num_robots=num, num_objects=num)
    for obj in world.movable_objects:
        obj.grasp_config.pitch_interval = "top"
    state = next(generate_states(world))
    assert state is not None

    Arm = Predicate("?arm")
    Conf = Predicate("?arm ?q")
    At = Function("?arm", condition=[Arm("?arm"), Conf(f"?arm {OUTPUT}")])

    Object = Predicate("?obj")
    Placement = Predicate("?obj ?p")
    Pose = Function("?obj", condition=[Object("?obj")])
    Holding = Function("?arm", condition=[Arm("?arm")])
    Attached = Function("?obj", condition=[Object("?obj")])

    Grasp = Predicate("?arm ?obj ?g", condition=[Arm("?arm"), Object("?obj")])
    Kin = Predicate(
        "?arm ?q ?obj ?g ?p",
        condition=[Conf("?arm ?q"), Placement("?obj ?p"), Grasp("?arm ?obj ?g")],
    )
    Motion = Predicate("?arm ?q1 ?t ?q2", condition=[Conf("?arm ?q1"), Conf("?arm ?q2")])

    Duration = NumericFunction(
        "?t",
        definition=lambda t: t.duration,
        lazy=True,
    )
    ObjCollision = Predicate(
        "?t ?p",
        definition=partial(check_obj_collision, world) if collisions else lambda *args: False,
        lazy=True,
    )
    ArmCollision = Predicate(
        "?t1 ?t2",
        definition=partial(check_arm_collision, world) if collisions else lambda *args: False,
        lazy=True,
    )

    rename_anonymous(locals())

    initial = []
    goal = []
    for arm in world.arms:
        q = state.arm_configuration(arm)
        initial.extend(
            [
                At(arm) <= q,
                Holding(arm) <= None,
            ]
        )
        if reset:
            goal.append(At(arm) <= q)

    for obj in world.movable_names:
        p = PlacementPose(world, obj)
        initial.extend(
            [
                Pose(obj) <= p,
                Attached(obj) <= None,
                Placement(obj, p),
            ]
        )

    for arm, obj in zip(world.arms, world.movable_names):
        goal.append(Holding(arm) == obj)

    collision_condition = [~ObjCollision("?t", Pose(o)) for o in world.movable_names] + [
        ~ArmCollision("?t", At(a)) for a in world.arms
    ]
    actions = [
        DurativeAction(
            name="move",
            parameters="?arm ?q1 ?t ?q2",
            start_condition=[Motion("?arm ?q1 ?t ?q2"), At("?arm") == "?q1"],
            start_effect=[At("?arm") <= "?t"],
            over_condition=collision_condition,
            end_condition=[],
            end_effect=[At("?arm") <= "?q2"],
            min_duration=Duration("?t"),
        ),
        Action(
            name="pick",
            parameters="?arm ?q ?obj ?g ?p",
            precondition=[
                Kin("?arm ?q ?obj ?g ?p"),
                Pose("?obj") == "?p",
                At("?arm") == "?q",
                Attached("?obj") == None,
                Holding("?arm") == None,
            ],
            effect=[Holding("?arm") <= "?obj", Pose("?obj") <= "?g", Attached("?obj") <= "?arm"],
        ),
    ]

    streams = [
        Grasp.stream(
            inputs="?arm ?obj",
            conditional_generator=from_unary_gen_fn(
                lambda arm, obj: grasp_generator(world, arm=arm, obj=obj)
            ),
        ),
        Kin.stream(
            inputs="?arm ?obj ?g ?p",
            conditional_generator=from_unary_gen_fn(partial(ik_stream, world)),
        ),
        Motion.stream(
            inputs="?arm ?q1 ?q2",
            conditional_generator=from_unary_fn(
                partial(motion_stream, world, collisions=collisions)
            ),
        ),
    ]
    problem = Problem(
        initial=initial,
        goal=goal,
        actions=actions,
        streams=streams,
    )
    if debug:
        problem = problem.lazy_clone()
    problem.dump()

    start_time = current_time()
    solutions = solve(
        problem,
        lazy=False,
        heuristic_fn="hmax",
        successor_fn="offline",
        weight=1,
        **kwargs,
    )
    if not solutions:
        print(f"Solved: {False} | Elapsed: {elapsed_time(start_time):.3f} sec")
        if visualize:
            world.show()
        return solutions

    solution = solutions[0]
    plan = solution.plan
    print(
        f"Solved: {True} | Cost: {solution.cost:.3f} | Makespan: {solution.makespan:.3f} | "
        f"Elapsed: {elapsed_time(start_time):.3f} sec\nPlan ({len(plan)}): {plan}"
    )
    if not visualize:
        return solutions
    if debug:
        world.show()
        return solutions

    timed_commands = []
    for timed_action in plan:
        action_time = timed_action.start
        if timed_action.name == "move":
            traj = timed_action.parameter_values["?t"].unwrap()
            for command in traj.decompose():
                timed_commands.append((action_time, command))
                action_time += command.duration
        elif timed_action.name == "pick":
            arm, _, obj, _, _ = timed_action.unwrap_arguments()
            arm_link = world.get_arm_link(arm)
            timed_commands.append((action_time, Attach(world, obj, arm_link)))
        else:
            raise NotImplementedError(timed_action.name)
    timed_commands.sort()
    commands = combine_timed_commands(world, timed_commands)
    animate_commands(state, commands)
    return solutions


def main():
    np.set_printoptions(precision=2, suppress=True, threshold=3)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seed", nargs="?", const=None, default=0, type=int, help="The random seed."
    )
    parser.add_argument(
        "-n", "--num", default=2, type=int, help="The number of robots and objects."
    )
    parser.add_argument("-r", "--reset", action="store_true", help="Resets the robot states.")
    parser.add_argument("-c", "--cfree", action="store_true", help="Enables collision free.")
    parser.add_argument("-d", "--debug", action="store_true", help="Run with debug streams")
    parser.add_argument(
        "-a", "--algorithm", type=str, default="focused", choices=list(STREAM_ALGORITHMS)
    )
    args = parser.parse_args()
    print("Args:", args)

    tutorial(
        num=args.num,
        seed=args.seed,
        reset=args.reset,
        collisions=not args.cfree,
        debug=args.debug,
        algorithm=args.algorithm,
        visualize=True,
    )


if __name__ == "__main__":
    main()
