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
from typing import Any, Optional

# NVIDIA
from schedulestream.algorithm.temporal import TimedPlan, sequential_from_timed
from schedulestream.algorithm.utils import compute_partial_orders, visualize_plan
from schedulestream.applications.custream.animate import (
    combine_timed_commands,
    decompose_timed_action,
)
from schedulestream.applications.custream.collision import joint_state_pair_collision
from schedulestream.applications.custream.command import Composite
from schedulestream.applications.custream.state import State
from schedulestream.applications.custream.utils import concatenate_joint_states
from schedulestream.common.milp import Constraint, Cost, Variable, solve_milp
from schedulestream.common.utils import flatten, get_pairs
from schedulestream.language.durative import EndInstance, StartInstance


def postprocess_plan(
    state: State,
    timed_plan: Optional[TimedPlan],
    makespan_weight: Optional[float] = 1.0,
    individual_weight: Optional[float] = 1e-2,
    delta_weight: Optional[float] = 1e-2,
    collision_duration: float = 0.0,
    action_orders: bool = True,
    leader: bool = False,
    collisions: bool = True,
    max_time: float = 5.0,
    visualize: bool = False,
    **kwargs: Any,
):
    if timed_plan is None:
        return None
    world = state.world
    t0 = 0.0
    dt = world.time_step

    state.set()
    arm_times = {}
    arm_commands = {}
    arm_confs = {}
    action_indices = {}
    for timed_action in timed_plan:
        for t, composite in decompose_timed_action(timed_action):
            if isinstance(composite, Composite):
                commands = composite.commands
            else:
                commands = [composite]
            for command in commands:
                arm = command.arm
                index = len(arm_times.get(arm, []))
                arm_times.setdefault(arm, []).append(t)
                arm_commands.setdefault(arm, []).append(command)
                conf = world.arm_configuration(arm)
                arm_confs.setdefault(arm, []).append(conf)
                action_indices.setdefault(timed_action.action, {}).setdefault(arm, []).append(index)

    constraints = []
    costs = []

    makespan = Variable(name="makespan", lower=0.0)
    if makespan_weight is not None:
        costs.append(
            Cost(
                coefficients={makespan.name: makespan_weight},
            )
        )

    delta = Variable(name="delta", lower=0.0)
    if delta_weight is not None:
        costs.append(
            Cost(
                coefficients={delta.name: delta_weight},
            )
        )

    arm_variables = {}
    for arm, confs in arm_confs.items():
        arm_variables[arm] = [Variable(name=(arm, index), lower=t0) for index in range(len(confs))]
        for t1, t2 in get_pairs(arm_variables[arm]):
            constraints.append(
                Constraint(
                    lower=dt,
                    coefficients={
                        t2.name: +1,
                        t1.name: -1,
                    },
                )
            )

        end_variable = arm_variables[arm][-1]
        constraints.append(
            Constraint(
                lower=0.0,
                coefficients={
                    makespan.name: +1,
                    end_variable.name: -1,
                },
            )
        )
        if individual_weight is not None:
            costs.append(
                Cost(
                    coefficients={end_variable.name: individual_weight},
                )
            )

    for action, arm_indices in action_indices.items():
        for arm, indices in arm_indices.items():
            for index1, index2 in get_pairs(indices):
                constraints.extend(
                    [
                        Constraint(
                            coefficients={
                                arm_variables[arm][index2].name: +1,
                                arm_variables[arm][index1].name: -1,
                                delta.name: -1,
                            },
                            upper=0,
                        ),
                    ]
                )

    if action_orders:
        sequential_plan = sequential_from_timed(timed_plan, over=False)
        if visualize:
            visualize_plan(sequential_plan)

        for event1, event2 in compute_partial_orders(sequential_plan):
            durative1 = event1.durative_instance
            durative2 = event2.durative_instance
            for arm1, arm2 in itertools.product(
                action_indices[durative1], action_indices[durative2]
            ):
                if arm1 == arm2:
                    continue
                if isinstance(event1, StartInstance):
                    index1 = action_indices[durative1][arm1][0]
                elif isinstance(event1, EndInstance):
                    index1 = action_indices[durative1][arm1][-1]
                else:
                    raise NotImplementedError(event1)
                if isinstance(event2, StartInstance):
                    index2 = action_indices[durative2][arm2][0]
                elif isinstance(event2, EndInstance):
                    index2 = action_indices[durative2][arm2][-1]
                else:
                    raise NotImplementedError(event2)
                print(f"Arm1: {arm1} | Index1: {index1} | Arm2: {arm2} | Index2: {index2}")
                constraints.append(
                    Constraint(
                        lower=dt,
                        coefficients={
                            arm_variables[arm2][index2].name: +1,
                            arm_variables[arm1][index1].name: -1,
                        },
                    )
                )

    end_times = {}
    for timed_action in timed_plan:
        arguments = timed_action.unwrap_arguments()
        arm = arguments[0]
        end_times[arm] = max(timed_action.end, end_times.get(arm, 0))
    arm_order = sorted(end_times, key=end_times.get)
    print("Arm order:", arm_order)

    if collisions:
        for arm1, arm2 in itertools.combinations(arm_order, 2):
            joint_state1 = concatenate_joint_states(conf.joint_state for conf in arm_confs[arm1])
            joint_state2 = concatenate_joint_states(conf.joint_state for conf in arm_confs[arm2])
            colliding_pairs = joint_state_pair_collision(
                world, joint_state1, joint_state2, strict=True
            )

            for index1, index2 in colliding_pairs:
                t1 = arm_times[arm1][index1]
                t2 = arm_times[arm2][index2]

                _arm1, _arm2 = arm1, arm2
                _index1, _index2 = index1, index2
                if not leader and (t1 > t2):
                    _arm1, _arm2 = _arm2, _arm1
                    _index1, _index2 = _index2, _index1
                constraints.append(
                    Constraint(
                        lower=collision_duration,
                        coefficients={
                            arm_variables[_arm2][_index2].name: +1,
                            arm_variables[_arm1][_index1 + 1].name: -1,
                        },
                    )
                )

    variables = list(flatten(arm_variables.values())) + [makespan, delta]
    solution = solve_milp(
        variables,
        constraints=constraints,
        costs=costs,
        gap_percent=None,
        max_time=max_time,
        verbose=True,
        **kwargs,
    )
    if solution is None:
        return None

    timed_commands = []
    for arm, variables in arm_variables.items():
        for variable in variables:
            t = solution[variable.name]
            _, index = variable.name
            timed_commands.append((t, arm_commands[arm][index]))
    return combine_timed_commands(world, timed_commands)
