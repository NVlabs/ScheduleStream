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
import itertools
from typing import Any, List, Optional, Tuple

# Third Party
import numpy as np

# NVIDIA
from schedulestream.algorithm.temporal import TimedPlan, distinct_plan, sequential_from_timed
from schedulestream.algorithm.utils import compute_partial_orders, visualize_plan
from schedulestream.applications.custream2.animate import (
    combine_timed_commands,
    decompose_timed_action,
)
from schedulestream.applications.custream2.collision import conf_pair_collision
from schedulestream.applications.custream2.command import Commands, Composite, Trajectory
from schedulestream.applications.custream2.state import Configuration, State
from schedulestream.applications.custream2.world import World
from schedulestream.common.milp import Constraint, Cost, Variable, solve_milp
from schedulestream.common.utils import flatten, get_pairs
from schedulestream.language.durative import EndInstance, StartInstance


def _earliest_collision_per_i(colliding_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """For each i, keep only the pair with the smallest j (earliest collision target)."""
    min_j_per_i: dict = {}
    for i, j in colliding_pairs:
        if i not in min_j_per_i or j < min_j_per_i[i]:
            min_j_per_i[i] = j
    return list(min_j_per_i.items())


def _plot_arm_schedules(arm_variables: dict, solution: dict, progress: bool = True) -> None:
    """Plot per-arm MILP solution times, with progress [0,1] or step index on the y-axis."""
    # Third Party
    import matplotlib.pyplot as plt

    markers = ["x", "+", "o", "s", "^", "v"]
    fig, ax = plt.subplots()
    for i, (arm, variables) in enumerate(arm_variables.items()):
        times = [solution[v.name] for v in variables]
        positions = (
            np.linspace(0.0, 1.0, len(variables), endpoint=True)
            if progress
            else list(range(len(variables)))
        )
        plt.plot(times, positions, label=arm, marker=markers[i % len(markers)])
    ax.set_xlabel("time")
    ax.set_ylabel("progress" if progress else "step")
    plt.legend(loc="upper right")
    plt.title("MILP Schedule")
    plt.grid(False)
    plt.show()


def postprocess_plan(
    state: State,
    timed_plan: Optional[TimedPlan],
    makespan_weight: Optional[float] = 1.0,
    individual_weight: Optional[float] = 1e-2,
    delta_weight: Optional[float] = 1e-2,
    collision_duration: float = 0.0,
    action_orders: bool = True,
    collisions: bool = True,
    max_time: float = 5.0,
    visualize: bool = False,
    **kwargs: Any,
) -> Optional[Commands]:
    if timed_plan is None:
        return None
    timed_plan = distinct_plan(timed_plan)
    world = state.world
    t0 = 0.0
    dt = world.time_step

    state.set()
    task_space = False
    arm_times: dict = {}
    arm_commands: dict = {}
    arm_confs: dict = {}
    action_indices: dict = {}
    arm_last_conf: dict = {}
    for timed_action in timed_plan:
        arm = timed_action.unwrap_arguments()[0]
        command = timed_action.unwrap_arguments()[-1]
        task_space |= command.is_task_space
        for t, composite in decompose_timed_action(timed_action, task_space=task_space):
            for command in Composite.flatten([composite]):
                index = len(arm_times.get(arm, []))
                arm_times.setdefault(arm, []).append(t)
                arm_commands.setdefault(arm, []).append(command)
                if isinstance(command, Configuration):
                    conf = command.extract_configuration(world.get_arm_joints(arm))
                    arm_last_conf[arm] = conf
                else:
                    conf = arm_last_conf.get(arm) or world.arm_configuration(arm)
                arm_confs.setdefault(arm, []).append(conf)
                action_indices.setdefault(timed_action.action, {}).setdefault(arm, []).append(index)
    state.set()

    constraints = []
    costs = []

    makespan = Variable(name="makespan", lower=0.0)
    if makespan_weight is not None:
        costs.append(Cost(coefficients={makespan.name: makespan_weight}))

    delta = Variable(name="delta", lower=0.0)
    if delta_weight is not None:
        costs.append(Cost(coefficients={delta.name: delta_weight}))

    arm_variables: dict = {}
    for arm, confs in arm_confs.items():
        arm_variables[arm] = [Variable(name=(arm, index), lower=t0) for index in range(len(confs))]
        for (v1, v2), (tp, tn) in zip(get_pairs(arm_variables[arm]), get_pairs(arm_times[arm])):
            lower = dt if tn > tp else 0.0
            constraints.append(Constraint(lower=lower, coefficients={v2.name: +1, v1.name: -1}))
        end_variable = arm_variables[arm][-1]
        constraints.append(
            Constraint(lower=0.0, coefficients={makespan.name: +1, end_variable.name: -1})
        )
        if individual_weight is not None:
            costs.append(Cost(coefficients={end_variable.name: individual_weight}))

    for action, arm_indices in action_indices.items():
        for arm, indices in arm_indices.items():
            for index1, index2 in get_pairs(indices):
                constraints.append(
                    Constraint(
                        coefficients={
                            arm_variables[arm][index2].name: +1,
                            arm_variables[arm][index1].name: -1,
                            delta.name: -1,
                        },
                        upper=0,
                    )
                )

    if action_orders:
        sequential_plan = sequential_from_timed(timed_plan, over=False)
        if visualize:
            visualize_plan(sequential_plan)
        for event1, event2 in compute_partial_orders(sequential_plan):
            durative1 = event1.durative_instance
            durative2 = event2.durative_instance
            for arm1, arm2 in itertools.product(
                action_indices.get(durative1, {}), action_indices.get(durative2, {})
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
                constraints.append(
                    Constraint(
                        lower=dt,
                        coefficients={
                            arm_variables[arm2][index2].name: +1,
                            arm_variables[arm1][index1].name: -1,
                        },
                    )
                )

    end_times: dict = {}
    for timed_action in timed_plan:
        arguments = timed_action.unwrap_arguments()
        arm = arguments[0]
        end_times[arm] = max(timed_action.end, end_times.get(arm, 0))
    arm_order = sorted(end_times, key=end_times.get)

    if collisions:
        for arm1, arm2 in itertools.combinations(arm_order, 2):
            colliding_pairs = _earliest_collision_per_i(
                conf_pair_collision(world, arm1, arm2, arm_confs[arm1], arm_confs[arm2])
            )
            for index1, index2 in colliding_pairs:
                if index1 + 1 < len(arm_variables[arm1]):
                    constraints.append(
                        Constraint(
                            lower=collision_duration,
                            coefficients={
                                arm_variables[arm2][index2].name: +1,
                                arm_variables[arm1][index1 + 1].name: -1,
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

    if visualize:
        _plot_arm_schedules(arm_variables, solution)

    timed_commands = []
    for arm in reversed(arm_order):
        for variable in arm_variables[arm]:
            t = solution[variable.name]
            _, index = variable.name
            timed_commands.append((t, arm_commands[arm][index]))
    return combine_timed_commands(world, timed_commands)
