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
"""cuStream2 TAMP domain (predicates, actions, streams) and the solve_tamp entry point."""

from __future__ import annotations

# Standard Library
import math
from functools import partial
from typing import Any, List, Optional, Set, Tuple, Union

# Third Party
import numpy as np

# NVIDIA
from schedulestream.algorithm.llm import interpret_llm, solve_llm
from schedulestream.algorithm.solver import solve_hierarchical
from schedulestream.applications.custream2.animate import (
    extract_sequential_commands,
    extract_timed_commands,
)
from schedulestream.applications.custream2.collision import (
    commands_conf_collision,
    commands_placement_collision,
    object_object_collision,
)
from schedulestream.applications.custream2.command import Commands
from schedulestream.applications.custream2.costs import distance_function
from schedulestream.applications.custream2.grasp import grasp_stream
from schedulestream.applications.custream2.initialize import (
    get_stacked_objects,
    order_stacked_objects,
)
from schedulestream.applications.custream2.motion import motion_stream
from schedulestream.applications.custream2.placement import Placement as PlacementPose
from schedulestream.applications.custream2.placement import (
    concatenate_placement_poses,
    placement_stream,
)
from schedulestream.applications.custream2.postprocess import postprocess_plan
from schedulestream.applications.custream2.state import State
from schedulestream.applications.custream2.streams import (
    pick_stream,
    place_stream,
)
from schedulestream.applications.custream2.utils import (
    free_gpu_memory,
    position_from_pose,
    to_cpu,
    transform_points,
)
from schedulestream.applications.custream2.world import World
from schedulestream.common.utils import (
    INF,
    SEPARATOR,
    current_time,
    elapsed_time,
    profiler,
    remove_duplicates,
)
from schedulestream.language.action import Action, AppliedPredicate
from schedulestream.language.anonymous import rename_anonymous
from schedulestream.language.argument import OUTPUT, Constant, unwrap_arguments
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
    negate_test,
    scale_fn,
)
from schedulestream.language.lazy import LazyOutput
from schedulestream.language.predicate import FUNCTIONS, Predicate, Type
from schedulestream.language.problem import Problem
from schedulestream.language.stream import PredicateStream, Stream

Arm = Type()
Conf = Predicate(["?arm", "?conf"], condition=Arm("?arm"))
Traj = Predicate(["?arm", "?traj"], condition=Arm("?arm"))
At = Function(["?arm"], condition=Conf("?arm", OUTPUT))
Holding = Function(["?arm"], condition=Arm("?arm"), language=f"Arm ?arm is holding object {OUTPUT}")

Object = Type()
AtPose = Function(["?obj"], condition=Object("?obj"))
Attached = Function(
    ["?obj"], condition=Object("?obj"), language=f"Object ?obj is attached to {OUTPUT}"
)
Supporting = Predicate(["?obj"], condition=Object("?obj"))

Graspable = Predicate(["?arm", "?obj"], condition=Arm("?arm") & Object("?obj"))
Placeable = Predicate(["?obj", "?obj2"], condition=Object("?obj") & Object("?obj2"))
Grasped = Predicate(["?obj"], condition=Object("?obj"), language="Object ?obj is grasped")
Placed = Predicate(["?obj"], condition=Object("?obj"), language="Object ?obj is placed")

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
Assignment = NumericFunction(
    ["?arm", "?obj"],
    condition=Arm("?arm") & Object("?obj"),
    definition=None,
)
ArmArmCollision = Predicate(
    ["?arm1", "?traj1", "?arm2", "?conf2"],
    definition=None,
)
ArmObjCollision = Predicate(
    ["?arm1", "?traj1", "?obj", "?pose"],
    definition=None,
    batch_size=INF,
)
ObjObjCollision = Predicate(
    ["?obj1", "?placement1", "?obj2", "?placement2"],
    definition=None,
    batch_size=INF,
)
ObjObjRelation = Predicate(
    ["?relation", "?obj1", "?placement1", "?obj2", "?placement2"],
    definition=None,
    batch_size=INF,
)

TAMP_FUNCTIONS = list(filter(lambda f: isinstance(f, Function), locals().values()))

rename_anonymous(locals())


def reset_globals() -> None:
    for function in FUNCTIONS + TAMP_FUNCTIONS:
        if isinstance(function, Function):
            function.reset()
    LazyOutput.reset()
    Constant.reset()
    free_gpu_memory()


def compute_assignment_distance(world: World, arm: str, obj: str) -> float:
    """Distance from ``arm``'s base link to ``obj``'s (initial) pose; ``pair = (arm, obj)``."""
    base_position = position_from_pose(world.get_link_pose(world.get_arm_base_link(arm)))
    obj_position = position_from_pose(world.get_object_pose(obj))
    return float(np.linalg.norm(obj_position - base_position))


PROXIMITY = "proximity"

RELATION_AXES = {
    "left": (1, +1),
    "right": (1, -1),
    "behind": (0, +1),
    "front": (0, -1),
}


def object_object_relation(
    world: World,
    batch_inputs: List[Tuple[str, str, Any, str, Any]],
    cone_deg: float = 45.0,
    max_separation: Tuple[float, float, float] = (0.2, 0.2, 0.2),
    verbose: bool = True,
) -> List[bool]:
    """Batched spatial relation between two placed objects.

    The directional relations come from RoboLab's ``spatial_condition_check_vector_based``:
    obj1 satisfies ``relation`` when the obj2->obj1 vector, flattened to XY in the arm-base
    frame, points within ``cone_deg`` of that relation's axis. The cone is what distinguishes
    this from a bare ``y1 > y2`` half-plane -- an object far out in X barely offset in Y is
    not "left".

    ``proximity`` instead comes from Arena's ``objects_in_proximity``: an axis-aligned test
    on the per-axis centre separation against ``max_separation`` (x, y, z), evaluated in world
    axes rather than the arm-base frame, and keeping Z rather than flattening it. The bound is
    centre-to-centre, so it must exceed the two objects' half-extents or no non-colliding
    placement can satisfy it and the planner resamples forever.

    Inputs whose placement is a Grasp (held) rather than a Placement are False.
    """
    t0 = current_time()
    num = len(batch_inputs)
    if num == 0:
        return []
    relations1, obj1s, placement1s, obj2s, placement2s = zip(*batch_inputs)
    [relation] = set(relations1)
    [obj1] = set(obj1s)
    [obj2] = set(obj2s)
    outputs = num * [True]
    if obj1 == obj2:
        return outputs
    if (relation != PROXIMITY) and (relation not in RELATION_AXES):
        raise NotImplementedError(
            f"{relation!r}; expected one of {sorted([PROXIMITY] + list(RELATION_AXES))}"
        )
    indices = [
        i
        for i, (p1, p2) in enumerate(zip(placement1s, placement2s))
        if isinstance(p1, PlacementPose) and isinstance(p2, PlacementPose)
    ]
    if not indices:
        return outputs

    pose1 = concatenate_placement_poses([placement1s[i] for i in indices])
    pose2 = concatenate_placement_poses([placement2s[i] for i in indices])

    if relation == PROXIMITY:
        separation = np.abs(to_cpu(pose1.position) - to_cpu(pose2.position))
        holds = np.all(separation < np.asarray(max_separation, dtype=separation.dtype), axis=-1)
    else:
        base_link = world.get_arm_base_link(world.arms[0])
        base_inverse = world.get_link_pose(base_link).inverse()
        local1 = to_cpu(transform_points(base_inverse, pose1.position))[0]
        local2 = to_cpu(transform_points(base_inverse, pose2.position))[0]

        axis, sign = RELATION_AXES[relation]
        delta = local1 - local2
        delta[:, 2] = 0.0
        norm = np.linalg.norm(delta, axis=-1)
        component = sign * delta[:, axis]
        cos_theta = component / np.clip(norm, 1e-8, None)
        holds = (norm > 1e-6) & (component > 0.0) & (cos_theta >= math.cos(math.radians(cone_deg)))
    for i, value in zip(indices, holds.tolist()):
        outputs[i] = value
    if verbose:
        print(
            f"{object_object_relation.__name__}) Relation: {relation} | Object 1: {obj1} | Object"
            f" 2: {obj2} | Satisfying: {sum(outputs)}/{num} | Elapsed: {elapsed_time(t0):.3f} sec"
        )
    return outputs


def create_actions(world: World, cost: str = "joint", collisions: bool = True) -> List[Action]:
    Duration.set_definition(lambda *pair: pair[1].duration)
    Distance.set_definition(
        scale_fn(
            partial(distance_function, world, cost=cost, holding_weight=2, num_steps=1), scale=1
        )
    )
    Assignment.set_definition(scale_fn(partial(compute_assignment_distance, world), scale=1e-3))
    ArmArmCollision.set_definition(
        partial(commands_conf_collision, world) if collisions and world.multi_arm else false_test
    )
    ArmObjCollision.set_definition(
        partial(commands_placement_collision, world) if collisions else batch_fn_from_fn(false_test)
    )
    ObjObjCollision.set_definition(
        partial(object_object_collision, world) if collisions else batch_fn_from_fn(false_test)
    )
    ObjObjRelation.set_definition(negate_test(partial(object_object_relation, world)))

    arm_colliders = world.arm_names if collisions else []
    if len(arm_colliders) <= 1:
        arm_colliders = []
    object_colliders = world.movable_names if collisions else []
    arm_conditions = [~ArmArmCollision("?arm", "?traj", arm, At(arm)) for arm in arm_colliders]
    obj_conditions = [
        ~ArmObjCollision("?arm", "?traj", obj, AtPose(obj)) for obj in object_colliders
    ]
    traj_condition = Conjunction(*arm_conditions, *obj_conditions)
    placement_condition = Conjunction(
        *(~ObjObjCollision("?obj", "?placement", obj, AtPose(obj)) for obj in object_colliders)
    )

    move = DurativeAction(
        parameters="?arm ?conf1 ?conf2 ?traj",
        start_condition=(At("?arm") == "?conf1"),
        start_effect=At("?arm") <= "?traj",
        over_condition=Motion("?arm", "?conf1", "?conf2", "?traj") & traj_condition,
        end_effect=At("?arm") <= "?conf2",
        min_duration=Duration("?arm", "?traj"),
    )
    pick = DurativeAction(
        parameters="?arm ?obj ?grasp ?placement ?conf1 ?conf2 ?traj",
        start_condition=(At("?arm") == "?conf1")
        & (AtPose("?obj") == "?placement")
        & (Holding("?arm") == None)
        & ~Supporting("?obj"),
        start_effect=(At("?arm") <= "?traj")
        & (AtPose("?obj") <= None)
        & (Holding("?arm") <= False),
        over_condition=Pick("?arm", "?obj", "?grasp", "?placement", "?conf1", "?conf2", "?traj")
        & traj_condition,
        end_effect=(At("?arm") <= "?conf2")
        & (Holding("?arm") <= "?obj")
        & (AtPose("?obj") <= "?grasp")
        & (Attached("?obj") <= "?arm")
        & Grasped("?obj")
        & ~Placed("?obj"),
        min_duration=Duration("?arm", "?traj"),
        cost=Distance("?arm", At("?arm"), "?conf1") + Assignment("?arm", "?obj"),
        language="Arm ?arm picks object ?obj",
    )
    place = DurativeAction(
        parameters="?arm ?obj ?grasp ?placement ?o2 ?p2 ?conf1 ?conf2 ?traj",
        start_condition=(At("?arm") == "?conf1")
        & (AtPose("?obj") == "?grasp")
        & (AtPose("?o2") == "?p2")
        & (Holding("?arm") == "?obj"),
        start_effect=(At("?arm") <= "?traj")
        & (AtPose("?obj") <= None)
        & Supporting("?o2")
        & (Holding("?arm") <= False),
        over_condition=Place("?arm", "?obj", "?grasp", "?placement", "?conf1", "?conf2", "?traj")
        & Supported("?obj", "?placement", "?o2", "?p2")
        & traj_condition
        & placement_condition,
        end_effect=(At("?arm") <= "?conf2")
        & (Holding("?arm") <= None)
        & (AtPose("?obj") <= "?placement")
        & (Attached("?obj") <= "?o2")
        & Placed("?obj")
        & ~Grasped("?obj"),
        min_duration=Duration("?arm", "?traj"),
        cost=Distance("?arm", At("?arm"), "?conf1"),
        language="Arm ?arm places object ?obj on object ?o2",
    )
    rename_anonymous(locals())
    return [move, pick, place]


def create_streams(
    world: World,
    collisions: bool = True,
    motion: str = "joint",
    defer: bool = False,
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
                max_iterations=1,
            ),
            inputs=["?arm", "?obj"],
            greedy_outputs=None,
        ),
        Supported.stream(
            conditional_generator=batch_list_gen_fn(
                from_unary_gen_fn(
                    partial(placement_stream, world, object_proximity=0.0 if collisions else None)
                ),
                batch_size=world.ik_batch,
                max_iterations=1,
            ),
            inputs=["?obj1", "?obj2", "?placement2"],
            greedy_outputs=None,
        ),
        Motion.stream(
            conditional_generator=from_unary_fn(
                partial(motion_stream, world, collisions=collisions, motion=motion, defer=defer)
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
        conf = state.arm_configuration(arm, label="initial")
        initial.extend([At(arm) <= conf, Holding(arm) <= None])

    placements = {}
    stacked_objects = get_stacked_objects(world, names=world.movable_names)
    ordered_stacks = order_stacked_objects(world, stacked_objects)
    for obj, parent in ordered_stacks.items():
        parent_placement = placements.get(parent)
        placement = PlacementPose(world, obj, placement=parent_placement, label="initial")
        initial.extend(
            [
                Placement(obj, placement),
                AtPose(obj) <= placement,
                Attached(obj) <= None,
                Placed(obj),
            ]
        )
        placements[obj] = placement
    initial.extend(create_placeable(world))
    return initial


def movable_from_goal(goal: Optional[Formula]) -> Set[str]:
    """Return the movable objects that the goal's Attached/Holding evaluations
    require manipulating (the attached/held objects, not their surfaces)."""
    movable = set()
    if goal is None:
        return movable
    for formula in goal.clause:
        if not isinstance(formula, Evaluation):
            continue
        if formula.function == Attached:
            (obj,) = unwrap_arguments(formula.inputs)
            movable.add(obj)
        elif formula.function == Holding:
            movable.add(formula.output.unwrap())
    return movable


def initial_from_goal(world: World, goal: Optional[Formula]) -> List[Evaluation]:
    initial = []
    if goal is None:
        return initial
    for formula in goal.evaluations:
        initial.extend(formula.condition.clause)
        if isinstance(formula, Evaluation):
            if isinstance(formula.function, AppliedPredicate):
                action = formula.function.action
                parameter_values = formula.term.parameter_values
                if action.name == "place":
                    top_obj = parameter_values["?obj"].unwrap()
                    bottom_obj = parameter_values["?o2"].unwrap()
                    initial.extend(Graspable(arm, top_obj) for arm in world.arms)
                    initial.append(Placeable(top_obj, bottom_obj))
            elif formula.function == Attached:
                (top_obj,) = unwrap_arguments(formula.inputs)
                bottom_obj = formula.output.unwrap()
                initial.extend(Graspable(arm, top_obj) for arm in world.arms)
                if bottom_obj in world.object_names:
                    initial.append(Placeable(top_obj, bottom_obj))
    return initial


def create_problem(
    state: State,
    goal: Formula,
    collisions: bool = True,
    cost: str = "joint",
    motion: str = "joint",
    defer: bool = False,
) -> Problem:
    """Build the TAMP Problem for ``state``/``goal``: initial atoms, actions, and streams."""
    world = state.world
    initial = create_initial(state)
    initial.extend(initial_from_goal(world, goal))
    return Problem(
        initial=initial,
        goal=goal,
        actions=create_actions(world, cost=cost, collisions=collisions),
        streams=create_streams(world, motion=motion, collisions=collisions, defer=defer),
    )


def solve_tamp(
    state: State,
    goal: Formula,
    prompt: Optional[str] = None,
    interpret: bool = False,
    hierarchical: Union[bool, int] = True,
    sequential: Optional[bool] = None,
    collisions: bool = True,
    cost: str = "joint",
    motion: str = "joint",
    defer: bool = False,
    debug: bool = False,
    profile: bool = False,
    max_time: float = 60.0,
    **kwargs,
) -> Optional[Commands]:
    start_time = current_time()
    world = state.world
    if sequential is None:
        sequential = not world.multi_arm
    problem = create_problem(
        state, goal, collisions=collisions, cost=cost, motion=motion, defer=defer
    )
    if sequential:
        problem = problem.instantaneous()

    world.set_open_positions()

    if prompt is not None:
        print(SEPARATOR)
        llm_problem = problem.clone(
            initial=problem.initial.new_state(create_placeable(world, stackable=True))
        )
        context = world.description()
        if interpret:
            problem.goal &= interpret_llm(llm_problem, prompt, context)
        else:
            partial_plan = solve_llm(llm_problem, prompt, context)
            problem.add_partial_plan(partial_plan)
        problem.initial = problem.initial.new_state(initial_from_goal(world, problem.goal))

    if debug:
        problem = problem.lazy_clone(parent=False)

    hierarchical_functions = []
    hierarchical_actions = []
    if hierarchical:
        hierarchical_actions.append(problem.get_action("move"))
        if problem.has_function("ArmArmCollision"):
            hierarchical_functions.append(problem.get_function("ArmArmCollision"))
    with profiler(field="tottime" if profile else None, num=25):
        solutions = solve_hierarchical(
            problem,
            hierarchical_functions,
            hierarchical_actions,
            makespan_weight=0,
            sequential=sequential,
            successor_fn="offline",
            max_time=max_time,
            **kwargs,
        )
    if not solutions:
        return None

    print(SEPARATOR)
    solution = solutions[0]
    solution.dump()
    if debug:
        return Commands(world, commands=[])

    plan = solution.filtered_plan
    if len(world.arms) <= 1:
        return extract_sequential_commands(state, plan)
    if sequential:
        return extract_timed_commands(state, plan)
    with state:
        return postprocess_plan(state, plan, collisions=collisions)
