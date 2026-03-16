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

# NVIDIA
from schedulestream.algorithm.solver import solve
from schedulestream.common.utils import SEPARATOR
from schedulestream.language.action import Action
from schedulestream.language.anonymous import rename_anonymous
from schedulestream.language.argument import OUTPUT
from schedulestream.language.durative import DurativeAction
from schedulestream.language.function import NumericFunction
from schedulestream.language.predicate import Function, Predicate
from schedulestream.language.problem import Problem

# See: https://arxiv.org/abs/2511.04758

# Declare static robot predicates
Arm = Predicate("?arm")
Conf = Predicate("?arm ?q")

# Declare fluent current configuration function
At = Function("?arm", condition=[Arm("?arm"), Conf("?arm", OUTPUT)])

# Declare static object predicates
Object = Predicate("?obj")
Placement = Predicate("?obj ?p")

# Declare fluent object current state functions and predicates
Pose = Function("?obj", condition=[Object("?obj")])
Holding = Function("?arm", condition=[Arm("?arm")])
Attached = Function("?obj", condition=[Object("?obj")])

# Declare static action/stream parameter predicates
Grasp = Predicate("?arm ?obj ?g", condition=[Arm("?arm"), Object("?obj")])
Kin = Predicate(
    "?arm ?q ?obj ?g ?p",
    condition=[Conf("?arm ?q"), Placement("?obj ?p"), Grasp("?arm ?obj ?g")],
)
Motion = Predicate("?arm ?q1 ?t ?q2", condition=[Conf("?arm ?q1"), Conf("?arm ?q2")])

# Declare static procedural predicates evaluated using Python
Duration = NumericFunction("?t", definition=lambda t: 1e-2)
ObjCollision = Predicate("?t ?p", definition=lambda *inputs: False)
ArmCollision = Predicate("?t1 ?t2", definition=lambda *inputs: False)

# Robot constants
arm1 = "arm1"
arm2 = "arm2"
q1 = [0.0, 1.0]  # [x, y] position
q2 = [1.0, 1.0]  # [x, y] position

# Object constants
obj1 = "obj1"
obj2 = "obj2"
p1 = [0.0, 0.0]  # [x, y] position
p2 = [1.0, 0.0]  # [x, y] position

# The initial state
initial = [
    At(arm1) <= q1,
    At(arm2) <= q2,
    Holding(arm1) <= None,
    Holding(arm2) <= None,
    Pose(obj1) <= p1,
    Pose(obj2) <= p2,
    Placement(obj1, p1),
    Placement(obj2, p2),
    Attached(obj1) <= None,
    Attached(obj2) <= None,
]

# The goal conditions
goal = [Holding(arm1) == obj1, Holding(arm2) == obj2]

# Nested collision conditions
arm_obj_collisions = [~ObjCollision("?t", Pose(o)) for o in [obj1, obj2]]
arm_arm_collisions = [~ArmCollision("?t", At(a)) for a in [arm1, arm2]]

# Declare pick action
pick = Action(
    # robot arm ?arm, configuration ?q, object ?obj, grasp ?g, placement ?p
    parameters="?arm ?q ?obj ?g ?p",
    precondition=[
        Kin("?arm ?q ?obj ?g ?p"),  # kinematic constraint FK(?q) * ?g = ?p
        At("?arm") == "?q",  # ?arm is currently at configuration ?q
        Pose("?obj") == "?p",  # ?obj is currently at placement ?p
        Attached("?obj") == None,  # ?obj is not attached to anything
        Holding("?arm") == None,  # ?arm is not holding anything
    ],
    effect=[
        Pose("?obj") <= "?g",  # ?obj is now at grasp ?g
        Attached("?obj") <= "?arm",  # ?obj is now attached to ?arm
        Holding("?arm") <= "?obj",  # ?arm is now holding ?obj
    ],
)

# Declare move action
move = DurativeAction(
    # robot arm ?arm, start conf ?q1, trajectory ?t, end conf ?q2
    parameters="?arm ?q1 ?t ?q2",
    start_condition=[
        Motion("?arm ?q1 ?t ?q2"),  # valid motion constraint
        At("?arm") == "?q1",  # ?arm is currently at start conf ?q1
    ],
    start_effect=[At("?arm") <= "?t"],  # ?arm is now along trajectory ?t
    over_condition=arm_obj_collisions + arm_arm_collisions,
    end_condition=[],
    end_effect=[At("?arm") <= "?q2"],  # ?arm is now at end conf ?q2
    min_duration=Duration("?t"),  # Duration of trajectory ?t elapses
)

# Sets the name of each function or action as its Python variable name
rename_anonymous(locals())

# Derive a dummy stream from each of the following static predicates
streams = [
    Grasp.stream(inputs="?arm ?obj", conditional_generator=None),
    Kin.stream(inputs="?arm ?obj ?g ?p", conditional_generator=None),
    Motion.stream(inputs="?arm ?q1 ?q2", conditional_generator=None),
]

# Create a problem instance
problem = Problem(
    initial=initial,
    goal=goal,
    actions=[move, pick],
    streams=streams,
)

# Solve the problem instance
solutions = solve(
    problem,
    lazy=False,
    heuristic_fn="hmax",
    successor_fn="offline",
    weight=1,
    verbose=False,
)
print(SEPARATOR)

# Print the solution
if solutions:
    solutions[0].dump()
