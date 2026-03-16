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
from typing import Any

# NVIDIA
from schedulestream.algorithm.finite.online import disable_lazy
from schedulestream.algorithm.schedule import schedule
from schedulestream.algorithm.temporal import TemporalSolution
from schedulestream.language.problem import InstantiatedProblem, Problem


def eagerly_evaluate(instantiated: InstantiatedProblem) -> None:
    for term in instantiated.terms:
        if term.is_static:
            term.evaluate()


def solve_eager(problem: Problem, **kwargs: Any) -> TemporalSolution:
    assert not problem.streams
    disable_lazy(problem)
    instantiated = problem.instantiate()
    eagerly_evaluate(instantiated)
    return schedule(instantiated, **kwargs)
