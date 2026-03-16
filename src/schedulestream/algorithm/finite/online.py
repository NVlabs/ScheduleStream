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
from schedulestream.algorithm.schedule import schedule
from schedulestream.algorithm.temporal import TemporalSolution
from schedulestream.language.problem import Problem


def disable_lazy(problem: Problem) -> None:
    for function in problem.functions:
        function.lazy = False


def solve_online(problem: Problem, **kwargs: Any) -> TemporalSolution:
    disable_lazy(problem)
    instantiated = problem.instantiate()
    return schedule(instantiated, **kwargs)
