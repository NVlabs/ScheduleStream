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
from schedulestream.algorithm.finite.eager import solve_eager
from schedulestream.algorithm.finite.lazy import solve_lazy
from schedulestream.algorithm.finite.online import solve_online
from schedulestream.algorithm.temporal import TemporalSolution
from schedulestream.language.problem import Problem

FINITE_ALGORITHMS = {
    "lazy": solve_lazy,
    "online": solve_online,
    "eager": solve_eager,
}


def solve_finite(problem: Problem, algorithm: str = "lazy", **kwargs: Any) -> TemporalSolution:
    if algorithm not in FINITE_ALGORITHMS:
        raise ValueError(
            f"Invalid algorithm: {algorithm}. Valid algorithms: {list(FINITE_ALGORITHMS)}"
        )
    return FINITE_ALGORITHMS[algorithm](problem, **kwargs)
