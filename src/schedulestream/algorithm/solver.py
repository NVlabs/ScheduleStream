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
from typing import Any, List

# NVIDIA
from schedulestream.algorithm.finite.solver import solve_finite
from schedulestream.algorithm.stream.solver import solve_stream
from schedulestream.algorithm.utils import Solution
from schedulestream.language.problem import Problem


def solve(problem: Problem, **kwargs: Any) -> List[Solution]:
    if problem.is_stream:
        return solve_stream(problem, **kwargs)
    solution = solve_finite(problem, **kwargs)
    if solution is None:
        return []
    return [solution]
