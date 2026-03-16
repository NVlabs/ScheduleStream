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
from schedulestream.algorithm.stream.common import StreamSolution
from schedulestream.algorithm.stream.focused import solve_focused
from schedulestream.algorithm.stream.incremental import solve_incremental
from schedulestream.language.problem import Problem

STREAM_ALGORITHMS = {
    "focused": solve_focused,
    "incremental": solve_incremental,
}


def solve_stream(
    problem: Problem, algorithm: str = "focused", **kwargs: Any
) -> List[StreamSolution]:
    if algorithm not in STREAM_ALGORITHMS:
        raise ValueError(
            f"Invalid algorithm: {algorithm}. Valid algorithms: {list(STREAM_ALGORITHMS)}"
        )
    return STREAM_ALGORITHMS[algorithm](problem, **kwargs)
