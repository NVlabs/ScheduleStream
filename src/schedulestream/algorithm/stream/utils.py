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
from typing import Any, Dict, Optional

# NVIDIA
from schedulestream.algorithm.schedule import schedule_solution
from schedulestream.algorithm.stream.common import StreamSolution
from schedulestream.algorithm.stream.stream_plan import StreamPlan, extract_stream_plan
from schedulestream.algorithm.temporal import retime_plan
from schedulestream.algorithm.utils import bind_plan
from schedulestream.common.utils import safe_zip
from schedulestream.language.argument import Constant
from schedulestream.language.lazy import LazyOutput
from schedulestream.language.problem import Problem
from schedulestream.language.stream import StreamOutput


def rebind_stream_plan(stream_plan: StreamPlan) -> Dict[Constant, Constant]:
    mapping = {}
    new_stream_plan = []
    for stream in stream_plan:
        stream = stream.bind(mapping)
        new_stream_plan.append(stream)
        if isinstance(stream, StreamOutput):
            outputs = LazyOutput.from_instance(stream.stream_instance)
            mapping.update(safe_zip(stream.outputs, outputs))
    return mapping


def retrace_solution(solution: StreamSolution) -> StreamSolution:
    if solution.plan is None:
        return solution
    stream_plan = [stream.root for stream in solution.stream_plan]
    mapping = rebind_stream_plan(stream_plan)
    stream_plan = [stream.root for stream in bind_plan(stream_plan, mapping)]
    plan = retime_plan(bind_plan(solution.plan, mapping))
    return StreamSolution(plan=plan, stream_plan=stream_plan)


def complete_solution(
    problem: Problem, solution: StreamSolution, **kwargs: Any
) -> Optional[StreamSolution]:
    plan = schedule_solution(problem, solution)
    if plan is None:
        return solution
    streams_from_evaluation = dict(solution.streams_from_atoms)
    streams_from_evaluation.update({evaluation: None for evaluation in problem.initial})
    initial = problem.initial.new_state(streams_from_evaluation)
    stream_plan = extract_stream_plan(
        initial,
        plan,
        streams_from_evaluation,
        **kwargs,
    )
    return StreamSolution(plan=plan, stream_plan=stream_plan)
