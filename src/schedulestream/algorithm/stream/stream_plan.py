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
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# NVIDIA
from schedulestream.algorithm.temporal import sequential_from_timed
from schedulestream.algorithm.utils import Plan
from schedulestream.common.graph import (
    get_ancestors,
    get_incoming_from_vertex,
    kahn,
    kahn_layers,
    kahn_topological_sort,
    visualize_graph,
)
from schedulestream.common.ordered_set import OrderedSet
from schedulestream.common.utils import INF, partition, remove_duplicates
from schedulestream.language.argument import Constant
from schedulestream.language.constraint import Constraint, Cost
from schedulestream.language.function import Evaluation
from schedulestream.language.state import State
from schedulestream.language.stream import StreamOutput

StreamPlan = List[StreamOutput]


def get_stream_plan_orders(
    stream_plan: StreamPlan, evaluations: Optional[List[Evaluation]] = None
) -> List[Tuple[StreamOutput, StreamOutput]]:
    reached_evaluations = set(evaluations or [])
    stream_orders = []
    for i, stream1 in enumerate(stream_plan):
        new_evaluations = set(stream1.output_condition.clause) - reached_evaluations
        for stream2 in stream_plan[i + 1 :]:
            if new_evaluations & set(stream2.input_condition.clause):
                stream_orders.append((stream1, stream2))
        reached_evaluations.update(new_evaluations)
    return stream_orders


def visualize_stream_plan(stream_plan: StreamPlan, **kwargs: Any) -> str:
    stream_colors = {}
    for stream in stream_plan:
        if isinstance(stream, Constraint):
            stream_colors[stream] = "LightSalmon"
        elif isinstance(stream, Cost):
            stream_colors[stream] = "LightBlue"
        else:
            stream_colors[stream] = "LightYellow"
    stream_orders = get_stream_plan_orders(stream_plan)
    return visualize_graph(stream_orders, stream_plan, vertex_colors=stream_colors, **kwargs)


def compute_preimage(initial: State, plan: Plan) -> Dict[Cost, int]:
    evaluation_steps = defaultdict(int)
    state = initial
    preimage = OrderedSet()
    for step, action in enumerate(plan):
        for precondition in action.precondition.clause:
            static = precondition.is_static
            for condition in precondition.support(state):
                if evaluation_steps[condition] == 0:
                    constraint = Constraint(condition, static=static)
                    if constraint not in preimage:
                        preimage[constraint] = step
        for cost in action.cost.support(state):
            if evaluation_steps[cost] == 0:
                cost = Cost(cost.term, static=cost.is_static)
                if cost not in preimage:
                    preimage[cost] = step

        evaluations = action.effect.apply(state)
        assert evaluations is not None
        for evaluation in evaluations:
            evaluation_steps[evaluation] = step + 1
        state = state.new_state(evaluations)
    return preimage


def get_evaluation_layers(
    streams_from_evaluation: Dict[Evaluation, Optional[List[StreamOutput]]],
    evaluations: List[Evaluation],
) -> Dict[Evaluation, int]:
    stream_plan = retrace_stream_plan(streams_from_evaluation, evaluations)
    orders = get_stream_plan_orders(stream_plan)
    evaluation_layers = {}
    for i, streams in enumerate(kahn(orders, stream_plan)):
        for stream in streams:
            for evaluation in stream.output_condition.clause:
                if evaluation not in evaluation_layers:
                    evaluation_layers[evaluation] = i + 1
    return evaluation_layers


def retrace_stream_plan(
    streams_from_evaluation: Dict[Evaluation, Optional[List[StreamOutput]]],
    evaluations: List[Evaluation],
    achieved_evaluations: Optional[Set[Evaluation]] = None,
) -> StreamPlan:
    stream_plan = []
    achieved_evaluations = set(achieved_evaluations or set())
    evaluations = filter(lambda e: streams_from_evaluation.get(e, None) is not None, evaluations)
    evaluations = sorted(evaluations, key=lambda e: len(streams_from_evaluation[e]))
    for evaluation in evaluations:
        if evaluation in achieved_evaluations:
            continue
        parent_stream = streams_from_evaluation.get(evaluation, None)
        if parent_stream is not None:
            parent_stream = streams_from_evaluation[evaluation][0]
            parent_evaluations = parent_stream.input_condition.clause
            parent_plan = retrace_stream_plan(
                streams_from_evaluation, parent_evaluations, achieved_evaluations
            ) + [parent_stream]
            for stream in parent_plan:
                achieved_evaluations.update(stream.output_condition.clause)
            stream_plan.extend(parent_plan)
    return stream_plan


def layer_retrace_stream_plan(
    streams_from_evaluation: Dict[Evaluation, Optional[List[StreamOutput]]],
    evaluations: List[Evaluation],
) -> StreamPlan:
    stream_plan = []
    if not evaluations:
        return stream_plan
    evaluation_layers = get_evaluation_layers(streams_from_evaluation, evaluations)
    assert evaluation_layers
    active_evaluations = OrderedSet(evaluations)
    max_layer = max(evaluation_layers.values())
    for layer in reversed(range(1, max_layer + 1)):
        layer_evaluations = OrderedSet(
            e for e in active_evaluations if evaluation_layers[e] == layer
        )
        layer_streams = remove_duplicates(
            streams_from_evaluation[e][0] for e in layer_evaluations if streams_from_evaluation[e]
        )
        while layer_evaluations:
            stream = max(
                layer_streams,
                key=lambda s: len(layer_evaluations.intersection(s.output_condition.clause)),
            )
            stream_plan.insert(0, stream)
            for evaluation in stream.output_condition.clause:
                layer_evaluations.discard(evaluation)
            for evaluation in stream.input_condition.clause:
                if evaluation in evaluation_layers:
                    active_evaluations.add(evaluation)
    return stream_plan


def reorder_stream_plan(
    stream_plan: StreamPlan,
    stream_steps: Optional[Dict[StreamOutput, int]] = None,
    eager_constraints: bool = False,
) -> StreamPlan:
    if stream_steps is None:
        stream_steps = {}
    stream_orders = get_stream_plan_orders(stream_plan)

    incoming_from_vertex = get_incoming_from_vertex(stream_orders)
    if eager_constraints:
        key = lambda s: (
            isinstance(s, StreamOutput),
            isinstance(s, Constraint),
            len(incoming_from_vertex[s]),
        )
    else:
        key = lambda s: (
            isinstance(s, Constraint),
            isinstance(s, StreamOutput),
            -s.priority,
            stream_steps.get(s, INF),
            len(incoming_from_vertex[s]),
        )
    stream_plan = sorted(stream_plan, key=key)
    stream_plan = kahn_topological_sort(stream_orders, stream_plan, greedy=True)
    return stream_plan


def get_streams_from_output(
    stream_plan: StreamPlan, verbose: bool = True
) -> Dict[Constant, List[StreamOutput]]:
    streams_from_output = {}
    for stream in stream_plan:
        for output in stream.outputs:
            streams_from_output.setdefault(output, []).append(stream)
    if verbose:
        for i, (output, streams) in enumerate(streams_from_output.items()):
            print(f"{i}/{len(streams_from_output)}) {output} | Streams ({len(streams)}): {streams}")
    return streams_from_output


def extract_stream_plan(
    initial: State,
    plan: Optional[Plan],
    streams_from_evaluation: Dict[Evaluation, Optional[List[StreamOutput]]],
    over_conditions: bool = True,
    visualize: bool = False,
    verbose: bool = False,
) -> Optional[StreamPlan]:
    if plan is None:
        return None
    plan = sequential_from_timed(plan, over=over_conditions)
    preimage_steps = compute_preimage(initial, plan)
    function_costs, stream_constraints = partition(lambda c: c.is_procedural, preimage_steps)
    evaluation_steps = {
        constraint.expression: preimage_steps[constraint] for constraint in stream_constraints
    }

    stream_evaluations = [
        evaluation
        for evaluation in evaluation_steps
        if streams_from_evaluation.get(evaluation, None) is not None
    ]
    if verbose:
        if stream_evaluations:
            print(f"Streams: {len(stream_evaluations)}")
        for i, evaluation in enumerate(stream_evaluations):
            print(
                f"{i}/{len(stream_evaluations)}) {evaluation} | Step:"
                f" {evaluation_steps[evaluation]} | Streams"
                f" ({len(streams_from_evaluation[evaluation])}):"
                f" {streams_from_evaluation[evaluation]}"
            )
        if function_costs:
            print(f"Functions: {len(function_costs)}")
        for i, cost in enumerate(function_costs):
            step = preimage_steps[cost]
            print(f"{i}/{len(function_costs)}) {cost} | Step: {step}")

    stream_plan = layer_retrace_stream_plan(streams_from_evaluation, stream_evaluations)
    stream_orders = get_stream_plan_orders(stream_plan)

    stream_steps = {}
    for evaluation, step in evaluation_steps.items():
        if streams_from_evaluation.get(evaluation, None) is None:
            continue
        for stream in streams_from_evaluation[evaluation]:
            if stream in stream_plan:
                stream_steps[stream] = min(step, stream_steps.get(stream, step))
    for stream, step in list(stream_steps.items()):
        for ancestor_stream in get_ancestors(stream_orders, source_vertices=[stream]):
            stream_steps[ancestor_stream] = min(step, stream_steps.get(ancestor_stream, step))

    procedure_plan = stream_plan + function_costs
    if visualize:
        visualize_stream_plan(procedure_plan)
    procedure_plan = reorder_stream_plan(procedure_plan, stream_steps)
    return procedure_plan
