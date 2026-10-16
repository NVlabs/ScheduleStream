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
import math
import random
from collections import defaultdict, deque
from functools import cached_property
from typing import Any, Callable, Deque, Dict, FrozenSet, List, Optional, Sized, Tuple

# NVIDIA
from schedulestream.algorithm.finite.lazy import partition_costs
from schedulestream.algorithm.stream.common import StreamSolution, partition_stream_plan
from schedulestream.algorithm.stream.stream_plan import (
    StreamPlan,
    get_stream_plan_orders,
    reorder_stream_plan,
    visualize_stream_plan,
)
from schedulestream.algorithm.temporal import get_makespan, retime_plan
from schedulestream.algorithm.utils import Plan, bind_plan
from schedulestream.common.graph import (
    get_incoming_from_vertex,
    get_outgoing_from_vertex,
    get_reachable,
    reverse_edges,
    topological_sort,
)
from schedulestream.common.queue import StablePriorityQueue
from schedulestream.common.utils import (
    INF,
    current_time,
    elapsed_time,
    flatten,
    get_length,
    implies,
    instance_cache,
)
from schedulestream.language.argument import Constant
from schedulestream.language.constraint import Constraint, Cost
from schedulestream.language.problem import Problem
from schedulestream.language.stream import (
    BatchPredicateStream,
    Context,
    StreamInstance,
    StreamOutput,
)


class Skeleton:
    def __init__(
        self, satisfier: "Satisfier", stream_plan: StreamPlan, plan: Optional[Plan] = None
    ):
        self.satisfier = satisfier
        self.num = len(self.satisfier.skeletons)

        incoming_from_stream = get_incoming_from_vertex(get_stream_plan_orders(stream_plan))
        streams, constraints, costs = partition_stream_plan(stream_plan)
        self.stream_plan = streams
        stream_indices = {stream: index for index, stream in enumerate(self.stream_plan)}
        self.costs_from_index = defaultdict(list)
        for cost in constraints + costs:
            index = 0
            if incoming_from_stream[cost]:
                index = max(map(stream_indices.get, incoming_from_stream[cost])) + 1
            self.costs_from_index[index].append(cost)

        self.plan = plan
        self.siblings = []
        self.active_indices = set()
        self.num_bindings = (self.num_streams + 1) * [0]
        self.num_calls = (self.num_streams + 1) * [0]
        self.iteration_calls = (self.num_streams + 1) * [0]
        self.num_queued = (self.num_streams + 1) * [0]
        self.num_generating = (self.num_streams + 1) * [0]
        self.generating_ancestors = (self.num_streams + 1) * [True]
        self.batches = {
            index: {}
            for index, stream_output in enumerate(self.stream_plan)
            if isinstance(stream_output, StreamOutput)
            and isinstance(stream_output.stream, BatchPredicateStream)
        }
        self.dequeued = []
        self.solution_bindings = []
        self.best_binding = None
        self.root_binding = Binding(self)
        self.add_binding(self.root_binding)

    @cached_property
    def num_streams(self) -> int:
        return len(self.stream_plan)

    @cached_property
    def streams(self) -> FrozenSet[StreamOutput]:
        return frozenset(stream for stream in self.stream_plan if isinstance(stream, StreamOutput))

    @cached_property
    def parameters(self) -> FrozenSet[Constant]:
        return frozenset(flatten(stream.outputs for stream in self.streams))

    @property
    def num_solutions(self) -> int:
        return len(self.solution_bindings)

    def sorted_solution_bindings(self) -> List["Binding"]:
        return sorted(self.solution_bindings, key=lambda b: b.metric)

    @cached_property
    def stream_orders(self) -> List[Tuple[StreamOutput, StreamOutput]]:
        return get_stream_plan_orders(self.stream_plan)

    @cached_property
    def index_from_stream(self) -> Dict[StreamOutput, int]:
        return {stream: i for i, stream in enumerate(self.stream_plan)}

    @cached_property
    def bound_index(self) -> int:
        stream_indices = [
            index for index, stream in enumerate(self.streams) if isinstance(stream, StreamOutput)
        ]
        if not stream_indices:
            return 0
        return max(stream_indices) + 1

    @property
    def satisfied(self) -> bool:
        return (self.best_binding is not None) and self.best_binding.satisfied

    @property
    def processed(self) -> int:
        if self.best_binding is None:
            return 0
        return self.best_binding.index

    @property
    def remaining(self) -> int:
        return len(self.stream_plan) - self.processed

    @property
    def min_cost(self) -> float:
        if not self.satisfied:
            return INF
        return self.best_binding.cost

    @property
    def min_makespan(self) -> float:
        if not self.satisfied:
            return INF
        return self.best_binding.makespan

    @property
    def min_metric(self) -> float:
        if not self.satisfied:
            return INF
        return self.best_binding.metric

    def add_binding(self, binding: "Binding") -> None:
        is_better = binding < self.best_binding
        if binding.satisfied and implies(self.satisfier.prune, is_better):
            self.solution_bindings.append(binding)
        if is_better:
            if (self.best_binding is None) or (binding.index > self.best_binding.index):
                ancestors = self.get_active(binding.skeleton_output) + [binding.skeleton_output]
                self.active_indices.clear()
                self.active_indices.update(map(self.index_from_stream.get, ancestors))
                for binding in self.dequeued:
                    self.satisfier.push_binding(binding)
                self.dequeued.clear()
                if isinstance(binding.parent_output, StreamOutput):
                    self.satisfier.improved = True
            self.best_binding = binding

    @instance_cache
    def get_parents(self, stream: StreamOutput) -> List[StreamOutput]:
        incoming_from_vertex = get_incoming_from_vertex(self.stream_orders)
        return list(incoming_from_vertex[stream])

    @instance_cache
    def get_children(self, stream: StreamOutput) -> List[StreamOutput]:
        outgoing_from_vertex = get_outgoing_from_vertex(self.stream_orders)
        return list(outgoing_from_vertex[stream])

    @instance_cache
    def get_ancestors(self, stream: StreamOutput) -> List[StreamOutput]:
        if stream is None:
            return self.stream_plan
        index = self.index_from_stream[stream]
        stream_plan = self.stream_plan[: index + 1]
        stream_orders = reverse_edges(get_stream_plan_orders(stream_plan))
        ancestors = topological_sort(stream_orders, vertices=[stream], use_dfs=True)[1:]
        return ancestors[::-1]

    @instance_cache
    def get_descendants(self, stream: StreamOutput) -> List[StreamOutput]:
        return topological_sort(self.stream_orders, vertices=[stream], use_dfs=True)[1:]

    @instance_cache
    def get_connected(self, stream: StreamOutput) -> List[StreamOutput]:
        index = self.index_from_stream[stream]
        stream_plan = self.stream_plan[: index + 1]
        stream_orders = get_stream_plan_orders(stream_plan)
        connected = get_reachable(stream_orders, source_vertices=[stream])[1:]
        return connected[::-1]

    def get_active(self, stream: StreamOutput) -> List[StreamOutput]:
        return self.get_ancestors(stream)

    def is_generating(self, stream: StreamOutput) -> bool:
        if stream is None:
            return False
        index = self.index_from_stream[stream]
        return self.num_generating[index] != 0

    def ancestors_generating(self, stream: StreamOutput) -> bool:
        return any(map(self.is_generating, self.get_ancestors(stream)))

    def propagate_exhausted(self, stream: StreamOutput) -> None:
        if self.is_generating(stream):
            return
        for child_stream in self.get_children(stream):
            if not self.ancestors_generating(child_stream):
                child_index = self.index_from_stream[child_stream]
                self.process_batch(child_index)
                self.propagate_exhausted(child_stream)

    def evaluate_cost(
        self, cost: Cost, mappings: List[Dict[Constant, Constant]], verbose: bool = False
    ) -> List[float]:
        """Evaluates a batched cost under each mapping (INF if a constraint fails)."""
        start_time = current_time()
        bound_costs = [cost.bind(mapping) for mapping in mappings]
        num_new = sum(not c.is_evaluated() for c in bound_costs)
        cost.evaluate_batch(bound_costs)
        values = [c.evaluate() for c in bound_costs]
        if isinstance(cost, Constraint):
            values = [0.0 if value else INF for value in values]
            breakdown = f"Satisfied: {sum(value < INF for value in values)}/{len(values)}"
        else:
            values = [float(value) for value in values]
            breakdown = f"Values: [{min(values, default=INF):.3f}, {max(values, default=INF):.3f}]"
        if verbose:
            print(
                f"{cost} | Evaluated: {num_new}/{len(bound_costs)} | {breakdown} |"
                f" Elapsed: {elapsed_time(start_time):.3f} sec"
            )
        return values

    def evaluate_costs(
        self, costs: List[Cost], mappings: List[Dict[Constant, Constant]], verbose: bool = False
    ) -> List[float]:
        totals = [0.0] * len(mappings)
        for cost in costs:
            values = self.evaluate_cost(cost, mappings, verbose)
            totals = [total + value for total, value in zip(totals, values)]
        return totals

    def evaluate_batch_costs(self, bindings: List["Binding"], verbose: bool = False) -> None:
        start_time = current_time()
        bindings = [binding for binding in bindings if not binding.costs_evaluated]
        if not bindings:
            return
        for binding in bindings:
            binding.costs_evaluated = True
        [index] = {binding.index for binding in bindings}
        batched_costs = [
            cost for cost in self.costs_from_index[index] if cost.batch_size is not None
        ]
        if not batched_costs:
            return
        costs = self.evaluate_costs(
            batched_costs, [binding.mapping for binding in bindings], verbose
        )
        if verbose:
            print(
                f"Cost: {index - 1}/{len(self.stream_plan)} | Valid:"
                f" {sum(cost < INF for cost in costs)}/{len(bindings)} | Costs: [{min(costs):.3f},"
                f" {max(costs):.3f}] | Cost terms: {batched_costs} | Elapsed:"
                f" {elapsed_time(start_time):.3f} sec"
            )

    def sort_bindings(
        self,
        bindings: List["Binding"],
        verbose: bool = False,
    ) -> List["Binding"]:
        start_time = current_time()
        bindings = list(bindings)
        if not bindings:
            return bindings
        self.evaluate_batch_costs(bindings)
        if self.satisfier.randomize:
            self.satisfier.random.shuffle(bindings)
        bindings.sort(key=lambda b: (b.iteration, b.invalid, b.cost))
        if self.satisfier.sorted_fraction < 1:
            num_sorted = math.ceil(self.satisfier.sorted_fraction * len(bindings))
            rest = bindings[num_sorted:]
            self.satisfier.random.shuffle(rest)
            bindings[num_sorted:] = rest
        [index] = {binding.index for binding in bindings}
        if verbose and self.costs_from_index[index]:
            costs = [INF if binding.invalid else binding.cost for binding in bindings]
            num_valid = sum(not binding.invalid for binding in bindings)
            print(
                f"Sort: {index-1}/{len(self.stream_plan)} | Valid: {num_valid}/{len(bindings)} |"
                f" Costs: [{min(costs):.3f}, {max(costs):.3f}] | Cost terms:"
                f" {self.costs_from_index[index]} | Elapsed: {elapsed_time(start_time):.3f} sec"
            )
        return bindings

    def introduce_bindings(self, new_bindings: List["Binding"]) -> List["Binding"]:
        new_bindings = self.sort_bindings(new_bindings)
        if not new_bindings:
            return new_bindings

        greedy_outputs = len(new_bindings)
        [index] = set(new_binding.index - 1 for new_binding in new_bindings)
        stream = self.stream_plan[index].stream
        if stream.greedy_outputs is not None:
            greedy_outputs = min(stream.greedy_outputs, greedy_outputs)
        for i, new_binding in enumerate(new_bindings):
            self.add_binding(new_binding)
            if i < greedy_outputs:
                self.satisfier.introduce_binding(new_binding)
            else:
                self.satisfier.push_binding(new_binding)

    def process_batch(self, index: int) -> None:
        if index not in self.batches:
            return
        instance_bindings = dict(self.batches[index])
        if not instance_bindings:
            return
        self.batches[index].clear()
        stream = self.stream_plan[index].stream
        stream_instances = list(instance_bindings)
        bindings = list(flatten(instance_bindings.values()))
        stream.get_batch(stream_instances)
        self.iteration_calls[index] += 1
        self.num_calls[index] += 1
        new_bindings = []
        for binding in bindings:
            new_bindings.extend(binding.next_bindings())
        self.introduce_bindings(new_bindings)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(streams={self.num_streams},"
            f" actions={get_length(self.plan)}, satisfied={self.satisfied},"
            f" makespan={self.min_makespan}, cost={self.min_cost},"
            f" solutions={len(self.solution_bindings)})"
        )

    __repr__ = __str__


class Binding:
    def __init__(
        self,
        skeleton: Skeleton,
        parent_binding: Optional["Binding"] = None,
        parent_output: Optional[StreamOutput] = None,
    ):
        self.skeleton = skeleton
        self.parent_binding = parent_binding
        self.parent_output = parent_output
        self.rescheduled = False
        self.costs_evaluated = False
        self.queued = False
        self.num = self.skeleton.num_bindings[self.index]
        self.skeleton.num_bindings[self.index] += 1
        self.skeleton.num_generating[self.index] += 1
        self.iteration = 0
        self.queued_outputs: Optional[Deque[StreamOutput]] = None
        self.discovery_time = elapsed_time(self.satisfier.start_time)

    @cached_property
    def total_num(self) -> int:
        parent_num = 0 if self.parent_binding is None else self.parent_binding.total_num
        return parent_num + self.num

    @property
    def satisfier(self) -> "Satisfier":
        return self.skeleton.satisfier

    @property
    def is_batcher(self) -> bool:
        return self.index in self.skeleton.batches

    @property
    def has_context(self) -> bool:
        return (
            isinstance(self.stream_output, StreamOutput) and self.stream_output.stream.has_context
        )

    @cached_property
    def index(self) -> int:
        return 0 if self.parent_binding is None else self.parent_binding.index + 1

    @cached_property
    def complete(self) -> bool:
        return self.index == self.skeleton.num_streams

    @property
    def bound(self) -> bool:
        return self.index >= self.skeleton.bound_index

    @property
    def satisfied(self) -> bool:
        return self.complete and not self.invalid

    @property
    def active(self) -> bool:
        return self.index in self.skeleton.active_indices

    @cached_property
    def mapping(self) -> Dict[Constant, Constant]:
        if self.parent_binding is None:
            return {}
        assert self.parent_output is not None
        mapping = dict(self.parent_binding.mapping)
        mapping.update(
            zip(self.parent_binding.stream_output.outputs, self.parent_output.outputs, strict=True)
        )
        return mapping

    @cached_property
    def stream_plan(self) -> StreamPlan:
        if self.parent_output is None:
            return []
        head_plan = list(self.parent_binding.stream_plan)
        if not isinstance(self.parent_output, Cost):
            head_plan.append(self.parent_output)
        return head_plan

    @property
    def constraints(self) -> List[Constraint]:
        return [
            constraint
            for constraint in self.skeleton.costs_from_index[self.index]
            if isinstance(constraint, Constraint)
        ]

    @property
    def costs(self) -> List[Cost]:
        return [
            cost
            for cost in self.skeleton.costs_from_index[self.index]
            if not isinstance(cost, Constraint)
        ]

    @cached_property
    def plan(self) -> Optional[Plan]:
        if self.skeleton.plan is None:
            return self.skeleton.plan
        plan = bind_plan(self.skeleton.plan, self.mapping)
        return retime_plan(plan)

    @cached_property
    def makespan(self) -> float:
        if self.plan is None:
            return 0.0
        return get_makespan(self.plan)

    @cached_property
    def cost(self) -> float:
        total_cost = 0.0 if self.parent_binding is None else self.parent_binding.cost
        for cost in self.costs:
            total_cost += cost.bind(self.mapping).evaluate()
        return total_cost

    @cached_property
    def invalid(self) -> int:
        invalid = 0 if self.parent_binding is None else self.parent_binding.invalid
        for constraint in self.constraints:
            invalid += not constraint.bind(self.mapping).evaluate()
        return invalid

    @property
    def metric(self) -> float:
        if not self.complete:
            return self.cost
        if not self.satisfier.makespan_weight:
            return self.cost
        return self.satisfier.makespan_weight * self.makespan + self.cost

    @property
    def skeleton_output(self) -> Optional[StreamOutput]:
        if self.complete:
            return None
        return self.skeleton.stream_plan[self.index]

    @cached_property
    def stream_output(self) -> Optional[StreamOutput]:
        if self.skeleton_output is None:
            return None
        return self.skeleton_output.bind(self.mapping)

    @cached_property
    def stream_instance(self) -> Optional[StreamInstance]:
        if self.stream_output is None:
            return None
        if not isinstance(self.stream_output, StreamOutput):
            return self.stream_output
        return self.stream_output.stream_instance

    @cached_property
    def parent_plan(self) -> StreamPlan:
        if self.parent_binding is None:
            return []
        return self.parent_binding.parent_plan + [self.parent_output]

    @cached_property
    def context(self) -> Optional[Context]:
        if not self.has_context:
            return None
        costs = self.skeleton.costs_from_index[self.index + 1]
        costs = [
            cost for cost in costs if cost.function in self.stream_output.stream.context_functions
        ]
        if not costs:
            return None
        costs = [atom.bind(self.mapping) for atom in costs]
        constraints, costs = partition_costs(costs)
        return Context(self.stream_output.outputs, constraints=constraints, costs=costs)

    @property
    def dominated(self) -> bool:
        return self.invalid or (self.metric >= self.skeleton.min_metric)

    @property
    def aligned(self) -> bool:
        return self.iteration >= self.stream_instance.get_iterations(self.context)

    @property
    def exhausted(self) -> bool:
        return self.complete or (
            (self.queued_outputs is None)
            and self.stream_instance.is_exhausted(self.context)
            and self.aligned
        )

    def priority(self, metric: bool = False) -> Tuple:
        parts = [self.iteration, -self.index]
        if metric:
            parts.append(self.metric)
        if self.satisfier.randomize:
            parts.append(self.satisfier.random.random())
        return tuple(parts)

    def get_outputs(self, **kwargs: Any) -> List[StreamOutput]:
        return self.stream_instance.get_outputs(
            self.iteration,
            context=self.context,
            **kwargs,
        )

    def next_outputs(self, verbose: bool = False) -> List[StreamOutput]:
        stream_outputs = self.get_outputs()
        self.skeleton.num_calls[self.index] += 1
        if verbose and not stream_outputs and not self.is_batcher:
            self.dump()
        self.iteration += 1
        return stream_outputs

    def new_binding(self, stream_output: StreamOutput) -> "Binding":
        return Binding(self.skeleton, parent_binding=self, parent_output=stream_output)

    def next_bindings(self, **kwargs: Any) -> List["Binding"]:
        new_bindings = list(map(self.new_binding, self.next_outputs(**kwargs)))
        return new_bindings

    def next_binding(
        self, cost_fn: Optional[Callable[[StreamOutput], float]] = None
    ) -> Optional["Binding"]:
        if self.queued_outputs is None:
            outputs = list(self.get_outputs())
            if cost_fn is not None:
                outputs.sort(key=cost_fn)
            self.queued_outputs = deque(outputs)
        if not self.queued_outputs:
            self.queued_outputs = None
            self.iteration += 1
            return None
        return self.new_binding(self.queued_outputs.popleft())

    @cached_property
    def partial_solution(self) -> StreamSolution:
        return StreamSolution(
            stream_plan=self.stream_plan,
            plan=self.plan,
            cost=self.cost,
            discovery_time=self.discovery_time,
        )

    @cached_property
    def solution(self) -> StreamSolution:
        if not self.satisfied:
            return StreamSolution()
        return self.partial_solution

    def dump(self) -> None:
        if isinstance(self.stream_output, StreamOutput):
            if not self.stream_instance.exhausted or not self.stream_instance.num_outputs:
                print(
                    f"Stream: {self.stream_instance} | Exhausted:"
                    f" {self.stream_instance.exhausted} | Attempts:"
                    f" {self.stream_instance.iterations} | Outputs:"
                    f" {self.stream_instance.num_outputs}"
                )
        else:
            print(f"Constraint: {self.stream_output} | Satisfied: {self.stream_output.evaluate()}")

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(skeleton={self.skeleton.num},"
            f" index={self.index}/{self.skeleton.num_streams}, stream={self.stream_output},"
            f" num={self.num}, iteration={self.iteration})"
        )

    def __lt__(self, other: Optional["Binding"]) -> bool:
        if other is None:
            return True
        assert self.skeleton == other.skeleton
        if self.index > other.index:
            return True
        if self.index < other.index:
            return False
        if self.complete and other.complete:
            if self.invalid < other.invalid:
                return True
            if self.metric < other.metric:
                return True
        return False

    __repr__ = __str__


class Satisfier(Sized):
    skeleton_class = Skeleton

    def __init__(
        self,
        problem: Optional[Problem] = None,
        prune: bool = False,
        deactivate: bool = True,
        greedy: bool = True,
        randomize: bool = True,
        max_solutions: int = 1,
        makespan_weight: float = 1,
        sorted_fraction: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.start_time = current_time()
        self.problem = problem
        self.prune = prune
        self.deactivate = deactivate
        self.greedy = greedy
        self.randomize = randomize
        self.makespan_weight = makespan_weight
        self.sorted_fraction = sorted_fraction
        self.improved = True
        self.queue = StablePriorityQueue()
        self.skeletons = []
        self.cumulative_time = 0.0
        self.max_solutions = max_solutions
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.queue)

    @property
    def num_skeletons(self) -> int:
        return len(self.skeletons)

    @property
    def solution_bindings(self) -> List["Binding"]:
        return list(flatten(skeleton.solution_bindings for skeleton in self.skeletons))

    @property
    def num_solutions(self) -> int:
        return sum(skeleton.num_solutions for skeleton in self.skeletons)

    @property
    def sorted_solution_bindings(self) -> List["Binding"]:
        return sorted(self.solution_bindings, key=lambda b: b.metric)

    @property
    def sorted_solutions(self) -> List[StreamSolution]:
        return [binding.solution for binding in self.sorted_solution_bindings]

    @property
    def solved(self) -> bool:
        return any(skeleton.satisfied for skeleton in self.skeletons)

    @property
    def min_cost(self) -> float:
        if not self.skeletons:
            return INF
        return min(skeleton.min_cost for skeleton in self.skeletons)

    @property
    def min_makespan(self) -> float:
        if not self.skeletons:
            return INF
        return min(skeleton.min_makespan for skeleton in self.skeletons)

    @property
    def min_metric(self) -> float:
        if not self.skeletons:
            return INF
        return min(skeleton.min_metric for skeleton in self.skeletons)

    def pop_binding(self) -> Binding:
        binding = self.queue.pop()
        binding.queued = False
        binding.skeleton.num_queued[binding.index] -= 1
        assert binding.skeleton.num_queued[binding.index] >= 0
        return binding

    def push_binding(self, binding: Binding) -> None:
        if not binding.exhausted:
            self.queue.push(binding.priority(), binding)
            binding.queued = True
            binding.skeleton.num_queued[binding.index] += 1
        else:
            binding.skeleton.num_generating[binding.index] -= 1
            assert binding.skeleton.num_generating[binding.index] >= 0
            binding.skeleton.propagate_exhausted(binding.skeleton_output)

    def process_batcher(self, binding: Binding) -> bool:
        assert binding.is_batcher
        binding.skeleton.num_generating[binding.index] -= 1
        assert binding.skeleton.num_generating[binding.index] >= 0

        instance = binding.stream_instance
        binding.skeleton.batches[binding.index].setdefault(instance, []).append(binding)
        batch_size = binding.stream_output.stream.batch_size
        if len(binding.skeleton.batches[binding.index]) >= batch_size:
            binding.skeleton.process_batch(binding.index)
        return True

    def process_binding(self, binding: Binding) -> None:
        if binding.complete:
            return
        if binding.is_batcher:
            self.process_batcher(binding)
            return

        new_bindings = binding.next_bindings()
        binding.skeleton.introduce_bindings(new_bindings)
        self.push_binding(binding)

    def introduce_binding(self, binding: Binding) -> None:
        if (binding.iteration == 0) and not self.is_full():
            self.process_binding(binding)
        else:
            self.push_binding(binding)

    def process_skeleton(self, skeleton: Skeleton) -> None:
        self.process_binding(skeleton.root_binding)

    def add_skeleton(
        self, stream_plan: StreamPlan, plan: Optional[Plan] = None
    ) -> Optional[Skeleton]:
        for other_skeleton in self.skeletons:
            if plan == other_skeleton.plan:
                return None
        skeleton = self.skeleton_class(self, stream_plan, plan)
        for other_skeleton in self.skeletons:
            if other_skeleton.streams <= skeleton.streams:
                skeleton.siblings.append(other_skeleton)
            if skeleton.streams <= other_skeleton.streams:
                other_skeleton.siblings.append(skeleton)
        self.skeletons.append(skeleton)
        self.process_skeleton(skeleton)
        return skeleton

    def add_solution(self, solution: StreamSolution) -> Optional[Skeleton]:
        if not solution.success:
            return None
        return self.add_skeleton(solution.stream_plan, solution.plan)

    def satisfy_skeleton(self, solution: StreamSolution) -> bool:
        satisfier_queue = self.queue
        self.queue = StablePriorityQueue()
        skeleton = self.add_solution(solution)
        if skeleton is None:
            self.queue = satisfier_queue
            return False
        while self.queue and skeleton.greedy:
            binding = self.pop_binding()
            if self.deactivate and not binding.active:
                binding.skeleton.dequeued.append(binding)
            else:
                self.process_binding(binding)
        skeleton_queue = self.queue
        self.queue = satisfier_queue
        while skeleton_queue:
            self.push_binding(skeleton_queue.pop())
        return True

    def is_full(self) -> bool:
        return self.num_solutions >= self.max_solutions

    def satisfy(
        self,
        max_time: float = INF,
        success_cost: float = INF,
        abort: bool = False,
        batchers: bool = True,
        verbose: bool = False,
    ) -> bool:
        assert (max_time < INF) or (self.max_solutions < INF)
        start_time = current_time()
        for skeleton in self.skeletons:
            for index in range(len(skeleton.iteration_calls)):
                skeleton.iteration_calls[index] = 0
        while (
            self.queue
            and (elapsed_time(start_time) < max_time)
            and not self.is_full()
            and implies(abort, not self.improved)
        ):
            binding = self.pop_binding()
            if self.deactivate and not binding.active:
                binding.skeleton.dequeued.append(binding)
                continue
            if verbose:
                print(
                    f"{len(self.queue)}) {binding} | Active: {binding.active} | Elapsed:"
                    f" {elapsed_time(start_time):.3f} sec"
                )
            self.process_binding(binding)
        if batchers:
            for skeleton in self.skeletons:
                for index in skeleton.batches:
                    if not skeleton.iteration_calls[index] and not self.is_full():
                        skeleton.process_batch(index)
        improved = self.improved
        self.cumulative_time += elapsed_time(start_time)
        return improved

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.skeletons})"

    __repr__ = __str__


def satisfy_streams(
    streams: StreamPlan, max_solutions: int = 1, visualize: bool = False, **kwargs: Any
) -> List[Dict[Constant, Constant]]:
    streams = reorder_stream_plan(streams)
    if visualize:
        visualize_stream_plan(streams)

    satisfier = Satisfier(max_solutions=max_solutions)
    satisfier.add_skeleton(streams)
    satisfier.satisfy(**kwargs)
    return [binding.mapping for binding in satisfier.sorted_solution_bindings]
