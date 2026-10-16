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
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional

# NVIDIA
from schedulestream.algorithm.stream.satisfier import Binding, Satisfier, Skeleton
from schedulestream.algorithm.stream.stream_plan import (
    StreamPlan,
    reorder_stream_plan,
    visualize_stream_plan,
)
from schedulestream.algorithm.utils import Plan
from schedulestream.common.utils import (
    INF,
    current_time,
    elapsed_time,
    flatten,
    implies,
    remove_duplicates,
)
from schedulestream.language.argument import Constant
from schedulestream.language.constraint import Cost
from schedulestream.language.stream import StreamOutput


class BatchSkeleton(Skeleton):
    def __init__(
        self, satisfier: "BatchSatisfier", stream_plan: StreamPlan, plan: Optional[Plan] = None
    ):
        self.bindings_from_index: Dict[int, List[Binding]] = defaultdict(list)
        self.num_processed = 0
        super().__init__(satisfier, stream_plan, plan)

    def add_binding(self, binding: Binding) -> None:
        super().add_binding(binding)
        if not binding.exhausted:
            self.bindings_from_index[binding.index].append(binding)

    def local_costs(self, index: int) -> List[Cost]:
        """Costs after step ``index`` whose only free parameters are that stream's inputs/outputs.
        """
        stream = self.stream_plan[index]
        local = set(stream.inputs) | set(stream.outputs)
        local_costs = []
        for cost in self.costs_from_index[index + 1]:
            free = {const for const in cost.expression.constants if const in self.parameters}
            if free and free <= local:
                local_costs.append(cost)
        return local_costs

    def local_cost_fn(self, bindings: List[Binding]) -> Optional[Callable[[StreamOutput], float]]:
        """Cost function over the new outputs of the bindings' step, evaluating their batched local
        costs together. None if there are no such costs."""
        local_costs = [
            cost for cost in self.local_costs(bindings[0].index) if cost.batch_size is not None
        ]
        if not local_costs:
            return None
        outputs = remove_duplicates(
            flatten(binding.get_outputs() for binding in bindings if binding.queued_outputs is None)
        )
        mappings = [output.parameter_values for output in outputs]
        costs = self.evaluate_costs(local_costs, mappings)
        return dict(zip(outputs, costs)).__getitem__

    def all_bindings(self) -> List[Binding]:
        bindings = flatten(self.bindings_from_index.values())
        return [binding for binding in bindings if not binding.exhausted]

    def process_step(self, index: int, verbose: bool = True) -> bool:
        start_time = current_time()
        stream = self.stream_plan[index].stream
        batch_size = stream.batch_size if index in self.batches else None
        bindings = self.sort_bindings(self.bindings_from_index[index], verbose=False)
        bindings_from_instance = defaultdict(list)
        for binding in bindings:
            if not binding.exhausted:
                bindings_from_instance[binding.stream_instance].append(binding)
        instances = list(bindings_from_instance)
        if not instances:
            return False
        num_instances = 1 if batch_size is None else batch_size
        if num_instances < len(instances):
            instances = instances[:num_instances]
        selected_bindings = set(flatten(bindings_from_instance[instance] for instance in instances))
        bindings = [binding for binding in bindings if binding in selected_bindings]

        if batch_size is not None:
            stream.get_batch(instances)
        cost_fn = self.local_cost_fn(bindings)
        queue = deque(bindings)
        new_bindings = []
        while queue and (len(new_bindings) < self.satisfier.max_candidates):
            binding = queue.popleft()
            new_binding = binding.next_binding(cost_fn)
            if new_binding is not None:
                new_bindings.append(new_binding)
                queue.append(binding)
        if verbose:
            print(
                f"Stream {index}/{self.num_streams}: {self.stream_plan[index]} | Bindings:"
                f" {len(bindings)} | Instances: {len(instances)} | Batch size: {batch_size} | New"
                f" bindings: {len(new_bindings)} | Elapsed: {elapsed_time(start_time):.3f} sec"
            )
        for new_binding in self.sort_bindings(new_bindings, verbose=verbose):
            self.add_binding(new_binding)
        return True

    def process(self, verbose: bool = True) -> bool:
        self.num_processed += 1
        processed = False
        for index in range(self.num_streams):
            processed |= self.process_step(index, verbose=verbose)
        return processed


class BatchSatisfier(Satisfier):
    skeleton_class = BatchSkeleton

    def __init__(self, *args: Any, max_candidates: int = 1000, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.max_candidates = max_candidates
        self.num_satisfy = 0

    def __len__(self) -> int:
        return len(self.all_bindings())

    def all_bindings(self) -> List[Binding]:
        return list(flatten(skeleton.all_bindings() for skeleton in self.skeletons))

    def process_skeleton(self, skeleton: BatchSkeleton) -> None:
        skeleton.process()

    def satisfy(
        self,
        max_time: float = INF,
        success_cost: float = INF,
        max_iterations: float = INF,
        abort: bool = False,
        verbose: bool = True,
    ) -> bool:
        assert (max_time < INF) or (self.max_solutions < INF) or (max_iterations < INF)
        start_time = current_time()
        iterations = 0
        while (
            (elapsed_time(start_time) < max_time)
            and not self.is_full()
            and (iterations < max_iterations)
            and implies(abort, not self.improved)
            and not (self.solved and (self.min_cost <= success_cost))
        ):
            iterations += 1
            processed = False
            for skeleton in self.skeletons:
                processed |= skeleton.process(verbose=verbose)
            if verbose:
                print(
                    f"Satisfy {self.num_satisfy} | Iteration {iterations} |"
                    f" Skeletons: {self.num_skeletons} |"
                    f" Solutions: {self.num_solutions} | Elapsed:"
                    f" {elapsed_time(start_time):.3f} sec\n"
                )
            if not processed:
                break
        self.num_satisfy += 1
        self.cumulative_time += elapsed_time(start_time)
        return self.improved


def batch_satisfy_streams(
    streams: StreamPlan, max_solutions: int = 1, visualize: bool = False, **kwargs: Any
) -> List[Dict[Constant, Constant]]:
    streams = reorder_stream_plan(streams)
    if visualize:
        visualize_stream_plan(streams)

    satisfier = BatchSatisfier(max_solutions=max_solutions)
    satisfier.add_skeleton(streams)
    satisfier.satisfy(**kwargs)
    return [binding.mapping for binding in satisfier.sorted_solution_bindings]
