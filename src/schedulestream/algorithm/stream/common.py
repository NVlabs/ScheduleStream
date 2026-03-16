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
from dataclasses import dataclass
from typing import Any, List, Optional

# NVIDIA
from schedulestream.algorithm.stream.stream_plan import StreamPlan
from schedulestream.algorithm.temporal import TemporalSolution
from schedulestream.common.utils import merge_dicts, remove_duplicates
from schedulestream.language.predicate import Atom


@dataclass
class StreamSolution(TemporalSolution):
    stream_plan: Optional[StreamPlan] = None

    @property
    def streams_from_atoms(self) -> List[Atom]:
        streams_from_atoms = {}
        for stream_output in self.stream_plan:
            for atom in stream_output.output_condition.clause:
                streams_from_atoms.setdefault(atom, []).append(stream_output)
        return streams_from_atoms

    @property
    def stream_atoms(self) -> List[Atom]:
        return list(self.streams_from_atoms)

    @staticmethod
    def from_solution(solution: TemporalSolution, **kwargs: Any) -> "StreamSolution":
        return StreamSolution(**merge_dicts(solution.as_dict(), kwargs))

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(success={self.success}, makespan={self.makespan:.3f},"
            f" cost={self.cost:.3f}, optimal={self.optimal},"
            f" length={self.length},\n  stream_plan={self.stream_plan},\n  plan={self.plan}))"
        )
