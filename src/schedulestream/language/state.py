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
from collections.abc import Iterable, Sized
from functools import cached_property
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Union

# NVIDIA
from schedulestream.common.utils import flatten, remove_duplicates, str_from_sequence
from schedulestream.language.argument import Constant
from schedulestream.language.connective import Conjunction
from schedulestream.language.function import Evaluation, Function, Term
from schedulestream.language.predicate import Atom


def evaluations_from_assignments(assignments: Dict[Term, Any]) -> List[Evaluation]:
    evaluations = []
    for term, value in assignments.items():
        assert isinstance(term, Term), term
        evaluation = term.equals(value)
        evaluations.append(evaluation)
    return evaluations


class State(Sized, Iterable):
    def __init__(
        self,
        evaluations: List[Evaluation] = None,
        assignments: Dict[Term, Union[Any, Evaluation]] = None,
        state: Optional["State"] = None,
    ):
        evaluations = list(evaluations or [])
        assignments = dict(assignments or {})
        evaluations.extend(evaluations_from_assignments(assignments))

        self.assignments = dict()
        if state is not None:
            self.assignments.update(state.assignments)
        for evaluation in evaluations:
            assert isinstance(evaluation, Evaluation), evaluation
            assert evaluation.grounded, evaluation
            default_value = evaluation.function.default_value
            if evaluation.value == default_value:
                self.assignments.pop(evaluation.term, default_value)
            else:
                self.assignments[evaluation.term] = evaluation
        self._hash = None

    def __len__(self) -> int:
        return len(self.assignments)

    def __iter__(self) -> Iterator[Evaluation]:
        return iter(self.evaluations)

    @property
    def terms(self) -> List[Term]:
        return list(self.assignments.keys())

    @property
    def evaluations(self) -> List[Evaluation]:
        return list(self.assignments.values())

    @property
    def functions(self) -> List[Function]:
        return remove_duplicates(term.function for term in self.terms)

    @property
    def constants(self) -> List[Constant]:
        return remove_duplicates(flatten(evaluation.constants for evaluation in self.evaluations))

    @property
    def clause(self):
        return self.evaluations

    @property
    def conjunction(self) -> Conjunction:
        return Conjunction(*self.clause)

    @property
    def atoms(self) -> List[Atom]:
        return [evaluation for evaluation in self.evaluations if isinstance(evaluation, Atom)]

    @property
    def evaluations_from_language(self) -> Dict[str, List[Evaluation]]:
        evaluations_from_language = {}
        for evaluation in self.evaluations:
            if evaluation.language is not None:
                evaluations_from_language.setdefault(evaluation.language, []).append(evaluation)
        return evaluations_from_language

    def get_terms(self, function: Function) -> List[Term]:
        return [term for term in self.terms if term.function == function]

    def get_evaluation(self, term: Term) -> Evaluation:
        assert isinstance(term, Term), term
        if term in self.assignments:
            return self.assignments[term]
        value = term.function.default_value
        evaluation = term.equals(value)
        return evaluation

    def get_value(self, term: Term) -> Constant:
        if term in self.assignments:
            return self.assignments[term].value
        return term.function.default_value

    def get_raw_value(self, term: Term) -> Any:
        return self.get_value(term).value

    def holds(self, evaluation: Evaluation) -> bool:
        return self.get_value(evaluation.term) == evaluation.value

    def __contains__(self, evaluation: Evaluation) -> bool:
        return self.holds(evaluation)

    def __getitem__(self, term: Term) -> Any:
        return self.get_value(term).value

    def new_state(self, evaluations: Optional[List[Evaluation]] = None) -> "State":
        return self.__class__(evaluations=evaluations, state=self)

    def clone(self) -> "State":
        return State(state=self)

    @cached_property
    def _frozenset(self) -> FrozenSet[Evaluation]:
        return frozenset(self.evaluations)

    def __eq__(self, other: Any) -> bool:
        if self.__class__ is not other.__class__:
            return False
        return self._frozenset == other._frozenset

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self.__class__, self._frozenset))
        return self._hash

    def dump(self):
        print(f"\n{self.__class__.__name__} ({len(self.assignments)}):")
        for i, evaluation in enumerate(sorted(self.evaluations)):
            print(f"{i+1}) {evaluation}")

    def __str__(self):
        return f"{self.__class__.__name__}({str_from_sequence(sorted(self.evaluations))})"

    __repr__ = __str__
