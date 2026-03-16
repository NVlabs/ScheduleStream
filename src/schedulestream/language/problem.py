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
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Union

# NVIDIA
from schedulestream.algorithm.utils import PartialPlan
from schedulestream.common.ordered_set import OrderedSet
from schedulestream.common.utils import (
    apply_mapping,
    assert_subset,
    flatten,
    not_null,
    remove_duplicates,
)
from schedulestream.language.action import DEFAULT_COST, Action, ActionInstance
from schedulestream.language.argument import Constant, wrap_argument
from schedulestream.language.connective import Conjunction
from schedulestream.language.durative import DurativeAction
from schedulestream.language.expression import Formula
from schedulestream.language.function import (
    Evaluation,
    Function,
    InputFormula,
    Term,
    create_formula,
)
from schedulestream.language.lazy import lazy_function
from schedulestream.language.predicate import Predicate, Type
from schedulestream.language.state import State
from schedulestream.language.stream import Stream
from schedulestream.language.utils import get_fluent_functions, infer_state, simplify_conjunction


class ProblemABC:
    def __init__(self, initial: State, goal: InputFormula = None):
        self.initial = initial
        goal = create_formula(goal)
        self.goal = goal

    def instantiate(self) -> "InstantiatedProblem":
        raise NotImplementedError()

    def simplify(self) -> "ProblemABC":
        raise NotImplementedError()


class Problem(ProblemABC):
    def __init__(
        self,
        initial: Optional[Union[State, List[Evaluation]]] = None,
        goal: InputFormula = None,
        actions: Optional[List[Action]] = None,
        streams: Optional[List[Stream]] = None,
        parent: Optional["Problem"] = None,
    ):
        self.actions = list(actions or [])
        self.streams = list(streams or [])

        if isinstance(initial, State):
            initial = initial
        elif isinstance(initial, dict):
            initial = State(assignments=initial)
        else:
            initial = State(evaluations=initial)
        initial = infer_state(initial)
        super().__init__(initial, goal)
        self.parent = parent

        self.simplify()

    @property
    def root(self) -> "Problem":
        if self.parent is None:
            return self
        return self.parent.root

    @property
    def is_ground(self) -> bool:
        return all(action.is_ground for action in self.actions)

    @property
    def is_temporal(self) -> bool:
        return any(isinstance(action, DurativeAction) for action in self.actions)

    @property
    def is_stream(self) -> bool:
        return bool(self.streams)

    @property
    def functions(self) -> List[Function]:
        return sorted(
            remove_duplicates(
                self.initial.functions
                + self.condition_functions
                + self.fluent_functions
                + self.cost_functions
                + self.duration_functions
            )
        )

    @property
    def condition_functions(self) -> List[Function]:
        conditions = [action.precondition for action in self.actions] + [self.goal]
        return remove_duplicates(flatten(condition.functions for condition in conditions))

    @property
    def fluent_functions(self) -> List[Function]:
        return get_fluent_functions(self.actions)

    @property
    def cost_functions(self) -> List[Function]:
        return remove_duplicates(flatten(action.cost.functions for action in self.actions))

    @property
    def duration_functions(self) -> List[Function]:
        return remove_duplicates(
            flatten(
                action.min_duration.functions + action.max_duration.functions
                for action in self.actions
                if isinstance(action, DurativeAction)
            )
        )

    @property
    def static_functions(self) -> List[Function]:
        return list(OrderedSet(self.functions).difference(OrderedSet(self.fluent_functions)))

    @property
    def assigned_functions(self) -> List[Function]:
        return remove_duplicates(self.initial.functions + self.fluent_functions)

    @property
    def procedural_functions(self) -> List[Function]:
        return list(filter(lambda f: f.is_procedural, self.functions))

    @property
    def stream_functions(self) -> List[Function]:
        return remove_duplicates(
            flatten(stream.output_condition.functions for stream in self.streams)
        )

    @property
    def predicates(self) -> List[Predicate]:
        return [function for function in self.functions if isinstance(function, Predicate)]

    @property
    def types(self) -> List[Type]:
        return [predicate for predicate in self.predicates if isinstance(predicate, Type)]

    @property
    def constants(self) -> List[Constant]:
        return remove_duplicates(
            self.initial.constants
            + self.goal.constants
            + list(flatten(action.constants for action in self.actions))
        )

    def _assert_static_costs(self):
        for action in self.actions:
            if isinstance(action.cost, Term):
                function = action.cost.function
                assert function not in self.assigned_functions
                assert_subset(function.fluents, self.static_functions)
                assert not function.fluents

    def has_function(self, name: str) -> bool:
        functions = [function for function in self.functions if function.name == name]
        return bool(functions)

    def get_function(self, name: str) -> Function:
        functions = [function for function in self.functions if function.name == name]
        if not functions:
            raise ValueError(f"Unknown functions: {name}\n\tFunctions: {self.functions}")
        if len(functions) >= 2:
            raise ValueError(f"Multiple functions with the same name: {functions}")
        [function] = functions
        return function

    def get_functions(self, names: List[str]) -> List[Function]:
        return list(map(self.get_function, names))

    def get_action(self, name: str) -> Action:
        actions = [action for action in self.actions if action.name == name]
        if not actions:
            raise ValueError(f"Unknown actions: {name}\n\tActions: {self.actions}")
        if len(actions) >= 2:
            raise ValueError(f"Multiple actions with the same name: {actions}")
        [action] = actions
        return action

    def get_actions(self, names: List[str]) -> List[Action]:
        return list(map(self.get_action, names))

    def satisfies_goal(self, state: Optional[State] = None) -> bool:
        if state is None:
            state = self.initial
        return self.goal.holds(state)

    def instantiate(self) -> "InstantiatedProblem":
        # NVIDIA
        from schedulestream.algorithm.instantiation import all_instantiate_actions

        actions = all_instantiate_actions(self.initial, self.actions)
        return InstantiatedProblem(self.initial, actions, self.goal, problem=self)

    def simplify(self) -> None:
        fluents = set(self.fluent_functions)
        for action in self.actions:
            action.simplify(fluents)

    def set_unit_costs(self) -> None:
        for action in self.actions:
            action.cost = wrap_argument(0.0)

    def add_partial_plan(self, partial_plan: PartialPlan) -> "Problem":
        goal = self.goal
        for action in partial_plan.actions:
            lifted = self.get_action(action.name)
            predicate = lifted.add_applied_predicate(action.bound_parameters)
            arguments = apply_mapping(action.parameter_values, predicate.parameters)
            atom = predicate.atom(arguments)
            goal = Conjunction(goal, atom).flatten()

        return self.__class__(
            initial=self.initial,
            goal=goal,
            actions=self.actions,
            streams=self.streams,
            parent=self,
        )

    def remove_conditions(self, functions: List[Function]) -> "Problem":
        return self.__class__(
            initial=self.initial,
            goal=self.goal.remove_functions(functions),
            actions=[action.remove_conditions(functions) for action in self.actions],
            streams=self.streams,
            parent=self,
        )

    def lazily_wrap_functions(self, duration: float = 1e-2) -> None:
        for function in self.procedural_functions:
            if function in self.duration_functions:
                function.self = wrap_argument(duration)
            elif function in self.cost_functions:
                function.default_value = wrap_argument(DEFAULT_COST)
            lazy_function(function)

    def clone(
        self,
        actions: Optional[List[Action]] = None,
        streams: Optional[List[Stream]] = None,
        parent: bool = True,
    ) -> "Problem":
        return self.__class__(
            initial=self.initial,
            goal=self.goal,
            actions=not_null(actions, self.actions),
            streams=not_null(streams, self.streams),
            parent=self if parent else None,
        )

    def lazy_clone(self, parent: bool = True, **kwargs: Any) -> "Problem":
        streams = [stream.lazy_clone(parent=False, **kwargs) for stream in self.streams]
        lazy_problem = self.clone(streams=streams, parent=parent)
        return lazy_problem

    def dump(self) -> None:
        initial_frequencies = Counter(evaluation.name for evaluation in self.initial)
        print(
            f"Statics ({len(self.static_functions)}): {self.static_functions}\n"
            f"Fluents ({len(self.fluent_functions)}): {self.fluent_functions}\n"
            f"Initial ({len(self.initial)}): {initial_frequencies}\n"
            f"Goal: {self.goal}"
        )
        if self.actions:
            print(f"\nActions ({len(self.actions)}):")
            for action in self.actions:
                action.dump()
        if self.streams:
            print(f"\nStreams ({len(self.streams)}):")
            for stream in self.streams:
                stream.dump()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(actions={self.actions}, streams={self.streams})"

    __repr__ = __str__


class InstantiatedProblem(ProblemABC):
    def __init__(
        self,
        initial: State,
        actions: List[ActionInstance],
        goal: Optional[Formula],
        problem: Optional[Problem] = None,
        parent: Optional["InstantiatedProblem"] = None,
    ):
        self.problem = problem
        self.actions = actions
        super().__init__(initial, goal)
        self.parent = parent

    @property
    def simplified(self) -> bool:
        return self.parent is not None

    def instantiate(self) -> "InstantiatedProblem":
        return self

    @property
    def terms(self) -> List[Term]:
        terms = OrderedSet(self.goal.terms)
        for action in self.actions:
            terms.update(action.terms)
        return list(terms)

    @property
    def fluents(self) -> Set[Term]:
        fluents = set()
        for action in self.actions:
            fluents.update(action.fluents)
        return fluents

    @property
    def evaluations(self) -> List[Evaluation]:
        evaluation = OrderedSet(self.initial.evaluations)
        for action in self.actions:
            evaluation.update(action.effect.simple_clause)
        return sorted(evaluation)

    @property
    def actions_from_language(self) -> Dict[str, List[ActionInstance]]:
        actions_from_language = {}
        for action in self.actions:
            if action.language is not None:
                actions_from_language.setdefault(action.language, []).append(action)
        return actions_from_language

    @property
    def evaluations_from_language(self) -> Dict[str, List[Evaluation]]:
        evaluations_from_language = {}
        for evaluation in self.evaluations:
            if evaluation.language is not None:
                evaluations_from_language.setdefault(evaluation.language, []).append(evaluation)
        return evaluations_from_language

    def clone(
        self,
        initial: Optional[State] = None,
        actions: Optional[List[ActionInstance]] = None,
        goal: Optional[Formula] = None,
    ) -> "InstantiatedProblem":
        return self.__class__(
            initial=not_null(initial, self.initial),
            actions=not_null(actions, self.actions),
            goal=not_null(goal, self.goal),
            problem=self.problem,
            parent=self,
        )

    def simplify(self) -> "InstantiatedProblem":
        fluents = set(self.fluents)
        actions = []
        for action in self.actions:
            action = action.simplify(self.initial, fluents)
            if action is not None:
                actions.append(action)

        goal = simplify_conjunction(self.goal, self.initial, fluents)

        fluent_evaluations = [
            evaluation for evaluation in self.initial.evaluations if evaluation.term in fluents
        ]
        initial = State(evaluations=fluent_evaluations)

        return self.__class__(initial, actions, goal, problem=self.problem, parent=self)

    def dump(self) -> None:
        print(f"Initial ({self.initial}):")
        print(f"Goal ({self.goal}):")
        print(f"Actions ({len(self.actions)}):")
        for action in self.actions:
            action.dump()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(#initial={len(self.initial)}, #actions={len(self.actions)},"
            f" goal={self.goal})"
        )

    __repr__ = __str__
