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
from functools import cached_property
from typing import Any, Dict, Iterator, List, Optional, Set, Union

# NVIDIA
from schedulestream.common.utils import EPSILON, INF, not_null
from schedulestream.language.action import Action, ActionInstance
from schedulestream.language.argument import (
    Argument,
    Constant,
    InputParameters,
    Parameter,
    wrap_argument,
)
from schedulestream.language.connective import Conjunction
from schedulestream.language.effect import Assignment, EffectFunction
from schedulestream.language.expression import Expression, Formula
from schedulestream.language.function import (
    Evaluation,
    Function,
    InputFormula,
    Term,
    create_formula,
)
from schedulestream.language.predicate import Atom, Predicate
from schedulestream.language.state import State
from schedulestream.language.utils import INTERNAL_PREFIX, simplify_conjunction

START_SUFFIX = "start"
OVER_SUFFIX = "over"
END_SUFFIX = "end"
CLOCK = EPSILON


class OngoingPredicate(Predicate):
    def __init__(self, durative_action: "DurativeAction", **kwargs: Any) -> None:
        super().__init__(
            durative_action.typed_parameters,
            name=f"{INTERNAL_PREFIX}{durative_action.name}_ongoing",
            **kwargs,
        )
        self.durative_action = durative_action


class RemainingFunction(Function):
    def __init__(self, durative_action: "DurativeAction", **kwargs: Any) -> None:
        super().__init__(
            durative_action.typed_parameters,
            name=f"{INTERNAL_PREFIX}{durative_action.name}_remaining",
            default=durative_action.default_remaining,
            **kwargs,
        )
        self.durative_action = durative_action


class StartAction(Action):
    def __init__(self, durative_action: "DurativeAction", **kwargs: Any) -> None:
        super().__init__(
            name=f"{INTERNAL_PREFIX}{START_SUFFIX}_{durative_action.name}",
            parameters=durative_action.typed_parameters,
            precondition=durative_action.start_condition
            & ~durative_action.ongoing_atom
            & durative_action.over_condition,
            effect=durative_action.start_effect
            & durative_action.ongoing_atom
            & Assignment(durative_action.remaining_term, durative_action.min_duration),
            cost=durative_action.cost,
            **kwargs,
        )
        self.durative_action = durative_action

    def _instantiate(self, arguments: List[Any]) -> "StartInstance":
        raise RuntimeError()


class OverAction(Action):
    def __init__(self, durative_action: "DurativeAction", **kwargs: Any) -> None:
        super().__init__(
            name=f"{INTERNAL_PREFIX}{OVER_SUFFIX}_{durative_action.name}",
            parameters=durative_action.typed_parameters,
            precondition=durative_action.over_condition,
            effect=None,
            cost=0.0,
            **kwargs,
        )
        self.durative_action = durative_action

    def _instantiate(self, arguments: List[Any]) -> "OverInstance":
        raise RuntimeError()


def get_ongoing_terms(state: State) -> Iterator[Term]:
    for ongoing_predicate in state.functions:
        if isinstance(ongoing_predicate, OngoingPredicate):
            yield from state.get_terms(ongoing_predicate)


def get_ongoing_actions(state: State) -> Dict["DurativeInstance", float]:
    ongoing_actions = {}
    for ongoing_term in get_ongoing_terms(state):
        durative_action = ongoing_term.predicate.durative_action
        arguments = ongoing_term.arguments
        durative_instance = durative_action.instantiate(arguments)
        remaining_term = durative_action.remaining_function.instantiate(arguments)
        remaining_value = remaining_term.evaluate(state)
        ongoing_actions[durative_instance] = remaining_value
    return ongoing_actions


class EndAction(Action):
    def __init__(
        self, durative_action: "DurativeAction", exact: bool = False, **kwargs: Any
    ) -> None:
        def end_fn(state: State, *arguments: Any) -> Optional[State]:
            remaining_term = durative_action.remaining_function.instantiate(arguments)
            remaining_value = state[remaining_term]

            evaluations = []
            for ongoing_term in get_ongoing_terms(state):
                other_action = ongoing_term.predicate.durative_action
                other_arguments = ongoing_term.arguments
                other_remaining_term = other_action.remaining_function.instantiate(other_arguments)
                if remaining_term == other_remaining_term:
                    continue
                other_remaining_value = other_remaining_term.evaluate(state)
                new_remaining_value = other_remaining_value - remaining_value
                if not exact:
                    new_remaining_value = max(0.0, new_remaining_value)
                if new_remaining_value < 0.0:
                    return None
                evaluations.append(other_remaining_term <= new_remaining_value)
            return evaluations

        end_term = EffectFunction(
            durative_action.typed_parameters,
            name=f"{INTERNAL_PREFIX}{durative_action.name}_{END_SUFFIX}",
            definition=end_fn,
            fluent=True,
        ).instantiate(durative_action.parameters)

        super().__init__(
            name=f"{INTERNAL_PREFIX}{END_SUFFIX}_{durative_action.name}",
            parameters=durative_action.typed_parameters,
            precondition=durative_action.end_condition & durative_action.ongoing_atom,
            effect=durative_action.end_effect
            & ~durative_action.ongoing_atom
            & end_term
            & Assignment(durative_action.remaining_term, durative_action.default_remaining),
            cost=durative_action.remaining_term,
            **kwargs,
        )
        self.durative_action = durative_action

    def _instantiate(self, arguments: List[Any]) -> "EndInstance":
        raise RuntimeError()


def over_fn(state: State) -> Optional[List[Evaluation]]:
    for ongoing_predicate in state.functions:
        if not isinstance(ongoing_predicate, OngoingPredicate):
            continue
        for ongoing_term in state.get_terms(ongoing_predicate):
            pass

    evaluations = []
    return evaluations


OVER_EFFECT = EffectFunction(
    parameters=[],
    name=f"{INTERNAL_PREFIX}over",
    definition=over_fn,
    fluent=True,
).instantiate([])


class DurativeAction(Action):
    def __init__(
        self,
        parameters: InputParameters = None,
        over_condition: InputFormula = None,
        start_condition: InputFormula = None,
        end_condition: InputFormula = None,
        start_effect: InputFormula = None,
        end_effect: InputFormula = None,
        min_duration: Union[float, Expression] = 1.0,
        max_duration: Union[float, Expression] = INF,
        cost: Union[float, Expression] = 0.0,
        **kwargs: Any,
    ):
        self.start_condition = create_formula(start_condition)
        self.over_condition = create_formula(over_condition)
        self.end_condition = create_formula(end_condition)
        self.start_effect = create_formula(start_effect)
        self.end_effect = create_formula(end_effect)
        self.min_duration = wrap_argument(min_duration)
        self.max_duration = wrap_argument(max_duration)
        assert self.max_duration.value == INF

        precondition = self.start_condition & self.over_condition & self.end_condition

        effects = {}
        for evaluation in self.start_effect.clause + self.end_effect.clause:
            effects[evaluation.term] = evaluation
        effect = Conjunction(*effects.values())

        super().__init__(
            parameters=parameters,
            precondition=precondition,
            effect=effect,
            cost=cost,
            **kwargs,
        )

    def _instantiate(self, arguments: List[Any]) -> "DurativeInstance":
        return DurativeInstance(self, arguments)

    @cached_property
    def ongoing_predicate(self) -> OngoingPredicate:
        return OngoingPredicate(self)

    @property
    def ongoing_atom(self) -> Atom:
        return self.ongoing_predicate.atom(self.parameters)

    @property
    def default_remaining(self) -> float:
        return CLOCK

    @cached_property
    def remaining_function(self) -> RemainingFunction:
        return RemainingFunction(self)

    @cached_property
    def remaining_term(self) -> Term:
        return self.remaining_function.instantiate(self.parameters)

    @property
    def fluents(self) -> List[Function]:
        return super().fluents + [self.ongoing_predicate, self.remaining_function]

    @cached_property
    def start_action(self) -> StartAction:
        return StartAction(self)

    @cached_property
    def over_action(self) -> OverAction:
        return OverAction(self)

    @cached_property
    def end_action(self) -> EndAction:
        return EndAction(self)

    @property
    def actions(self) -> List[Action]:
        return [self.start_action, self.end_action]

    def remove_conditions(self, functions: List[Function]) -> "DurativeAction":
        return self.__class__(
            name=self.name,
            parameters=self.parameters,
            over_condition=self.over_condition.remove_functions(functions),
            start_condition=self.start_condition.remove_functions(functions),
            end_condition=self.end_condition.remove_functions(functions),
            start_effect=self.start_effect,
            end_effect=self.end_effect,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
            cost=self.cost,
        )


class DurativeInstance(ActionInstance):
    def __init__(self, action: DurativeAction, arguments: List[Any], **kwargs: Any) -> None:
        super().__init__(action, arguments, **kwargs)

        self.start_condition = action.start_condition.bind(self.parameter_values)
        self.start_effect = action.start_effect.bind(self.parameter_values)
        self.over_condition = action.over_condition.bind(self.parameter_values)
        self.end_condition = action.end_condition.bind(self.parameter_values)
        self.end_effect = action.end_effect.bind(self.parameter_values)
        self.min_duration = action.min_duration.bind(self.parameter_values)
        self.max_duration = action.max_duration.bind(self.parameter_values)

        self.start_instance = StartInstance(self)
        self.over_instance = OverInstance(self)
        self.end_instance = EndInstance(self)

    @property
    def start_action(self) -> StartAction:
        return self.action.start_action

    @property
    def over_action(self) -> OverAction:
        return self.action.over_action

    @property
    def end_action(self) -> EndAction:
        return self.action.end_action

    @property
    def instances(self) -> List[ActionInstance]:
        return [self.start_instance, self.end_instance]

    @property
    def ongoing_atom(self) -> Atom:
        return self.action.ongoing_predicate.atom(self.arguments)

    @cached_property
    def remaining_term(self) -> Term:
        return self.action.remaining_function.instantiate(self.arguments)

    @property
    def fluents(self) -> List[Term]:
        return super().fluents + [self.ongoing_atom.term, self.remaining_term]

    def simplify(self, initial: State, fluents: Set[Term]) -> Optional["DurativeInstance"]:
        action = super().simplify(initial, fluents)
        if action is None:
            return action
        action.start_condition = simplify_conjunction(action.start_condition, initial, fluents)
        action.over_condition = simplify_conjunction(action.over_condition, initial, fluents)
        action.end_condition = simplify_conjunction(action.end_condition, initial, fluents)
        action.start_instance = action.start_instance.simplify(initial, fluents)
        action.over_instance = action.over_instance.simplify(initial, fluents)
        action.end_instance = action.end_instance.simplify(initial, fluents)
        if not all(
            [
                action.start_condition,
                action.over_condition,
                action.end_condition,
                action.start_instance,
                action.over_instance,
                action.end_instance,
            ]
        ):
            return None
        return action

    def dump(self):
        print(
            f"{self.__class__.__name__}(\n"
            f" name={self.name},\n"
            f" arguments={self.arguments},\n"
            f" start_condition={self.start_condition},\n"
            f" start_effect={self.start_effect},\n"
            f" over_condition={self.over_condition},\n"
            f" end_condition={self.end_condition},\n"
            f" end_effect={self.end_effect},\n"
            f" duration=[{self.min_duration}, {self.max_duration}],\n"
            f" cost={self.cost})"
        )


class StartInstance(ActionInstance):
    def __init__(self, durative_instance: DurativeInstance, *args: Any, **kwargs: Any) -> None:
        self.durative_instance = durative_instance
        action = durative_instance.start_action
        arguments = durative_instance.arguments
        super().__init__(action, arguments, *args, **kwargs)

    def clone(self) -> "StartInstance":
        instance = self.__class__(self.durative_instance, parent=self)
        return instance


class OverInstance(ActionInstance):
    def __init__(self, durative_instance: DurativeInstance, *args: Any, **kwargs: Any) -> None:
        self.durative_instance = durative_instance
        action = durative_instance.over_action
        arguments = durative_instance.arguments
        super().__init__(action, arguments, *args, **kwargs)

    def clone(self) -> "OverInstance":
        return self.__class__(self.durative_instance, parent=self)


class EndInstance(ActionInstance):
    def __init__(self, durative_instance: DurativeInstance, *args: Any, **kwargs: Any) -> None:
        self.durative_instance = durative_instance
        action = durative_instance.end_action
        arguments = durative_instance.arguments
        super().__init__(action, arguments, *args, **kwargs)

    @property
    def ongoing_atom(self) -> Atom:
        return self.durative_instance.ongoing_atom

    @cached_property
    def remaining_term(self) -> Term:
        return self.durative_instance.remaining_term

    def clone(self) -> "EndInstance":
        return self.__class__(self.durative_instance, parent=self)


@dataclass
class TimedAction:
    action: ActionInstance
    start: float
    end: float = INF

    @property
    def parameter_values(self) -> Dict[Parameter, Constant]:
        return self.action.parameter_values

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def name(self) -> str:
        return self.action.name

    @property
    def precondition(self) -> Formula:
        return self.action.precondition

    @property
    def effect(self) -> Formula:
        return self.action.precondition

    @property
    def cost(self) -> float:
        return self.action.cost

    @property
    def language(self) -> Optional[str]:
        if self.action.language is None:
            return None
        return f"{self.action.language} from {self.start:.2f} to {self.end:.2f}"

    @property
    def root(self) -> "TimedAction":
        return self.clone(action=self.action.root)

    def clone(
        self,
        action: Optional[ActionInstance] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> "TimedAction":
        return self.__class__(
            action=not_null(action, self.action),
            start=not_null(start, self.start),
            end=not_null(end, self.end),
        )

    def bind(self, mapping: Dict[Parameter, Argument]) -> "TimedAction":
        return self.clone(action=self.action.bind(mapping))

    def unwrap_arguments(self) -> List[Any]:
        return self.action.unwrap_arguments()

    def __str__(self) -> str:
        return f"{self.action}[{self.start:.2f}, {self.end:.2f}]"

    __repr__ = __str__
