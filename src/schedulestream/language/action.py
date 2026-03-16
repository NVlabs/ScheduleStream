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
from functools import cache, cached_property
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# NVIDIA
from schedulestream.common.ordered_set import OrderedSet
from schedulestream.common.utils import (
    EPSILON,
    apply_binding,
    compute_mapping,
    remove_duplicates,
    str_from_sequence,
)
from schedulestream.language.anonymous import Anonymous
from schedulestream.language.argument import (
    Constant,
    InputParameter,
    InputParameters,
    Parameter,
    Parameters,
    create_typed_parameters,
    is_parameter,
    unwrap_argument,
    unwrap_arguments,
    wrap_argument,
    wrap_arguments,
)
from schedulestream.language.connective import Conjunction
from schedulestream.language.expression import Expression
from schedulestream.language.function import (
    Evaluation,
    Function,
    InputFormula,
    Term,
    bind_language,
    create_formula,
    create_type_formula,
)
from schedulestream.language.predicate import Predicate
from schedulestream.language.state import State
from schedulestream.language.utils import simplify_conjunction, simplify_expression

DEFAULT_COST = EPSILON


class Action(Anonymous):
    def __init__(
        self,
        parameters: InputParameters = None,
        precondition: InputFormula = None,
        effect: InputFormula = None,
        cost: Union[float, Expression] = DEFAULT_COST,
        language: Optional[str] = None,
        parent: Optional["Action"] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        precondition = create_formula(precondition)
        effect = create_formula(effect)
        self.typed_parameters = create_typed_parameters(parameters)
        self.precondition = precondition & create_type_formula(self.typed_parameters)
        self.effect = effect
        self.cost = wrap_argument(cost)
        self.language = language
        self.parent = parent

        parameters = (
            set(self.precondition.parameters)
            | set(self.effect.parameters)
            | set(self.cost.parameters)
        )
        assert parameters <= set(self.parameters), parameters - set(self.parameters)

        self.applied_predicates = {}
        self.instances = {}

    def simplify(self, fluents: Optional[Set[Function]] = None) -> None:
        fluents = fluents or set()
        self.precondition = self.precondition & self.effect.simple_conjunction.condition
        implied_atoms = OrderedSet()
        for atom in self.precondition.simple_conjunction.clause:
            if atom.function not in fluents:
                implied_atoms.update(atom.condition.clause)
        self.precondition = Conjunction(
            *(formula for formula in self.precondition.clause if formula not in implied_atoms)
        )

    @property
    def parameters(self) -> Tuple[Parameter]:
        return tuple(self.typed_parameters.keys())

    @property
    def language_parameters(self) -> List[Parameter]:
        parameters = []
        if self.language is None:
            return parameters
        language = self.language
        for parameter in sorted(self.parameters, key=len, reverse=True):
            if parameter in language:
                parameters.append(parameter)
                language = language.replace(parameter, "")
        return [parameter for parameter in self.parameters if parameter in parameters]

    @property
    def constants(self) -> List[Constant]:
        return remove_duplicates(
            self.precondition.constants + self.effect.constants + self.cost.constants
        )

    @property
    def is_ground(self) -> bool:
        return not self.parameters

    @property
    def fluents(self) -> List[Function]:
        return self.effect.functions

    def normalize(self) -> List["Action"]:
        return [self]

    def applied_predicate(self, parameters: Optional[List[Parameter]]) -> Predicate:
        if parameters is None:
            parameters = self.parameters
        key = frozenset(parameters)
        if key not in self.applied_predicates:
            name = f"Applied{self.name.capitalize()}"
            active_parameters = [int(parameter in parameters) for parameter in self.parameters]
            if not all(active_parameters):
                name += "".join(map(str, active_parameters))
            typed_parameters = {
                parameter: self.typed_parameters[parameter] for parameter in parameters
            }
            self.applied_predicates[key] = Predicate(parameters=typed_parameters, name=name)
        return self.applied_predicates[key]

    def add_applied_predicate(self, parameters: Optional[List[Parameter]]) -> Predicate:
        if parameters is None:
            parameters = self.parameters
        key = frozenset(parameters)
        added = key in self.applied_predicates
        predicate = self.applied_predicate(parameters)
        if not added:
            atom = predicate.atom(predicate.parameters)
            self.effect = Conjunction(self.effect, atom)
        return predicate

    def remove_conditions(self, functions: List[Function]) -> "Action":
        return self.__class__(
            name=self.name,
            parameters=self.parameters,
            precondition=self.precondition.remove_functions(functions),
            effect=self.effect,
            cost=self.cost,
            language=self.language,
        )

    def _instantiate(self, arguments: List[Any]) -> "ActionInstance":
        return ActionInstance(self, arguments)

    def instantiate(self, arguments: List[Any]) -> "ActionInstance":
        arguments = tuple(wrap_arguments(arguments))
        if arguments not in self.instances:
            self.instances[arguments] = self._instantiate(list(arguments))
        return self.instances[arguments]

    def __call__(self, *arguments: Any) -> "ActionInstance":
        return self.instantiate(list(arguments))

    def partially_instantiate(
        self, parameter_values: Dict[Parameter, Constant]
    ) -> "ActionInstance":
        arguments = apply_binding(parameter_values, self.parameters)
        return self.instantiate(arguments)

    @property
    def actions(self) -> List["Action"]:
        return [self]

    def dump(self):
        print(
            f"{self.__class__.__name__}(\n"
            f" name={self.name},\n"
            f" parameters={self.parameters},\n"
            f" precondition={self.precondition},\n"
            f" effect={self.effect},\n"
            f" cost={self.cost})"
        )

    def __str__(self):
        return f"{self.name}({str_from_sequence(self.parameters)})"

    __repr__ = __str__


class ActionInstance:
    def __init__(
        self,
        action: Action,
        arguments: List[Any],
        parent: Optional["ActionInstance"] = None,
    ) -> None:
        arguments = list(arguments)
        assert len(action.parameters) == len(arguments), (action.parameters, arguments)
        self.action = action
        self.arguments = arguments
        self.parent = parent
        self.precondition = self.action.precondition.bind(self.parameter_values)
        self.effect = self.action.effect.bind(self.parameter_values)
        self.cost = self.action.cost.bind(self.parameter_values)

    @property
    def lifted(self) -> Action:
        return self.action

    @property
    def name(self) -> str:
        return self.action.name

    @property
    def parameters(self):
        return self.action.parameters

    @cached_property
    def parameter_values(self) -> Dict[Parameter, Constant]:
        return compute_mapping(self.parameters, self.arguments)

    @property
    def unbound_parameters(self) -> List[Parameter]:
        return [
            parameter for parameter, value in self.parameter_values.items() if is_parameter(value)
        ]

    @property
    def bound_parameters(self) -> List[Parameter]:
        return [
            parameter for parameter in self.parameters if parameter not in self.unbound_parameters
        ]

    @property
    def terms(self) -> List[Term]:
        return remove_duplicates(self.precondition.terms + self.effect.terms + self.cost.terms)

    @property
    def fluents(self) -> List[Term]:
        return self.effect.terms

    @property
    def language(self):
        return bind_language(self.action.language, self.parameter_values)

    def is_applicable(self, state: State, debug: bool = False) -> bool:
        if debug:
            holds = self.precondition.holds(state)
            print(f"\nState: {state}\nAction: {self}\nHolds: {holds}")
            for i, evaluation in enumerate(self.precondition.evaluations):
                print(f"{i}) {evaluation} => {evaluation.holds(state)}")
            if not holds:
                input("Continue?")
        return self.precondition.holds(state)

    @cache
    def support(self, state: State) -> List[Evaluation]:
        return remove_duplicates(self.precondition.support(state) + self.cost.support(state))

    @cache
    def apply(self, state: State) -> Optional[State]:
        evaluations = self.effect.apply(state)
        if evaluations is None:
            return None
        return state.new_state(evaluations)

    def get_argument(self, parameter: InputParameter) -> Any:
        return unwrap_argument(self.parameter_values[parameter])

    def get_arguments(self, parameters: Optional[List[InputParameter]] = None) -> List[Any]:
        if parameters is None:
            parameters = self.parameters
        return list(map(self.get_argument, parameters))

    def unwrap_arguments(self) -> List[Any]:
        return unwrap_arguments(self.arguments)

    def unwrap(self) -> "ActionInstance":
        return self.__class__(self.action, self.unwrap_arguments(), parent=self)

    def clone(self) -> "ActionInstance":
        return self.__class__(self.action, self.arguments, parent=self)

    def bind(self, mapping: Dict[Constant, Constant]) -> "ActionInstance":
        arguments = apply_binding(mapping, self.arguments)
        return self.action.instantiate(arguments)

    def partially_instantiate(self, parameters: List[Parameter]) -> "ActionInstance":
        parameter_values = {
            parameter: value
            for parameter, value in self.parameter_values.items()
            if parameter in parameters
        }
        return self.action.partially_instantiate(parameter_values)

    def simplify(self, initial: State, fluents: Set[Term]) -> Optional["ActionInstance"]:
        action = self.clone()
        precondition = simplify_conjunction(action.precondition, initial, fluents)
        if precondition is None:
            return None
        action.precondition = precondition
        action.cost = simplify_expression(action.cost, initial, fluents)
        return action

    @property
    def instances(self) -> List["ActionInstance"]:
        return [self]

    @property
    def root(self) -> "ActionInstance":
        if self.parent is None:
            return self
        return self.parent.root

    def dump(self):
        print(
            f"{self.__class__.__name__}(\n"
            f" name={self.name},\n"
            f" arguments={self.arguments},\n"
            f" precondition={self.precondition},\n"
            f" effect={self.effect},\n"
            f" cost={self.cost})"
        )

    def __str__(self) -> str:
        return f"{self.name}({str_from_sequence(self.arguments)})"

    __repr__ = __str__
