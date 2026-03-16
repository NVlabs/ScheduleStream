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
from collections import Counter, defaultdict
from itertools import product
from typing import Any, Dict, List, Optional, Set, Union

# NVIDIA
from schedulestream.common.ordered_set import OrderedSet
from schedulestream.common.utils import (
    apply_mapping,
    compute_mapping,
    current_time,
    elapsed_time,
    safe_zip,
)
from schedulestream.language.action import Action, ActionInstance
from schedulestream.language.argument import Constant, Parameter
from schedulestream.language.connective import Conjunction
from schedulestream.language.expression import Formula
from schedulestream.language.function import Evaluation, Function
from schedulestream.language.predicate import Atom
from schedulestream.language.state import State
from schedulestream.language.stream import Stream, StreamInstance
from schedulestream.language.utils import get_fluent_functions


def atoms_from_evaluations(
    evaluations: List[Evaluation], fluents: Optional[Set[Function]] = None
) -> List[Atom]:
    if fluents is None:
        fluents = set()
    evaluations = [evaluation for evaluation in evaluations if evaluation.function not in fluents]
    atoms = [evaluation.literal for evaluation in evaluations]
    atoms = [atom for atom in atoms if isinstance(atom, Atom)]
    return atoms


def atoms_from_formula(formula: Formula, **kwargs: Any) -> List[Atom]:
    return atoms_from_evaluations(formula.simple_clause, **kwargs)


def get_static_condition(condition: Formula, fluents: Set[Function]) -> Conjunction:
    atoms = atoms_from_formula(condition, fluents=fluents)
    return Conjunction(*atoms)


def extract_parameter_mapping(
    condition_list: List[Atom], atom_list: List[Atom]
) -> Optional[Dict[Parameter, Constant]]:
    parameter_mapping = {}
    for precondition, atom in safe_zip(condition_list, atom_list):
        atom_mapping = compute_mapping(precondition.inputs, atom.inputs)
        for parameter, value in atom_mapping.items():
            if not isinstance(parameter, Parameter) and (parameter != value):
                return None
            if parameter_mapping.get(parameter, value) != value:
                return None
        parameter_mapping.update(atom_mapping)
    return parameter_mapping


def instantiate_condition(
    state: Union[State, List[Atom]], condition: Conjunction, fluents: Optional[Set[Function]] = None
) -> List[Dict[Parameter, Constant]]:
    if fluents is None:
        fluents = set()

    atoms_from_predicate = defaultdict(list)
    for evaluation in state:
        atom = evaluation.literal
        if isinstance(atom, Atom):
            atoms_from_predicate[atom.function].append(atom)

    parameters = condition.parameters
    unbound_parameters = set(parameters)
    condition_order = []
    while unbound_parameters:
        atoms = atoms_from_formula(condition, fluents=fluents)
        remaining_order = sorted(
            atoms,
            key=lambda a: (
                a.function in fluents,
                -len(set(a.inputs) & set(unbound_parameters)),
                len(atoms_from_predicate[a.function]),
            ),
        )
        if not remaining_order:
            raise RuntimeError(f"Unbound parameters {unbound_parameters} for condition {condition}")
        atom = remaining_order[0]
        if not set(atom.inputs) & set(unbound_parameters):
            raise RuntimeError(f"Unbound parameters {unbound_parameters} for condition {condition}")
        condition_order.append(atom)
        unbound_parameters -= set(atom.inputs)

    predicate_domains = [atoms_from_predicate[atom.function] for atom in condition_order]
    parameter_mappings = []
    for atom_combo in product(*predicate_domains):
        parameter_mapping = extract_parameter_mapping(condition_order, atom_combo)
        if parameter_mapping is None:
            continue
        parameter_mappings.append(parameter_mapping)
    return parameter_mappings


def instantiate_stream(
    state: Union[State, List[Atom]],
    stream: Stream,
    **kwargs: Any,
) -> List[ActionInstance]:
    parameter_mappings = instantiate_condition(state, stream.input_condition, **kwargs)
    stream_instances = OrderedSet()
    for parameter_mapping in parameter_mappings:
        values = apply_mapping(parameter_mapping, stream.inputs)
        stream_instances.add(stream.instantiate(values))
    return list(stream_instances)


def instantiate_streams(
    state: Union[State, List[Atom]],
    streams: List[Stream],
    **kwargs: Any,
) -> List[StreamInstance]:
    stream_instances = []
    for stream in streams:
        stream_instances.extend(instantiate_stream(state, stream, **kwargs))
    return stream_instances


def instantiate_action(
    state: Union[State, List[Atom]],
    action: Action,
    **kwargs: Any,
) -> List[ActionInstance]:
    if isinstance(action, ActionInstance):
        return [action]

    condition = action.precondition
    parameter_mappings = instantiate_condition(state, condition, **kwargs)

    action_instances = OrderedSet()
    for parameter_mapping in parameter_mappings:
        values = apply_mapping(parameter_mapping, action.parameters)
        action_instances.add(action.instantiate(values))
    return list(action_instances)


def instantiate_actions(
    state: Union[State, List[Atom]],
    actions: List[Action],
    **kwargs: Any,
) -> List[ActionInstance]:
    action_instances = []
    for action in actions:
        action_instances.extend(instantiate_action(state, action, **kwargs))
    return action_instances


def static_instantiate_actions(
    state: State, actions: List[Action], verbose: bool = True
) -> List[ActionInstance]:
    start_time = current_time()
    if all(isinstance(action, ActionInstance) for action in actions):
        return actions
    fluents = get_fluent_functions(actions)
    action_instances = []
    for action in actions:
        for action_instance in instantiate_action(state, action, fluents=fluents):
            atoms = atoms_from_formula(action_instance.precondition, fluents=fluents)
            if all(state.holds(atom) for atom in atoms):
                action_instances.append(action_instance)
    if verbose:
        frequencies = Counter(instance.name for instance in action_instances)
        print(
            f"Static Instantiation) Instances ({len(action_instances)}): {dict(frequencies)} |"
            f" Elapsed: {elapsed_time(start_time):.3f} sec"
        )
    return action_instances


def relaxed_instantiate_actions(
    state: State, actions: List[Action], verbose: bool = True
) -> List[ActionInstance]:
    start_time = current_time()
    fluents = get_fluent_functions(actions)
    relaxed_state = OrderedSet(atoms_from_evaluations(state.evaluations))
    action_instances = OrderedSet(
        action for action in actions if isinstance(action, ActionInstance)
    )
    actions = [action for action in actions if isinstance(action, Action)]
    iteration = 0
    while actions:
        iteration += 1
        if verbose:
            print(
                f"Relaxed Instantiation) Iteration: {iteration}) Instances:"
                f" {len(action_instances)} | Evaluations: {len(relaxed_state)} | Elapsed:"
                f" {elapsed_time(start_time):.3f} sec"
            )
        augmented = False
        for action in actions:
            for action_instance in instantiate_action(relaxed_state, action, fluents=fluents):
                if action_instance in action_instances:
                    continue
                atoms = atoms_from_formula(action_instance.precondition, fluents=fluents)
                if not all(atom in relaxed_state for atom in atoms):
                    continue
                action_instances.add(action_instance)
                for effect in atoms_from_formula(action_instance.effect):
                    if effect not in relaxed_state:
                        relaxed_state.add(effect)
                        augmented = True
        if not augmented:
            break

    return list(action_instances)


def all_instantiate_actions(
    state: State, actions: List[Action], relaxed: bool = False
) -> List[ActionInstance]:
    if relaxed:
        return relaxed_instantiate_actions(state, actions)
    return static_instantiate_actions(state, actions)
