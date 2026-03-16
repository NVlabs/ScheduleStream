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
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Third Party
import numpy as np
from scipy import optimize
from scipy.optimize import linprog, milp

# NVIDIA
from schedulestream.common.utils import INF, compute_mapping, current_time, elapsed_time


@dataclass
class Variable:
    name: Any
    integer: bool = False
    lower: float = -np.inf
    upper: float = +np.inf


@dataclass
class Constraint:
    coefficients: Dict[Any, float]
    lower: float = -np.inf
    upper: float = +np.inf


@dataclass
class Cost:
    coefficients: Dict[Any, float]


def solve_lp(
    variables: List[Variable],
    constraints: Optional[List[Constraint]] = None,
    costs: Optional[List[Cost]] = None,
    maximize: bool = False,
    verbose: bool = False,
) -> Optional[Dict[str, float]]:
    start_time = current_time()
    assert variables
    if constraints is None:
        constraints = []
    if costs is None:
        costs = []

    var_names = [v.name for v in variables]
    idx_from_var = {name: i for i, name in enumerate(var_names)}
    V = len(variables)

    c = np.zeros(V)
    for cost in costs:
        for n, w in cost.coefficients.items():
            idx = idx_from_var[n]
            c[idx] += w
    if maximize:
        c *= -1

    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    for i, constraint in enumerate(constraints):
        row = np.zeros(V)
        for n, w in constraint.coefficients.items():
            idx = idx_from_var[n]
            row[idx] += w
        equality = constraint.lower == constraint.upper
        if equality:
            A_eq.append(row)
            b_eq.append(constraint.lower)
        else:
            if constraint.lower > -np.inf:
                A_ub.append(-row)
                b_ub.append(-constraint.lower)
            if constraint.upper < np.inf:
                A_ub.append(row)
                b_ub.append(constraint.upper)

    bounds = []
    for v in variables:
        assert not v.integer, v
        lower = v.lower
        if lower == -np.inf:
            lower = None
        upper = v.upper
        if upper == +np.inf:
            upper = None
        bounds.append((lower, upper))

    result = linprog(
        c,
        A_ub,
        b_ub,
        A_eq,
        b_eq,
        bounds=bounds,
        options=dict(disp=verbose),
        x0=None,
        integrality=None,
    )
    print(
        f"Success: {result.success} | Solve time: {elapsed_time(start_time):.3f} sec | Message:"
        f" {result.message}"
    )

    solution = result.x
    if solution is None:
        return solution
    return compute_mapping(var_names, solution)


def solve_milp(
    variables: List[Variable],
    constraints: Optional[List[Constraint]] = None,
    costs: Optional[List[Cost]] = None,
    maximize: bool = False,
    gap_percent: float = 1e-2,
    max_time: float = np.inf,
    verbose: bool = False,
) -> Optional[Dict[str, float]]:
    start_time = current_time()
    assert variables
    if constraints is None:
        constraints = []
    if costs is None:
        costs = []

    var_names = [v.name for v in variables]
    idx_from_var = {name: i for i, name in enumerate(var_names)}

    coefficient_names = set()
    for constraint in constraints:
        coefficient_names.update(constraint.coefficients)
    for cost in costs:
        coefficient_names.update(cost.coefficients)
    assert set(coefficient_names) <= set(var_names), set(coefficient_names) - set(var_names)

    V = len(variables)
    C = len(constraints)

    A = np.zeros([C, V])
    lb = -np.inf * np.ones(C)
    ub = +np.inf * np.ones(C)
    for i, constraint in enumerate(constraints):
        lb[i] = constraint.lower
        ub[i] = constraint.upper
        for n, w in constraint.coefficients.items():
            idx = idx_from_var[n]
            A[i, idx] += w
    linear = optimize.LinearConstraint(A=A, lb=lb, ub=ub)

    c = np.zeros(V)
    for cost in costs:
        for n, w in cost.coefficients.items():
            idx = idx_from_var[n]
            c[idx] += w

    integrality = np.zeros(V)
    lb = -np.inf * np.ones(V)
    ub = +np.inf * np.ones(V)
    for v in variables:
        idx = idx_from_var[v.name]
        integrality[idx] = int(v.integer)
        lb[idx] = v.lower
        ub[idx] = v.upper
    bounds = optimize.Bounds(lb=lb, ub=ub)

    if maximize:
        c *= -1
    if verbose:
        print(
            f"Solving MILP for {max_time:.3f} sec) Variables: {V} | Constraints: {C} | Setup time:"
            f" {elapsed_time(start_time):.3f} sec"
        )
    start_time = current_time()
    result = milp(
        c=c,
        constraints=linear,
        integrality=integrality,
        bounds=bounds,
        options=dict(time_limit=max_time, mip_rel_gap=gap_percent, disp=False),
    )
    if verbose:
        optimal = result.status in {0, 2, 3}
        objective = result.fun if result.fun is not None else INF
        print(
            f"Success: {result.success} | Objective: {objective:.3f} | Optimal: {optimal} | Solve"
            f" time: {elapsed_time(start_time):.3f} sec"
        )
    if result.status == 0:
        pass

    solution = result.x
    if solution is None:
        return solution
    return compute_mapping(var_names, solution)
