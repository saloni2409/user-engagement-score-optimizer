import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from typing import Optional, Sequence, List, Union, Tuple, Dict, Any

from utility.utility import objective_function


def build_constraints(k: int,
                      weight_names: Optional[List[str]] = None,
                      hierarchy: Optional[List[Union[Tuple[Union[int,str], Union[int,str]],
                                            Tuple[Union[int,str], Union[int,str], str]]]] = None):
    """
    Build normalization equality constraint, hierarchy inequality constraints (G w <= 0),
    and non-negativity bounds for k weights.

    - weight_names: optional list of names for resolving string references in hierarchy.
    - hierarchy: list of relations. Each entry can be:
        * (higher, lower)  -> interpreted as higher >= lower
        * (a, b, op)       -> op in ('>=', '<='), where a,b are indices or names
    Returns: (normalization_constraint, hierarchy_constraint, bounds)
    """
    if weight_names is None:
        weight_names = [str(i) for i in range(k)]
    if hierarchy is None:
        hierarchy = []

    # Normalization: sum(w) = 1
    A_eq = np.ones((1, k))
    b_eq = np.array([1.0])
    normalization_constraint = LinearConstraint(A_eq, b_eq, b_eq)

    # Build hierarchy matrix G such that G @ w <= 0
    G_rows = []
    def resolve_index(x):
        if isinstance(x, str):
            return weight_names.index(x)
        return int(x)

    for rel in hierarchy:
        if len(rel) == 2:
            a, b = rel
            op = '>='  # default interpretation
        else:
            a, b, op = rel
        ai = resolve_index(a)
        bi = resolve_index(b)
        row = np.zeros(k)
        if op == '>=':
            # a >= b  -> b - a <= 0
            row[bi] = 1
            row[ai] = -1
        elif op == '<=':
            # a <= b -> a - b <= 0
            row[ai] = 1
            row[bi] = -1
        else:
            raise ValueError("Unsupported operator in hierarchy; use '>=' or '<='")
        G_rows.append(row)

    if G_rows:
        G_hierarchy = np.vstack(G_rows)
        h_hierarchy = np.zeros(G_hierarchy.shape[0])
    else:
        # Empty constraint (no hierarchy): shape (0, k)
        G_hierarchy = np.zeros((0, k))
        h_hierarchy = np.zeros(0)

    # Convert hierarchy bounds to plain Python sequences of floats so static type checkers accept them
    if h_hierarchy.size == 0:
        lb_hierarchy = []
        ub_hierarchy = []
    else:
        lb_hierarchy = list((-np.inf * np.ones_like(h_hierarchy)).astype(float))
        ub_hierarchy = list(h_hierarchy.astype(float))

    hierarchy_constraint = LinearConstraint(G_hierarchy, lb_hierarchy, ub_hierarchy)

    # Non-negativity bounds
    bounds = [(0, None) for _ in range(k)]

    return normalization_constraint, hierarchy_constraint, bounds


def solve_constrained_quadratic(Q: np.ndarray,
                                C: np.ndarray,
                                weight_names: Optional[List[str]] = None,
                                hierarchy: Optional[List[Union[Tuple[Union[int,str], Union[int,str]],Tuple[Union[int,str], Union[int,str], str]]]] = None,
                                method: str = 'SLSQP',
                                w0: Optional[np.ndarray] = None,
                                options: Optional[Dict[str, Any]] = None):
    """
    Solve min_w w^T Q w - C^T w subject to:
      - sum(w) = 1
      - hierarchy relations (if provided)
      - w >= 0

    Returns the scipy OptimizeResult.
    """
    k = Q.shape[0]
    normalization_constraint, hierarchy_constraint, bounds = build_constraints(k, weight_names, hierarchy)
    print(hierarchy_constraint)
    if w0 is None:
        w0 = np.ones(k) / k
    if options is None:
        options = {'disp': False}

    result = minimize(
        objective_function,
        w0,
        args=(Q, C),
        method=method,
        constraints=[normalization_constraint, hierarchy_constraint],
        bounds=bounds,
        options=options
    )
    results_df = pd.DataFrame({'Weight': result.x, 'Feature': weight_names})
    maximized_objective_value = -result.fun

    return {
    'method': 'QP Solver (' + method + ')',
    'weights_df': results_df,
    'w_star': result.x,
    'status': result.status,
    'max_f_w': maximized_objective_value
}
