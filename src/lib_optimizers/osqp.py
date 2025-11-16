from typing import List, Optional, Tuple, Union
import numpy as np
import osqp
from scipy.sparse import csc_matrix
import pandas as pd
from typing import Dict, Any


# Extracted helper to build constraint matrices (supports optional user-specified hierarchy)
def create_constraints(K: int,
                hierarchy: Optional[
                List[Union[
                    Tuple[Union[int, str], Union[int, str]],
                    Tuple[Union[int, str], Union[int, str], str]
                ]]
                ] = None,
                names: List[str] = None):
    
    """
    Build A_final, l_final, u_final for OSQP:
        - normalization: sum(w) = 1
        - optional hierarchy: list of (a, b) or (a, b, relation) where relation is one of
        '>=', '<=', '==', '>', '<', 'ge', 'le', 'eq' (default for a pair is '>=')
    Interpretation: (a, b, '>=') means weight[a] >= weight[b]
    a/b can be either integer indices (0-based) or names present in `names`.
        - non-negativity: w >= 0
    """
    if names is None:
        raise ValueError("create_constraints requires `names` to resolve string indices")

    if K != len(names):
        raise ValueError("K must equal len(names)")

    name_to_idx = {n: i for i, n in enumerate(names)}

    def resolve_index(x):
        if isinstance(x, str):
            if x not in name_to_idx:
                raise ValueError(f"Unknown weight name: {x}")
            return name_to_idx[x]
        try:
            ix = int(x)
        except Exception:
            raise ValueError(f"Index must be int or valid name, got: {x}")
        if ix < 0 or ix >= K:
            raise IndexError(f"Index out of range: {ix}")
        return ix

    rows = []
    l_list = []
    u_list = []

    # Normalization: sum(w) == 1
    rows.append(np.ones(K, dtype=float))
    l_list.append(1.0)
    u_list.append(1.0)

    # Hierarchy: user-specified 
    if hierarchy is None:
        hierarchy = []
    else:
        for item in hierarchy:
            if not (isinstance(item, (list, tuple)) and 2 <= len(item) <= 3):
                raise ValueError(f"Invalid hierarchy item: {item}")
            a, b = item[0], item[1]
            rel = item[2] if len(item) == 3 else '>='
            ia = resolve_index(a)
            ib = resolve_index(b)
            row = np.zeros(K, dtype=float)
            row[ia] = 1.0
            row[ib] = -1.0

            # normalize relation string
            rel_norm = str(rel).strip().lower()
            if rel_norm in ('>=', 'ge', '=>'):
                l_val, u_val = 0.0, np.inf
            elif rel_norm in ('<=', 'le', '=<'):
                l_val, u_val = -np.inf, 0.0
            elif rel_norm in ('==', '=', 'eq'):
                l_val, u_val = 0.0, 0.0
            elif rel_norm == '>':
                # strict not supported numerically; treat as >=
                l_val, u_val = 0.0, np.inf
            elif rel_norm == '<':
                l_val, u_val = -np.inf, 0.0
            else:
                raise ValueError(f"Unsupported relation in hierarchy: {rel}")

            rows.append(row)
            l_list.append(l_val)
            u_list.append(u_val)

    # Non-negativity bounds as additional rows: 0 <= w_k <= inf
    if len(rows) == 0:
        A_top = np.zeros((0, K), dtype=float)
    else:
        A_top = np.vstack(rows)

    A_bounds = np.eye(K, dtype=float)
    l_bounds = np.zeros(K, dtype=float)
    u_bounds = np.full(K, np.inf)

    A_final = csc_matrix(np.vstack([A_top, A_bounds]))
    l_final = np.hstack([np.array(l_list, dtype=float), l_bounds])
    u_final = np.hstack([np.array(u_list, dtype=float), u_bounds])

    return A_final, l_final, u_final

def minimizeUsingOsQP(Q_matrix: np.ndarray,
                  C_vector: np.ndarray,
                  weight_names: List[str],
                  hierarchy: Optional[
                List[Union[
                    Tuple[Union[int, str], Union[int, str]],
                    Tuple[Union[int, str], Union[int, str], str]
                ]]
                ] = None) -> Dict[str, Any]:
    """
    Solve the quadratic optimization problem using OSQP (ADMM method):
    min_w 0.5 * w^T Q w - C^T w
    subject to:
      - sum(w) = 1
      - w4 >= w1 (purchase >= product_view)
      - w1 >= w2 (product_view >= add_to_cart)                  
      - w2 >= w3 (add_to_cart >= add_to_wishlist)                 
      - w >= 0
    Args:
        Q_matrix (np.ndarray): The Q matrix in the quadratic term.
        C_vector (np.ndarray): The C vector in the linear term.
        weight_names (List[str]): List of weight names corresponding to the weights.    
    Returns:
        Dict[str, Any]: A dictionary containing optimal weights, minimum objective value, and status.
    """


    # 2. OSQP Formulation (QP must be min 1/2 x' P x + q' x subject to l <= A x <= u)

    # 2a. Define P (Hessian Matrix) and q (Linear Vector)
    # Your problem: min w' Q w - C' w
    # OSQP formulation: min 1/2 w' P w + q' w
    # Therefore: P = 2 * Q and q = -C
    P = 2 * Q_matrix
    q = -C_vector

    # Prepare problem-specific names/sizes
    K = len(weight_names)
    Event_Names = weight_names

    # Create constraint matrices via extracted function (pass in the hierarchy parameter)
    A_final, l_final, u_final = create_constraints(K, hierarchy, Event_Names)

    # 3. Setup and Solve using OSQP (ADMM Implementation)

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    # P must be sparse (csc_matrix)
    prob.setup(P=csc_matrix(P), q=q, A=A_final, l=l_final, u=u_final, verbose=False)

    # Solve problem
    results = prob.solve()

    # 4. Extract Results and Check Constraints
    w_osqp = results.x
    min_objective_value_osqp = results.info.obj_val

    weights_output = list(zip(Event_Names, w_osqp))

    print(f"OSQP (ADMM) Optimal Weights (w*):\n{weights_output}")
    print(f"OSQP Minimum Objective Value: {min_objective_value_osqp}")
    print(f"OSQP Status: {results.info.status}")
    maximized_objective_value = -min_objective_value_osqp
    return {
        'method': 'Dual Ascent (ADMM)',
        'weights_df': pd.DataFrame(weights_output, columns=['Weight_Name', 'Optimal_Weight']),
        'w_star': w_osqp,
        'status': results.info.status,
        'max_f_w': maximized_objective_value
    }


