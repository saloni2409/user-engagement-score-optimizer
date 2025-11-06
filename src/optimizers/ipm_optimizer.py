"""
Implementation of a Primal-Dual Interior Point Method (IPM) for Quadratic Programming.

Algorithm Description:
The Interior Point Method (IPM) is a powerful algorithm for solving constrained optimization problems.
This implementation uses a primal-dual path-following approach to solve a quadratic program
of the form:

    minimize      f(w) = 1/2 w^T Q w - c^T w
    subject to    A w = b  (e.g., sum(w) = 1)
                  w >= 0

1. KKT Conditions:
   The Karush-Kuhn-Tucker (KKT) conditions form the basis for the algorithm. They are the
   necessary conditions for optimality and consist of:
   - Stationarity:           Q w - c + A^T y - z = 0
   - Primal Feasibility:     A w = b
   - Dual Feasibility:       z >= 0
   - Complementary Slackness: w_i * z_i = 0 for all i

   Here, 'y' is the dual variable for the equality constraint and 'z' are the dual variables
   (Lagrange multipliers) for the non-negativity constraints.

2. Path-Following with a Barrier Parameter (mu):
   The complementary slackness condition (w_i * z_i = 0) is relaxed to w_i * z_i = mu,
   where mu > 0 is the duality gap. The set of solutions for varying 'mu' forms the
   "central path". The algorithm follows this path towards mu -> 0.

3. Newton's Method for Root Finding:
   For a given 'mu', the algorithm uses Newton's method to solve the perturbed KKT system.
   This involves finding a search direction (dw, dy, dz) for the primal and dual variables
   by solving a linear system of equations derived from the Jacobian of the KKT system.
   This implementation uses the Schur complement method to solve this system efficiently.

4. Iteration Steps:
   a. Initialize: Start with a feasible point (w > 0, z > 0).
   b. Compute Duality Gap (mu) and Residuals: Calculate how far the current point is
      from satisfying the KKT conditions.
   c. Solve for Newton Step: Solve the linear system for the search directions (dw, dy, dz).
   d. Line Search: Determine a step size that keeps the variables (w, z) positive.
   e. Update Variables: Update w, y, and z using the calculated step direction and step size.
   f. Repeat: Continue until the duality gap is below a specified tolerance.
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import List, Dict
from utility.utility import calculate_obj_function

def solve_with_ipm(Q_solver: np.ndarray, c: np.ndarray, weight_names: List[str], max_iter: int = 100, tolerance: float = 1e-9) -> Dict:
    """
    Solves the quadratic program using the Primal-Dual Interior Point Method.

    Args:
        Q_solver: The quadratic matrix (Hessian).
        c: The linear vector.
        weight_names: The names of the features/weights.
        max_iter: The maximum number of iterations.
        tolerance: The convergence tolerance for the duality gap.

    Returns:
        A dictionary containing the optimization results, including optimal weights and history.
    """
    K = len(weight_names)
    A = np.ones((1, K))
    b = np.array([1.0])

    # --- Initial Feasible Point ---
    w = np.ones(K) / K
    y = np.zeros(A.shape[0])
    z = np.ones(K) / K

    history = []
    start_time = time.time()

    # IPM parameters
    sigma = 0.2  # Centering parameter (0 < sigma < 1)

    print("\n" + "="*80)
    print(f"Primal-Dual Interior Point Method Execution (K={K} Features)")
    print("Goal: Minimize g(w) = 1/2 w^T Q w - c^T w, subject to Simplex Constraints")
    print("="*80)
    print(f"--- Initial Setup ---")
    print(f"  > Initial Objective (Min g(w)): {calculate_obj_function(Q_solver, c, w):.6f}")
    print(f"  > Initial Weights: {', '.join([f'{name}: {w_val:.4f}' for name, w_val in zip(weight_names, w)])}")
    history.append({
        'iteration': 0,
        'objective_value': calculate_obj_function(Q_solver, c, w),
        'duality_gap': np.dot(w, z) / K, # Total gap
        'weights': w.copy()
    })

    for iteration in range(max_iter):
        # --- Calculate Duality Gap and Residuals ---
        mu = np.dot(w, z) / K
        r_dual = Q_solver @ w - c + A.T @ y - z
        r_primal = A @ w - b
        
        obj_val = calculate_obj_function(Q_solver, c, w)
        history.append({
            'iteration': iteration + 1,
            'objective_value': obj_val,
            'duality_gap': mu * K, # Total gap
            'weights': w.copy()
        })

        print(f"--- Iteration {iteration + 1:02d} ---")
        print(f"  > Current Objective: {obj_val:.6f}")
        print(f"  > Duality Gap: {mu * K:.6g}")
        print(f"  > Current Weights: {', '.join([f'{name}: {w_val:.4f}' for name, w_val in zip(weight_names, w)])}")

        if (mu * K) < tolerance:
            print(f"\n[INFO] CONVERGENCE: Duality gap ({mu*K:.2e}) is below tolerance ({tolerance:.2e}).")
            break

        # --- Form and Solve the Newton System (KKT System) ---
        # This is a corrected and more robust way to solve the linear system
        
        # Target for the complementary slackness condition
        target_comp = sigma * mu
        
        try:
            # Define the components of the KKT matrix
            H_bar = Q_solver + np.diag(z / w)
            
            # Form the Schur complement system to solve for dy first
            # S = A @ inv(H_bar) @ A.T
            inv_H_A_T = np.linalg.solve(H_bar, A.T)
            S = A @ inv_H_A_T
            
            # Form the right-hand side for the 'dy' system
            rhs_dw_part = -r_dual + (target_comp - w*z)/w
            rhs_y = -r_primal + A @ np.linalg.solve(H_bar, rhs_dw_part)

            # Solve for dy, then back-substitute for dw and dz
            dy = np.linalg.solve(S, rhs_y)
            dw = np.linalg.solve(H_bar, rhs_dw_part - A.T @ dy)
            dz = (target_comp - w*z - z*dw) / w

        except np.linalg.LinAlgError:
            print(f"\n[ERROR] Iteration {iteration}: Matrix is singular. Cannot compute Newton step.")
            break

        # --- Line Search to maintain w > 0, z > 0 ---
        step_size = 1.0
        # Reduce step size to stay in the interior (w > 0, z > 0)
        while np.min(w + step_size * dw) <= 0:
            step_size *= 0.8
        while np.min(z + step_size * dz) <= 0:
            step_size *= 0.8
        
        # --- Update Variables ---
        w += step_size * dw
        y += step_size * dy
        z += step_size * dz

    end_time = time.time()

    print(f"\n[INFO] Total Iterations: {len(history)}")
    print(f"[INFO] Time elapsed: {end_time - start_time:.4f} seconds")
    print(f"[INFO] Final Sum of Weights: {np.sum(w)}")
    print("="*80)

    results_df = pd.DataFrame({'Weight': w, 'Feature': weight_names})
    return {
        'method': 'interior_point_method',
        'weights_df': results_df,
        'w_star': w,
        'status': 'Optimal (IPM)',
        'max_f_w': -obj_val,
        'history': history
    }

def plot_ipm_convergence(history: List[Dict], weight_names: List[str]):
    """
    Plots the objective value, duality gap, and weight convergence from IPM history.
    """
    if not history:
        print("No iteration history available for plotting.")
        return

    iterations = [h['iteration'] for h in history]
    obj_values = [h['objective_value'] for h in history]
    duality_gaps = [h['duality_gap'] for h in history]
    weight_data = pd.DataFrame([h['weights'] for h in history], columns=weight_names)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Interior Point Method Convergence Analysis')

    # --- Plot 1: Objective Value ---
    axes[0].plot(iterations, obj_values, marker='o', linestyle='-', color='blue', markersize=4)
    axes[0].set_ylabel('Objective Value f(w)')
    axes[0].set_title('Objective Function Value per Iteration')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Duality Gap ---
    axes[1].plot(iterations, duality_gaps, marker='o', linestyle='-', color='red', markersize=4)
    axes[1].set_ylabel('Duality Gap (mu)')
    axes[1].set_yscale('log')
    axes[1].set_title('Duality Gap per Iteration (Log Scale)')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 3: Weight Convergence ---
    for name in weight_names:
        axes[2].plot(iterations, weight_data[name], linestyle='-', label=name)
    
    axes[2].set_xlabel('Iteration Number')
    axes[2].set_ylabel('Weight Value')
    axes[2].set_title('Individual Weight Convergence')
    axes[2].legend(loc='best', fontsize='small')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()