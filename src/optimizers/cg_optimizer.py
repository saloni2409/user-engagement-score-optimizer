
"""     Solves min (1/2 w^T Q w - c^T w) subject to w >= 0 and sum(w) = 1 using the 
    Projected Conjugate Gradient (CG) Method.

    Algorithm Description:
    The Conjugate Gradient method is an iterative algorithm for solving quadratic optimization
    problems. The standard version is for unconstrained problems. This implementation adapts it
    for constrained optimization by adding a projection step.

    The problem is:
        minimize      f(w) = 1/2 w^T Q w - c^T w
        subject to    sum(w) = 1
                      w >= 0

    1. Initialization:
       - Start with a feasible weight vector w_0 (e.g., uniform weights).
       - Compute the initial gradient: g_0 = Q*w_0 - c.
       - The initial search direction is the negative gradient: d_0 = -g_0.

    2. Iteration Steps:
       a. Step Size (alpha): For a quadratic function, the optimal step size alpha_k that
          minimizes f(w_k + alpha * d_k) can be calculated analytically:
              alpha_k = -g_k^T d_k / (d_k^T Q d_k)

       b. Update (Pre-projection): Take a step in the search direction:
              w_temp = w_k + alpha_k * d_k

       c. Projection: Project the temporary vector back onto the feasible set (the simplex)
          to ensure the constraints (sum=1, non-negative) are met:
              w_{k+1} = projection_on_simplex(w_temp)

       d. Update Gradient: Calculate the gradient at the new point:
              g_{k+1} = Q*w_{k+1} - c

       e. Beta (Fletcher-Reeves): Calculate beta, which is used to compute the new
          conjugate direction. This ensures the new direction is Q-orthogonal to the previous one.
              beta_{k+1} = (g_{k+1}^T g_{k+1}) / (g_k^T g_k)

       f. Update Search Direction: The new direction is a combination of the new negative
          gradient and the previous direction:
              d_{k+1} = -g_{k+1} + beta_{k+1} * d_k

       g. Repeat: Continue until the norm of the change in weights is below a tolerance.
 """

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Dict, List
from utility.utility import calculate_obj_function


# --- Constraint Projection Function (Simplex Projection) ---
def projection_on_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """
    Projects a vector v onto the probability simplex {w | w >= 0, sum(w) = z}.
    This is a crucial step for handling the constraints of the optimization problem.
    The algorithm finds the closest point in the simplex to the given vector v.
    """
    n = v.shape[0]
    # Sort the vector in descending order
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    
    # Find the largest index rho such that u[rho-1] > (cssv[rho-1] - z) / rho
    rho = np.max(np.where(u > (cssv - z) / np.arange(1, n + 1)))
    
    # The threshold theta is calculated based on this rho
    theta = (cssv[rho] - z) / (rho + 1)
    
    # The projection is then max(v - theta, 0)
    w = np.maximum(v - theta, 0)
    return w

def solve_with_conjugate_gradient(Q_solver: np.ndarray, c: np.ndarray, weight_names: List[str], max_iter: int = 1000, tolerance: float = 1e-7) -> Dict:
    K = len(weight_names)
    w = np.ones(K) / K  # Start with a uniform, feasible weight vector

    history = []
    start_time = time.time()

    # Initial gradient and search direction
    g = Q_solver @ w - c
    d = -g

    print("\n" + "="*80)
    print(f"Projected Conjugate Gradient Method Execution (K={K} Features)")
    print("Goal: Minimize g(w) = 1/2 w^T Q w - c^T w, subject to Simplex Constraints")
    print("="*80)

    print(f"--- Initial Setup ---")
    print(f"  > Initial Objective (Min g(w)): {calculate_obj_function(Q_solver, c, w):.6f}")
    print(f"  > Initial Weights: {', '.join([f'{name}: {w_val:.4f}' for name, w_val in zip(weight_names, w)])}")

    history.append({
        'iteration':  0,
        'objective_value': calculate_obj_function(Q_solver, c, w),
        'step_change': 0,
        'weights': w.copy()
    })
    for iteration in range(max_iter):
        # --- Store Iteration History ---
        obj_val = calculate_obj_function(Q_solver, c, w)
        
        # --- Step 1: Calculate Optimal Step Size (alpha) ---
        d_Q_d = d.T @ Q_solver @ d
        if d_Q_d <= 1e-12: # Avoid division by zero or tiny numbers
            print("\n[INFO] Curvature is near zero. Stopping.")
            break
        
        alpha = - (g.T @ d) / d_Q_d

        # --- Step 2: Update weights and project back to simplex ---
        w_prev = w
        w_temp = w + alpha * d
        w = projection_on_simplex(w_temp)
        
        # --- Convergence Check ---
        change = np.linalg.norm(w - w_prev)
        history.append({
            'iteration': iteration + 1,
            'objective_value': obj_val,
            'step_change': change,
            'weights': w.copy()
        })

        print(f"--- Iteration {iteration + 1:02d} ---")
        print(f"  > Current Objective: {obj_val:.6f}")
        print(f"  > Step Change: {change:.6g}")
        print(f"  > Current Weights: {', '.join([f'{name}: {w_val:.4f}' for name, w_val in zip(weight_names, w)])}")

        if change < tolerance:
            print(f"\n[INFO] CONVERGENCE: Step change ({change:.2e}) is below tolerance ({tolerance:.2e}).")
            break

        # --- Step 3: Update gradient, beta, and search direction ---
        g_prev = g
        g = Q_solver @ w - c
        
        beta = (g.T @ g) / (g_prev.T @ g_prev)
        d = -g + beta * d

    end_time = time.time()

    print(f"\n[INFO] Total Iterations: {len(history)}")
    print(f"[INFO] Time elapsed: {end_time - start_time:.4f} seconds")
    print(f"[INFO] Final Sum of Weights: {np.sum(w)}")
    print("="*80)

    final_obj = calculate_obj_function(Q_solver, c, w)
    results_df = pd.DataFrame({'Weight': w, 'Feature': weight_names})
    return {
        'method': 'projected_conjugate_gradient',
        'weights_df': results_df,
        'w_star': w,
        'status': 'Optimal (CG)',
        'max_f_w': -final_obj,
        'history': history
    }


def plot_cg_convergence(history: List[Dict], weight_names: List[str]):
    """
    Plots the objective value and weight convergence from CG history.
    """
    if not history:
        print("No iteration history available for plotting.")
        return

    iterations = [h['iteration'] for h in history]
    obj_values = [h['objective_value'] for h in history]
    step_changes = [h['step_change'] for h in history]
    weight_data = pd.DataFrame([h['weights'] for h in history], columns=weight_names)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Projected Conjugate Gradient Convergence Analysis')

    # --- Plot 1: Objective Value ---
    axes[0].plot(iterations, obj_values, marker='o', linestyle='-', color='darkgreen', markersize=4)
    axes[0].set_ylabel('Objective Value f(w)')
    axes[0].set_title('Objective Function Value per Iteration')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Step Change ---
    axes[1].plot(iterations, step_changes, marker='o', linestyle='-', color='darkred', markersize=4)
    axes[1].set_ylabel('Step Change ||w_k+1 - w_k||')
    axes[1].set_yscale('log')
    axes[1].set_title('Change in Weights per Iteration (Log Scale)')
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
