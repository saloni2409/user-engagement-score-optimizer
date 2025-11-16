"""
Implementation of a fully constrained Primal-Dual Interior Point Method (IPM) for Quadratic Programming.

This version includes three types of constraints:
1. Equality (Normalization): A w = b
2. Simple Bounds (Non-Negativity): w >= 0 (via dual variable z)
3. General Linear Inequality (Hierarchy): G w <= h (via dual variable mu and slack s)
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import List, Dict
from utility.utility import calculate_obj_function

def solve_with_ipm_complete(Q_solver: np.ndarray, c: np.ndarray, weight_names: List[str], max_iter: int = 100, tolerance: float = 1e-9) -> Dict:
    """
    Solves the quadratic program using the Primal-Dual Interior Point Method,
    including equality, non-negativity, and hierarchy constraints.
    """
    K = len(weight_names)
    
    # 1. DEFINE ALL CONSTRAINTS
    # Equality Constraint (Normalization: sum(w) = 1)
    A = np.ones((1, K))
    b = np.array([1.0])
    M = A.shape[0] # Number of equality constraints (M=1)

    # Hierarchy Constraints (Inequalities: purchase > add_to_cart > add_to_wishlist > product_view)
    # Mapping: w = [w1, w2, w3, w4, w5] = [add_to_cart, add_to_wishlist, product_view, purchase, view_cart]
    # C1: w4 >= w1  => w1 - w4 <= 0
    # C2: w1 >= w2  => w2 - w1 <= 0
    # C3: w2 >= w3  => w3 - w2 <= 0
    G = np.array([
        [ 1,  0,  0, -1,  0],  
        [-1,  1,  0,  0,  0],
        [ 0, -1,  1,  0,  0]
    ])
    h = np.zeros(3)
    J = G.shape[0] # Number of hierarchy constraints (J=3)


    # --- 2. Initial Feasible Point (Primal and Dual Variables) ---
    # Primal Variables:
    w = np.ones(K) / K  # Weights
    s = h - G @ w       # Hierarchy Slack variables (s > 0)
    
    # Dual Variables:
    y = np.zeros(M)     # Dual for Normalization
    z = np.ones(K)      # Dual for Non-Negativity (w >= 0)
    mu_hier = np.ones(J) # Dual for Hierarchy (G w <= h)

    # Re-normalize initial z and mu to ensure w^T z + s^T mu_hier is positive
    mu_gap_init = (np.dot(w, z) + np.dot(s, mu_hier))
    z = z * (0.1 / mu_gap_init)
    mu_hier = mu_hier * (0.1 / mu_gap_init)
    

    history = []
    start_time = time.time()
    sigma = 0.2  # Centering parameter
    
    print("\n" + "="*80)
    print(f"Primal-Dual Interior Point Method Execution (K={K} Features, J={J} Hierachy Constraints)")
    print("="*80)

    for iteration in range(max_iter):
        # --- 3. Calculate Duality Gap and Target Gap ---
        duality_gap = (np.dot(w, z) + np.dot(s, mu_hier)) / (K + J)
        if duality_gap < tolerance:
            print(f"Converged! Duality gap ({duality_gap:.2e}) below tolerance at iteration {iteration}.")
            break
        
        target_gap = sigma * duality_gap # Centering parameter adjustment

        # --- 4. Define Residuals (The Perturbed KKT System) ---
        # The KKT system is solved for the step (dw, dy, dz, ds, dmu_hier)
        
        # Dual Residual (Stationarity): Q w - c + A^T y - z + G^T mu_hier = 0
        r_dual = Q_solver @ w - c + A.T @ y - z + G.T @ mu_hier
        
        # Primal Residual (Normalization Feasibility): A w = b
        r_primal_norm = A @ w - b

        # Primal Residual (Hierarchy Feasibility): G w + s = h
        r_primal_hier = G @ w + s - h 
        
        # Complementary Slackness Residuals (Perturbed): W Z 1 = mu, S Mu_hier 1 = mu
        # Note: We compute the full RHS term including the affine scaling part (y* = mu/z_i - w_i)
        r_comp_wz = w * z - target_gap
        r_comp_smu = s * mu_hier - target_gap

        # --- 5. Solve for Newton Step (The Linear System) ---
        
        # Define diagonal matrices for the IPM linear system
        epsilon = 1e-8  # Small value to prevent division by zero
        W_inv = np.diag(1 / (w + epsilon))
        S_inv = np.diag(1 / (s + 1e-8))  # Add epsilon to avoid division by zero
        Z = np.diag(z)
        Mu_hier = np.diag(mu_hier)

        # Simplified term M = Q + Z W_inv. This is the main term of the reduced system
        M_wz = Q_solver + Z @ W_inv
        
        # Simplified term M_smu = S_inv Mu_hier. This is the term for the slack variables
        M_smu = S_inv @ Mu_hier

        # The KKT matrix is large. We use the Schur Complement method to solve a reduced system.
        
        # RHS for the reduced system involving (dw, dy, dmu_hier)
        # Note: We incorporate the complementary slackness terms into the RHS of the dual residual
        r_dual_hat = r_dual + W_inv @ r_comp_wz - G.T @ S_inv @ r_comp_smu
        
        # System matrix for (dw, dy, dmu_hier)
        # This is a block matrix that requires careful construction:
        # T = [ M_wz   A^T     G^T ]
        #     [ A      0       0   ]
        #     [ G      0     -M_smu]
        # Since M_smu is a diagonal matrix, we can use the structure to simplify.
        
        # The full linear system matrix for (dw, dy, dmu_hier)
        # Block 1 (K x K): M_wz + G^T (Mu_hier S_inv) G
        # Block 2 (K x M): A^T
        # Block 3 (K x J): G^T
        
        # The most efficient way is to solve a reduced system, often by inverting M_wz
        # For simplicity and correctness: solve the full system T [dw; dy; dmu] = RHS

        
        # T matrix construction (Matrix size (K+M+J) x (K+M+J))
        T11 = M_wz
        T12 = A.T
        T13 = G.T
        T21 = A
        T22 = np.zeros((M, M))
        T23 = np.zeros((M, J))
        T31 = G
        T32 = np.zeros((J, M))
        T33 = -M_smu # The minus sign comes from the slack update dmu = S_inv * (r_smu - Mu_hier * ds)
        
        Top = np.hstack([T11, T12, T13])
        Mid = np.hstack([T21, T22, T23])
        Bot = np.hstack([T31, T32, T33])
        T = np.vstack([Top, Mid, Bot])

        # RHS vector construction
        RHS = -np.hstack([r_dual_hat, r_primal_norm, r_primal_hier])
        
        # Solve the linear system
        try:
            solution = np.linalg.solve(T, RHS)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered. IPM failed to find a search direction.")
            break
            
        dw = solution[:K]
        dy = solution[K:K+M]
        dmu_hier = solution[K+M:K+M+J]
        
        # Compute remaining dual steps from the Schur-reduced equations
        dz = -Z @ (dw + W_inv @ r_comp_wz)
        ds = -S_inv @ (r_comp_smu + Mu_hier @ dmu_hier)
        
        # --- 6. Line Search and Update ---
        
        # Calculate maximum step size (alpha) to maintain w > 0, s > 0, z > 0, mu_hier > 0
        def max_step_size(v, dv):
            # Only consider elements where dv < 0 and v > 0 to avoid division by zero or negative step sizes
            v = np.array(v)
            dv = np.array(dv)
            valid_idx = (dv < 0) & (v > 1e-12)  # Filter out non-positive v
            if not np.any(valid_idx):
                return 1.0
            # Avoid division by zero
            safe_dv = dv[valid_idx]
            safe_v = v[valid_idx]
            nonzero_mask = safe_dv != 0
            if not np.any(nonzero_mask):
                return 1.0
            alpha_max = np.min(-safe_v[nonzero_mask] / safe_dv[nonzero_mask])
            return min(1.0, alpha_max)

        # Affine step (uncentered)
        alpha_max_wz = max_step_size(np.hstack([w, s]), np.hstack([dw, ds]))
        alpha_max_mu = max_step_size(np.hstack([z, mu_hier]), np.hstack([dz, dmu_hier]))
        alpha_max_mu = max_step_size(np.hstack([z, mu_hier]), np.hstack([dz, dmu_hier]))
        
        # Apply step reduction (0.95 factor) for safety and to stay strictly interior
        alpha = 0.95 * min(alpha_max_wz, alpha_max_mu)

        # Update all variables
        w += alpha * dw
        y += alpha * dy
        z += alpha * dz
        s += alpha * ds
        mu_hier += alpha * dmu_hier

        # --- 7. History and Reporting ---
        history.append({
            'iteration': iteration + 1,
            'objective_value': calculate_obj_function(Q_solver, c, w),
            'duality_gap': duality_gap,
            'weights': w.copy()
        })
        
        # Reporting output (reduced frequency for conciseness)
        if (iteration + 1) % 10 == 0 or iteration == 0 or iteration == max_iter - 1:
            print(f"Iter {iteration + 1:3d} | Gap: {duality_gap:.2e} | Obj: {calculate_obj_function(Q_solver, c, w):.4e} | Alpha: {alpha:.3f}")

    end_time = time.time()
    
    # Final Result Formatting
    optimal_weights = {name: weight for name, weight in zip(weight_names, w)}
    
    print("\n" + "="*80)
    print(f"Optimization Complete in {end_time - start_time:.4f} seconds.")
    print(f"Final Weights: {optimal_weights}")
    print("="*80)

    results_df = pd.DataFrame({
            'Event': weight_names,
            'Optimal_Weight': w
        })
    return {
        'method': 'interior_point_method',
        'weights_df': results_df,
        'w_star': w,
        'status': 'Converged' if duality_gap < tolerance else 'Max Iterations Reached',
        'max_f_w': - calculate_obj_function(Q_solver, c, w),
        'history': history
    }


