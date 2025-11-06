import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt # New import for plotting
from typing import List, Dict

# # --- Helper Function: Data Simulation and Transformation ---

# def simulate_and_transform_data(raw_data_df: pd.DataFrame, lambda_1: float, lambda_2: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#     """
#     Transforms the event log format (long format) into the required
#     wide User-Item-Feature matrix and computes Q and c for the QP/Newton solver.
    
#     Args:
#         raw_data_df: DataFrame with columns 'ucid', 'product_id', 'eventType', 'event_count'.
#         lambda_1: Fairness penalty weight.
#         lambda_2: Regularization penalty weight.

#     Returns:
#         Q_solver: The quadratic matrix (Hessian).
#         c: The linear vector (Engagement).
#         weight_names: The names of the features/event types.
#     """
#     print("--- 1. Pivoting Raw Event Log Data ---")
    
#     # Pivot the data from long format (event log) to wide format (User-Item-Feature Matrix)
#     wide_df = raw_data_df.pivot_table(
#         index=['ucid', 'product_id'],
#         columns='eventType',
#         values='event_count',
#         fill_value=0 # Users who didn't perform an event for a product get a count of 0
#     )
    
#     # The column names are now the feature/event types
#     weight_names = wide_df.columns.tolist()
#     K = len(weight_names)
#     print(f"Features (Event Types) identified (K={K}): {weight_names}")
    
#     # 2. Transformation (e_ij^k = log(1 + E_ij^k))
#     A = np.log1p(wide_df.values)
    
#     # 3. User Aggregation (for Fairness Term)
#     print("--- 2. Aggregating by User for Covariance Matrix ---")
#     B_df = pd.DataFrame(A, index=wide_df.index).groupby(level='ucid').sum()
#     B = B_df.values
    
#     # 4. Final Structure for Minimization (min 1/2 w^T Q w - c^T w)
    
#     # --- Linear Component (c): Total Engagement ---
#     c = A.sum(axis=0) 
#     print(f"Linear Vector c (Total Engagement Sums): {c}")
    
#     # --- Quadratic Component (Q): Fairness + Regularization ---
#     Cov_B = np.cov(B, rowvar=False) # Covariance of user-aggregated features
#     I = np.identity(K)
    
#     # Q = 2 * (lambda_1 * Cov_B + lambda_2 * I)
#     Q_solver = 2 * (lambda_1 * Cov_B + lambda_2 * I)
#     print(f"Quadratic Matrix Q (Hessian) calculated. Shape: {Q_solver.shape}")

#     # Ensure c vector is a flat array for clean dot products
#     if c.ndim > 1:
#         c = c.flatten()

#     return Q_solver, c, weight_names

# --- Constraint Projection Function (Simplex Projection) ---

def projection_on_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """
    Projection of vector v onto the probability simplex {w | w >= 0, sum(w) = z}.
    """
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssum = np.cumsum(u)
    
    # Find rho (the optimal number of positive weights)
    rho = 0
    for j in range(n):
        theta = (cssum[j] - z) / (j + 1)
        if u[j] > theta:
            rho = j + 1
        else:
            break
            
    # Calculate threshold theta for the identified rho
    theta = (cssum[rho - 1] - z) / rho
    
    # Apply thresholding
    w = np.maximum(v - theta, 0)
    return w

# --- Projected Newton's Method Solver (Verbose) ---

def solve_with_projected_newtons_method(Q_solver: np.ndarray, c: np.ndarray, weight_names: List[str], max_iter: int = 50, tolerance: float = 1e-6) -> Dict:
    """
    Solves min (1/2 w^T Q w - c^T w) subject to w >= 0 and sum(w) = 1 using 
    Projected Newton's Method. Tracks and returns iteration history.
    """
    K = len(weight_names)
    # Start with a uniform weight vector
    w = np.ones(K) / K
    
    # List to store iteration history for plotting
    history = [] 
    
    # Define the objective function (minimization form)
    def objective(w_vec):
        if w_vec.ndim == 1:
             w_vec = w_vec.reshape(-1, 1)

        return 0.5 * w_vec.T @ Q_solver @ w_vec - c.T @ w_vec.flatten()

    print("\n" + "="*80)
    print(f"Projected Newton's Method Execution (K={K} Features)")
    print("Goal: Minimize g(w) = 1/2 w^T Q w - c^T w, subject to Simplex Constraints")
    print("="*80)
    print(f"--- Initial Setup ---")
    print(f"  > Initial Objective (Min g(w)): {objective(w).item():.6f}")
    print(f"  > Initial Weights: {', '.join([f'{name}: {w_val:.4f}' for name, w_val in zip(weight_names, w)])}")
    history.append({
        'iteration':  0,
        'objective_value': objective(w).item(),
        'step_change': 0,
        'weights': w.copy()
    })

    # Add stabilization to Q for inversion (ensures non-singularity)
    Q_stable = Q_solver + np.eye(K) * 1e-8
    
    start_time = time.time()
    
    for iteration in range(max_iter):
        
        # --- Algorithm Step 1: Calculate Gradient ---
        g = Q_solver @ w - c 
        grad_norm = np.linalg.norm(g)
        
        # --- Algorithm Step 2: Solve for Newton Step Direction (p) ---
        try:
            # Solves the linear system Q * p = -g
            p = np.linalg.solve(Q_stable, -g)
        except np.linalg.LinAlgError:
            print(f"\n[ERROR] Iteration {iteration}: Q matrix is singular. Cannot compute Newton step.")
            break
        
        # --- Algorithm Step 3: Descent Step ---
        alpha = 1.0 
        w_temp = w + alpha * p
        
        # --- Algorithm Step 4: Projection onto the Simplex ---
        w_next = projection_on_simplex(w_temp)
        
        # Convergence Check
        change = np.linalg.norm(w_next - w)
        w = w_next
        
        # Calculate objective value for logging
        obj_val = objective(w).item()
        
        # --- Logging Iteration History ---
        history.append({
            'iteration': iteration + 1,
            'objective_value': obj_val,
            'step_change': change,
            'weights': w.copy()
        })
        
        print(f"--- Iteration {iteration + 1:02d} ---")
        print(f"  > Current Objective (Min g(w)): {obj_val:.6f}")
        print(f"  > Gradient Norm: {grad_norm:.6f}")
        print(f"  > Step Change (|w_next - w|): {change:.6f}")
        print(f"  > Current Weights: {', '.join([f'{name}: {w_val:.4f}' for name, w_val in zip(weight_names, w)])}")
        
        if change < tolerance:
            print("\n[INFO] CONVERGENCE: Step change is below tolerance.")
            break

    end_time = time.time()
    
    print(f"\n[INFO] Total Iterations: {iteration + 1}")
    print(f"[INFO] Time elapsed: {end_time - start_time:.4f} seconds")
    print("="*80)
    
    # Return results including the full history
    results_df = pd.DataFrame({'Weight': w, 'Feature': weight_names})
    return {
        'method': 'newtons_method',
        'weights_df': results_df,
        'w_star': w,
        'status': 'Optimal (Newton)',
        'max_f_w': -objective(w).item(), # Maximize negative objective
        'history': history # Return the history for plotting
    }

# --- Plotting Function ---

def plot_convergence(history: List[Dict], weight_names: List[str]):
    """
    Plots the objective value and the individual weight values over iterations.
    """
    if not history:
        print("No iteration history available for plotting.")
        return

    iterations = [h['iteration'] for h in history]
    obj_values = [h['objective_value'] for h in history]
    
    # Extract weights for each iteration
    weight_data = pd.DataFrame([h['weights'] for h in history], columns=weight_names)
    
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Projected Newton\'s Method Convergence Analysis')

    # --- Plot 1: Objective Value Convergence ---
    axes[0].plot(iterations, obj_values, marker='o', linestyle='-', color='indigo', markersize=4)
    axes[0].set_ylabel('Objective Value $g(w)$')
    axes[0].set_title('Objective Function Value per Iteration')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Weight Convergence ---
    for name in weight_names:
        axes[1].plot(iterations, weight_data[name], linestyle='-', label=name)
        
    axes[1].set_xlabel('Iteration Number')
    axes[1].set_ylabel('Weight Value $w_k$')
    axes[1].set_title('Individual Weight Convergence')
    axes[1].legend(loc='best', fontsize='small')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

""" 
# --- Main Execution Block ---
if __name__ == '__main__':
    
    # --- SIMULATE PARQUET-LIKE INPUT DATA ---
    np.random.seed(42)
    sample_data = {
        'ucid': [f'U{i}' for i in range(10)] * 3 + ['U1'] * 2 + ['U2'] * 2,
        'product_id': ['P1'] * 10 + ['P2'] * 10 + ['P3'] * 10 + ['P1', 'P2'] + ['P1', 'P3'],
        'eventType': ['product_view'] * 10 + ['add_to_cart'] * 10 + ['purchase'] * 10 + ['product_view', 'add_to_cart'] + ['purchase', 'product_view'],
        'event_count': np.random.randint(1, 40, size=34)
    }
    raw_df = pd.DataFrame(sample_data).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print("--- Sample Input Data (Long/Event Log Format) ---")
    print(raw_df.head())
    print("-" * 40)
    
    # Define hyperparameters
    LAMBDA_1 = 0.5  # Fairness
    LAMBDA_2 = 0.01 # Regularization
    
    # 1. Transform data and get Q/c matrices
    Q_solver, c, weight_names = simulate_and_transform_data(raw_df, LAMBDA_1, LAMBDA_2)
    
    # 2. Run Projected Newton's Method
    result = solve_with_projected_newtons_method(Q_solver, c, weight_names)
    
    # 3. Analyze and Plot Convergence
    if 'history' in result and result['history']:
        print("\n[INFO] Plotting convergence history...")
        plot_convergence(result['history'], weight_names)
    
    print("\nFinal Optimal Weights Found by Projected Newton's Method:")
    print(result['weights_df'].set_index('Feature')) """