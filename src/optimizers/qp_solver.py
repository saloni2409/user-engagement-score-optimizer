import cvxpy as cp
import numpy as np
import pandas as pd
# Import the data prep function and constants

def solve_quadratic_programming(data: dict, constraints_type: str = 'hierarchy') -> dict:
    """
    Solves the Constrained Quadratic Programming (QP) problem to find the optimal
    weight vector w* that maximizes the weighted score function f(w).

    Optimization Goal (Maximization):
        max f(w) = c^T * w - 1/2 * w^T * Q_solver * w

    QP Solver Format (Minimization):
        min 1/2 * w^T * Q_solver * w - c^T * w

    Args:
        data (dict): The dictionary output from load_and_transform_data.
        constraints_type (str): Specifies the constraints to apply ('simple' or 'hierarchy').

    Returns:
        dict: Results including the optimal weights, objective value, and status.
    """
    K = data['weights_count']
    c = data['c']
    Q_solver = data['Q_solver']
    weight_names = data['weight_names']

    # 1. Define the Decision Variable (w)
    # w is a column vector of K weights.
    w = cp.Variable(K, name="WeightVector")

    # 2. Define the Objective Function
    # The QP solver minimizes: (1/2 * w^T * Q * w) - c^T * w
    objective = cp.Minimize(0.5 * cp.quad_form(w, Q_solver) - c @ w)

    # 3. Define the Constraints

    # A. Normalization (Equality Constraint: sum(w_k) = 1)
    constraints = [
        cp.sum(w) == 1
    ]

    # B. Non-Negativity (Bound Constraint: w_k >= 0)
    constraints.append(w >= 0)

    if constraints_type == 'hierarchy':
        print("\nApplying Logical Hierarchy Constraint:")
        # C. Logical Hierarchy Constraint (Inequality)
        # We use generic event names derived from the sample data: purchase > cart > view
        
        # Determine indices:
        try:
            view_idx = weight_names.index('product_view')
            cart_idx = weight_names.index('add_to_cart')
            purchase_idx = weight_names.index('purchase')
        except ValueError as e:
            print(f"Error: Required feature names ('product_view', 'add_to_cart', 'purchase') not found for hierarchy constraint. Falling back to simple constraints. Missing: {e}")
            constraints_type = 'simple' # Fallback
        
        if constraints_type == 'hierarchy':
            # w_purchase >= w_cart
            print(f"  Constraint: w['{weight_names[purchase_idx]}'] >= w['{weight_names[cart_idx]}']")
            constraints.append(w[purchase_idx] >= w[cart_idx])

            # w_cart >= w_view
            print(f"  Constraint: w['{weight_names[cart_idx]}'] >= w['{weight_names[view_idx]}']")
            constraints.append(w[cart_idx] >= w[view_idx])

    else: # constraints_type == 'simple'
        print("\nApplying Simple Constraints (Normalization + Non-Negativity only).")

    # 4. Solve the Problem
    problem = cp.Problem(objective, constraints)
    
    # We use the standard OSQP solver which is good for QP problems
    try:
        problem.solve(solver=cp.OSQP)
    except Exception as e:
        # Fallback to SCS solver if OSQP fails
        print(f"OSQP solver failed: {e}. Trying SCS...")
        problem.solve(solver=cp.SCS)

    # 5. Extract Results
    if problem.status in ["optimal", "optimal_inaccurate"]:
        optimal_weights = w.value
        maximized_objective_value = -problem.value # Flip the minimized value back to max f(w)
        
        results_df = pd.DataFrame({
            'Event': weight_names,
            'Optimal_Weight': optimal_weights.flatten()
        })
        
        print("\n--- Optimization Results ---")
        print(f"Solver Status: {problem.status}")
        print(f"Maximized Objective Value (f(w)): {maximized_objective_value:.4f}")
        print("\nOptimal Weight Vector (w*):")
        print(results_df)

        return {
            'method': 'QP Solver (' + constraints_type + ')',
            'weights_df': results_df,
            'w_star': optimal_weights,
            'status': problem.status,
            'max_f_w': maximized_objective_value
        }
    else:
        print(f"Optimization failed. Solver status: {problem.status}")
        return {
            'status': problem.status,
            'method': f'QP Failed ({constraints_type.capitalize()})',
            'weights': np.zeros(K),
            'max_f_w': -1e9 # Return a very low score
        }

"""
# --- Demonstration and Integration ---

if __name__ == '__main__':
    
    # --- SIMULATE PARQUET-LIKE INPUT DATA (Long Format) ---
    np.random.seed(42)
    sample_data = {
        UCID_COL: [f'U{i}' for i in range(10)] * 3 + ['U1'] * 2 + ['U2'] * 2,
        PRODUCT_ID_COL: ['P1'] * 10 + ['P2'] * 10 + ['P3'] * 10 + ['P1', 'P2'] + ['P1', 'P3'],
        EVENT_TYPE_COL: ['product_view'] * 10 + ['add_to_cart'] * 10 + ['purchase'] * 10 + ['product_view', 'add_to_cart'] + ['purchase', 'product_view'],
        EVENT_COUNT_COL: np.random.randint(1, 40, size=34)
    }
    raw_df = pd.DataFrame(sample_data).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Define hyperparameters
    LAMBDA_1 = 1.0  # Fairness
    LAMBDA_2 = 0.1 # Regularization

    # 1. Load and transform the data
    data_for_qp = load_and_transform_data(raw_df, lambda_1=LAMBDA_1, lambda_2=LAMBDA_2)

    if data_for_qp:
        print("\n=======================================================")
        print("= Method 1: Formal Constrained Optimization (Hierarchy) =")
        print("=======================================================")
        
        # 2. Solve the QP problem with Logical Hierarchy Constraints
        qp_results = solve_quadratic_programming(data_for_qp, constraints_type='hierarchy')
        
        # 3. Solve the QP problem with only Simple Constraints
        print("\n===================================================")
        print("= Method 1: Formal Constrained Optimization (Simple) =")
        print("===================================================")
        qp_simple_results = solve_quadratic_programming(data_for_qp, constraints_type='simple') 
"""