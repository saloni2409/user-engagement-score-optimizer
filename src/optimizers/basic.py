import numpy as np

def calculate_heuristic_weights(data: dict) -> dict:
    """
    Implements Method 2: Assigns fixed, heuristic weights and normalizes them.
    Also calculates the objective function value (f(w)) for the heuristic solution.
    """
    weight_names = data['weight_names']
    K = data['weights_count']
    
    # 1. Define heuristic weights
    heuristic_map = {
        'purchase': 10,
        'add_to_cart': 5,
        'product_view': 1
    }
    
    weights = np.array([heuristic_map.get(name, 1) for name in weight_names], dtype=float)

    # 2. Normalize the heuristic weights (sum to 1)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    
    # 3. Calculate objective score (f(w) = c^T * w - 1/2 * w^T * Q_solver * w)
    c = data['c']
    Q_solver = data['Q_solver']
    
    w_vec = weights.reshape(-1, 1)
    
    # Check shape consistency before matrix multiplication
    if w_vec.shape[0] != Q_solver.shape[0] or w_vec.shape[0] != c.shape[0]:
        print("[Error] Dimension mismatch when calculating heuristic objective.")
        max_f_w = -1e9
    else:
        # Linear Term (Engagement)
        linear_term = c.T @ w_vec
        # Quadratic Term (Fairness + Regularization)
        quadratic_term = 0.5 * w_vec.T @ Q_solver @ w_vec
        
        max_f_w = (linear_term - quadratic_term).item()

    return {
        'method': 'Heuristic Baseline',
        'weights': weights,
        'max_f_w': max_f_w
    }


def calculate_log_sum_weights(data: dict) -> dict:
    """
    Implements Method 3: Uniform weights (1/K) on the log-transformed features.
    Also calculates the objective function value (f(w)) for the uniform solution.
    """
    K = data['weights_count']
    uniform_weights = np.ones(K) / K
    
    # 1. Calculate objective score (f(w) = c^T * w - 1/2 * w^T * Q_solver * w)
    c = data['c']
    Q_solver = data['Q_solver']
    
    w_vec = uniform_weights.reshape(-1, 1)
    
    if w_vec.shape[0] != Q_solver.shape[0] or w_vec.shape[0] != c.shape[0]:
        print("[Error] Dimension mismatch when calculating uniform objective.")
        max_f_w = -1e9
    else:
        linear_term = c.T @ w_vec
        quadratic_term = 0.5 * w_vec.T @ Q_solver @ w_vec
        max_f_w = (linear_term - quadratic_term).item()
        
    return {
        'method': 'Log-Sum (Uniform) Baseline',
        'weights': uniform_weights.flatten(),
        'max_f_w': max_f_w
    }

