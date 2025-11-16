def calculate_obj_function(Q_solver, c, w):
    return 0.5 * (w.T @ Q_solver @ w) - (c.T @ w)

def objective_function(w, Q, C):
    """Calculates the value of the objective function F(w) = w^T Q w - C^T w."""
    # The term w^T Q w
    quadratic_term = w.T @ Q @ w
    # The term C^T w
    linear_term = C.T @ w
    return 0.5 * quadratic_term - linear_term