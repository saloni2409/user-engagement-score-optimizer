def calculate_obj_function(Q_solver, c, w):
    return 0.5 * w.T @ Q_solver @ w - c.T @ w
