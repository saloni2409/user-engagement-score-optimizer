import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Constants for the new long-format data schema
UCID_COL = 'ucid'
PRODUCT_ID_COL = 'product_id'
EVENT_TYPE_COL = 'eventType'
EVENT_COUNT_COL = 'event_count'


def transform_data(raw_data_df: pd.DataFrame, lambda_1: float = 1.0, lambda_2: float = 0.1) -> dict:
    """
    Transforms the event log format (long format) into the required wide User-Item-Feature 
    matrix and computes Q and c for the Quadratic Programming (QP) solver.
    
    Args:
        raw_data_df: DataFrame with columns 'ucid', 'product_id', 'eventType', 'event_count'.
        lambda_1 (float): The Fairness Coefficient (lambda_1).
        lambda_2 (float): The Regularization Coefficient (lambda_2).

    Returns:
        dict: A dictionary containing the transformed data, metadata, and QP matrices.
    """
    if raw_data_df.empty:
        print("Error: Input DataFrame is empty.")
        return {}

    print(f"Raw data received. Shape: {raw_data_df.shape}")

    # --- Step 1: Pivoting Raw Event Log Data ---
    # Convert the long (sparse) event data into a wide User-Item-Feature matrix.
    wide_df = raw_data_df.pivot_table(
        index=[UCID_COL, PRODUCT_ID_COL],
        columns=EVENT_TYPE_COL,
        values=EVENT_COUNT_COL,
        fill_value=0 # Users who didn't perform an event for a product get a count of 0
    )
    
    # The column names are now the feature/event types
    weight_names = wide_df.columns.tolist()
    K = len(weight_names)
    M = wide_df.shape[0] # Total number of unique user-item interactions
    print(f"Transformed to wide format. {M} User-Item pairs, {K} features: {weight_names}")
    
    # --- Step 2: Log-Transformation (e_ij^k = log(1 + E_ij^k)) ---
    # A is the matrix of transformed interaction counts for all (User, Item) pairs
    A = np.log1p(wide_df.values)
    
    # --- Step 3: User Aggregation (for Fairness Term) ---
    # Group by UCID (level 0 of the index) to get the total transformed activity per user 
    # for each event type.
    
    # B is the matrix of total transformed feature sums per user.
    B_df = pd.DataFrame(A, index=wide_df.index, columns=weight_names).groupby(level=UCID_COL).sum()
    B = B_df.values
    
    unique_users = B.shape[0] # Nu: Number of unique users

    # --- Step 4: Final Structure for QP Solver ---
    # Objective function: max w^T * c - 1/2 * w^T * Q * w
    
    # 1. Linear Component (c): Total Engagement
    # c_k = sum_i sum_j e_ij^k 
    c = A.sum(axis=0) 
    
    # 2. Quadratic Component (Q): Fairness + Regularization
    
    # Calculate Cov(B) - the covariance matrix of the user-sum vectors (B)
    Cov_B = np.cov(B, rowvar=False) 
    
    # The Hessian matrix (Q_solver) for the QP solver.
    # Q_solver = 2 * (lambda_1 * Cov_B + lambda_2 * I)
    I = np.identity(K)
    Q_solver = 2 * (lambda_1 * Cov_B + lambda_2 * I)

    return {
        'A': A,                # (M x K) Transformed interaction matrix (e_ij^k)
        'E': wide_df.values,   # (M x K) Raw interaction matrix (E_ij^k)
        'c': c,                # (K,) Linear vector for engagement
        'Q_solver': Q_solver,  # (K x K) Quadratic matrix (Hessian)
        'weights_count': K,    # K: Number of event types (weights)
        'interaction_count': M,# M: Number of user-item pairs
        'user_count': unique_users, # Nu: Number of unique users
        'weight_names': weight_names,
        'hyperparameters': {'lambda_1': lambda_1, 'lambda_2': lambda_2},
    }

# --- Main Execution Block for Demonstration ---

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

    print("--- Running Data Processing on Sample Data (New Long Format) ---")
    data_for_qp = transform_data(raw_df)

    if data_for_qp:
        print("\n--- Data Summary for QP Solver ---")
        print(f"Number of Weights (K): {data_for_qp['weights_count']}")
        print(f"Weight Names (Event Types): {data_for_qp['weight_names']}")
        print("\nShape of Raw Interaction Matrix (E - M x K):", data_for_qp['E'].shape)
        print("Shape of Transformed Interaction Matrix (A - M x K):", data_for_qp['A'].shape)