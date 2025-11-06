from ast import Dict
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Needed for visualization

def visualize_weights(all_results: list):
    """
    Generates a bar plot comparing the weight vectors w* from all implemented methods.
    """
    if not all_results or len(all_results) < 1:
        print("No results to visualize.")
        return

    # FIX: Filter out failed methods robustly. Check for key existence first.
    # Include the result if it has a good score OR if it's explicitly a baseline method.
    plot_data = [
        r for r in all_results 
        if ('max_f_w' in r and r['max_f_w'] > -1e8) 
        or 'Baseline' in r.get('method', '')
    ]
    
    if not plot_data:
        print("No valid results to visualize weights.")
        return
        
    # The weight names should be attached to every result, so grab them from the first valid one
    weight_names = plot_data[0]['weight_names']
    num_weights = len(weight_names)
    
    x = np.arange(num_weights)
    bar_width = 0.8 / len(plot_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, result in enumerate(plot_data):
        weights = result.get('weights', result.get('w_star'))
        method_name = result['method']
        
        offset = bar_width * i - (bar_width * (len(plot_data) - 1)) / 2
        
        rects = ax.bar(x + offset, weights, bar_width, label=method_name)
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=7)

    ax.set_ylabel('Weight Value ($w_k$)')
    ax.set_title('Comparison of Event Weight Vectors Across Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(weight_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def plot_objective_comparison(all_results: list):
    """
    Generates a bar plot comparing the final maximized objective function value (f(w))
    achieved by each optimization method.
    """
    if not all_results or len(all_results) < 1:
        print("No results to plot for objective comparison.")
        return

    # Filter out failed methods
    plot_data = [r for r in all_results if r['max_f_w'] > -1e8]
    if not plot_data:
        print("All methods failed or returned zero objective values.")
        return

    methods = [r['method'] for r in plot_data]
    scores = [r['max_f_w'] for r in plot_data]
    
    # Determine the best method
    best_score = max(scores)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, scores, color=['skyblue', 'salmon', 'lightgreen', 'gold'])
    
    ax.set_ylabel('Maximized Objective Value $f(w)$')
    ax.set_title('Performance Comparison: Maximized Objective Score by Method')
    
    # Add labels and highlight the optimal result
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (best_score * 0.01),
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=9,
                weight='bold' if score == best_score else 'normal')
        
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()

def visualize(data,all_weights: list):
    """
    Calculates the final User-Item Interaction Score (S_ij) and runs visualizations.
    """
    # The score calculation (S_ij) part is skipped for brevity as requested previously.
    
    # 1. Run the new Objective Comparison Plot
    plot_objective_comparison(all_weights)
    
    # 2. Run the existing Weight Comparison Plot
    visualize_weights(all_weights)

    best_result = get_best_method_and_weights(all_weights)
    if best_result:
        print("\nWeights for the Best Method:")
        best_weights_df = pd.DataFrame({
            'Event': data['weight_names'],
            'Weight': best_result['weights'].flatten()
        })
        print(best_weights_df.set_index('Event'))

# --- New function to find the best method ---
def get_best_method_and_weights(all_results: list) -> Dict:
    """
    Identifies the best optimization method based on the highest maximized 
    objective function value (max_f_w).
    
    Args:
        all_results (list): A list of dictionaries, each containing results 
                            from one optimization method.

    Returns:
        Dict: A dictionary containing the best method's details ('method', 'weights', 'max_f_w').
    """
    if not all_results:
        print("No results provided to determine the best method.")
        return {}

    best_result = None
    max_score = -np.inf

    for result in all_results:
        # Safely retrieve score, defaulting to a very low number if key is missing (for failed runs)
        score = result.get('max_f_w', -np.inf)
        
        if score > max_score:
            max_score = score
            best_result = result
            
    if best_result and max_score > -np.inf:
        print("\n--- Best Optimization Method Found ---")
        print(f"Method: {best_result['method']}")
        print(f"Max Objective Score (f(w)): {best_result['max_f_w']:.4f}")
        return {
            'method': best_result['method'],
            'weights': best_result.get('weights', best_result.get('w_star')),
            'max_f_w': best_result['max_f_w']
        }
    else:
        print("Could not find a valid result with a positive objective score.")
        return {}
