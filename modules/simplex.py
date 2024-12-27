"""
Input: an initial tableau produced by big-M
Output: solution by simplex method performed on the tableau
"""
import numpy as np

def solve_simplex(tableau):
    """
    Solve LP using simplex method starting from initial tableau, using Bland's rule
    
    Parameters:
    -----------
    tableau : ndarray
        Initial tableau from big-M method
        
    Returns:
    --------
    tableau : ndarray
        Final tableau after simplex method
    status : str
        'optimal', 'unbounded', or 'infeasible'
    basic_vars : list
        Indices of basic variables in optimal solution
    solution : ndarray
        Values of original variables in optimal solution (excluding artificial variables)
    """
    m, n = tableau.shape
    m -= 1  # Subtract 1 for objective row
    n -= 1  # Subtract 1 for RHS column
    
    num_original_vars = n - m  # Number of original variables (excluding artificial)
    
    # Initialize basic variables (initially artificial variables are basic)
    basic_vars = list(range(num_original_vars, n))
    
    while True:
        # Step 1: Find entering variable using Bland's rule
        obj_row = tableau[0, :-1]
        if np.all(obj_row >= -1e-10):  # Allow for small numerical errors
            # Construct solution (only for original variables)
            solution = np.zeros(num_original_vars)
            for i, basic_idx in enumerate(basic_vars):
                if basic_idx < num_original_vars:  # Only include original variables
                    solution[basic_idx] = tableau[i+1, -1]
            return tableau, 'optimal', basic_vars, solution
            
        # Choose smallest index with negative coefficient
        negative_indices = np.where(obj_row < -1e-10)[0]
        enter_col = negative_indices[0]
        
        # Step 2: Find leaving variable (minimum ratio test with Bland's rule)
        constraints = tableau[1:, enter_col]
        rhs = tableau[1:, -1]
        
        # Check if problem is unbounded
        if np.all(constraints <= 0):
            return tableau, 'unbounded', None, None
            
        # Calculate ratios and implement Bland's rule for ties
        min_ratio = np.inf
        leave_row = -1
        min_basic_var = np.inf
        
        for i in range(len(constraints)):
            if constraints[i] > 1e-10:  # Positive coefficient
                ratio = rhs[i] / constraints[i]
                # Update if:
                # 1. ratio is strictly smaller, or
                # 2. ratio is equal but basic variable index is smaller
                if ratio < min_ratio - 1e-10 or (abs(ratio - min_ratio) < 1e-10 and basic_vars[i] < min_basic_var):
                    min_ratio = ratio
                    leave_row = i + 1  # +1 because we skipped objective row
                    min_basic_var = basic_vars[i]
        
        # Update basic variables
        basic_vars[leave_row - 1] = enter_col
        
        # Step 3: Pivot
        pivot = tableau[leave_row, enter_col]
        tableau[leave_row] = tableau[leave_row] / pivot
        
        for i in range(tableau.shape[0]):
            if i != leave_row:
                tableau[i] = tableau[i] - tableau[i, enter_col] * tableau[leave_row]

if __name__ == "__main__":
    # Test with example from big_M initialization
    from big_m_initialization import initialize_tableau
    from standard_form import convert_to_standard_form
    
    c = np.array([-4, -1])
    A = np.array([
        [-1, 2],
        [2, 3],
        [1, -1],
        [1, 0],
        [0, 1]])
    b = np.array([4, 12, 3, 0, 0])
    ineqs = ['<=', '<=', '<=', '>=', '>=']
    c, A, b = convert_to_standard_form(c, A, b, ineqs)
    initial_tableau = initialize_tableau(c, A, b)
    final_tableau, status, basic_vars, solution = solve_simplex(initial_tableau)
    print(f"Status: {status}")
    if status == 'optimal':
        print("Solution (original variables only):", solution)
        print("Optimal value:", -final_tableau[0, -1])