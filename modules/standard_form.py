'''
input: an LP problem in general form
output: the same LP problem in standard form (minimization with equality constraints)
'''

import numpy as np
import os
from pathlib import Path

def convert_to_standard_form(c, A, b, inequalities, maximization=False):
    """
    Convert LP problem to standard form: min c^T x s.t. Ax = b, x >= 0
    
    Parameters:
    -----------
    c : array-like
        Objective function coefficients
    A : array-like
        Constraint coefficients matrix
    b : array-like
        Right-hand side values
    inequalities : list
        List of strings indicating inequality types ('<=', '>=', '=')
    maximization : bool
        True if original problem is maximization
    
    Returns:
    --------
    c_new : ndarray
        Updated objective coefficients
    A_new : ndarray
        Updated constraint matrix
    b_new : ndarray
        Updated right-hand side values
    """
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Step 1: Convert maximization to minimization
    if maximization:
        c = -c
    
    # Step 2: Convert inequalities to equalities using slack/surplus variables
    A_new = []
    c_new = list(c)
    equality_indices = []  # Keep track of equality constraint indices
    
    for i, (row, inequality) in enumerate(zip(A, inequalities)):
        if inequality == '<=':
            # Add slack variable (s >= 0)
            new_row = list(row)
            slack_columns = np.zeros(len(inequalities))
            slack_columns[i] = 1
            new_row.extend(slack_columns)
            A_new.append(new_row)
            c_new.append(0)  # Objective coefficient for slack variable
            
        elif inequality == '>=':
            # Add surplus variable (s >= 0)
            new_row = list(row)
            slack_columns = np.zeros(len(inequalities))
            slack_columns[i] = -1
            new_row.extend(slack_columns)
            A_new.append(new_row)
            c_new.append(0)  # Objective coefficient for surplus variable
            
        else:  # equality constraint
            new_row = list(row)
            slack_columns = np.zeros(len(inequalities))
            new_row.extend(slack_columns)
            A_new.append(new_row)
            equality_indices.append(i)
    
    # Convert to numpy arrays
    A_new = np.array(A_new, dtype=float)
    c_new = np.array(c_new, dtype=float)
    
    # Remove columns corresponding to equality constraints
    if equality_indices:
        cols_to_keep = [i for i in range(A_new.shape[1]) if i < len(c) or i-len(c) not in equality_indices]
        A_new = A_new[:, cols_to_keep]
        #c_new = c_new[cols_to_keep]
    
    return c_new, A_new, b


# read_and_standardize function first reads input_file="LP_input.in"
# then use convert_to_standard_form to return the standard LP problem
def read_and_standardize(input_file="LP_input.in"):
    """
    Read LP problem from input file and convert it to standard form
    """
    # Construct path relative to project root
    project_root = Path(__file__).parent.parent  # Go up 2 levels: modules -> LP_solver -> root
    input_path = project_root / "data" / input_file
    
    with open(input_path, 'r') as f:
        # First line: n m (dimensions)
        n, m = map(int, f.readline().split())
        
        # Second line: objective coefficients c
        c = list(map(float, f.readline().split()))
        
        # Remaining lines: augmented matrix [A | b]
        A = []
        b = []
        for _ in range(n):
            row = list(map(float, f.readline().split()))
            A.append(row[:-1])  # All elements except last
            b.append(row[-1])   # Last element
    
    # All constraints are equality constraints in this format
    inequalities = ['=' for _ in range(n)]
    
    # Convert to standard form (minimization problem with equality constraints)
    c_std, A_std, b_std = convert_to_standard_form(c, A, b, inequalities, maximization=False)
    
    return c_std, A_std, b_std


"""
# Example usage
c = [3, -1, 2]
A = [[1, 2, -1],
     [4, -2, 1],
     [1, 1, 1]]
b = [4, 6, 2]
inequalities = ['>=', '=', '<=']
maximization = False

c_std, A_std, b_std = convert_to_standard_form(c, A, b, inequalities, maximization)
print(f"c: {c_std}, A: {A_std}, b: {b_std}")
"""

