"""
In this file we construct an initial basic feasible solution (BFS) with the big-M method.
input: c, A, b (standard form)
output: initial simplex tableau 
"""
import numpy as np

def initialize_tableau(c, A, b, M = 1e4):
    """
    Construct initial simplex tableau using big-M method
    
    Parameters:
    -----------
    c : ndarray
        Objective function coefficients
    A : ndarray
        Constraint coefficients matrix
    b : ndarray
        Right-hand side values
        
    Returns:
    --------
    tableau : ndarray
        Initial simplex tableau with artificial variables
    """
    m, n = A.shape  # m constraints, n original variables
    
    # Create tableau with additional artificial variables
    tableau = np.zeros((m + 1, n + m + 1))
    
    # Process each constraint first
    for i in range(m):
        if b[i] < 0:
            tableau[i+1, :n] = -A[i]
            tableau[i+1, n+i] = 1
            tableau[i+1, -1] = -b[i]
        else:
            tableau[i+1, :n] = A[i]
            tableau[i+1, n+i] = 1
            tableau[i+1, -1] = b[i]
    
    # Compute the objective row using reduced costs
    # Original objective coefficients
    tableau[0, :n] = c
    
    # Subtract M times the coefficients in rows where artificial variables are basic
    # (initially all rows have basic artificial variables)
    for i in range(m):
        if b[i] < 0:
            tableau[0, :n] += M * A[i]  # Add because row was negated
        else:
            tableau[0, :n] -= M * A[i]
    
    # Artificial variables' coefficients in objective row are 0 (they're basic)
    tableau[0, n:n+m] = 0
    
    # Initial objective value is -M times sum of artificial variables' values
    initial_obj = -sum(M * abs(b[i]) for i in range(m))
    tableau[0, -1] = initial_obj
    
    return tableau

if __name__ == "__main__":
    # Example LP in standard form:
    # min  2x₁ + x₂
    # s.t. x₁ + x₂ = 4
    #      2x₁ - x₂ = 2

    c = np.array([2, 1])
    A = np.array([
        [1, 1],
        [2, -1]
    ])
    b = np.array([4, -2])

    tableau = initialize_tableau(c, A, b)
    print("Initial Tableau:")
    print(tableau)
