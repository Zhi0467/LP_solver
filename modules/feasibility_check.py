"""
input: a standard LP minimization problem (given A or converted from general form)
output: a standard LP minimization problem (with row full rank A)
"""

import numpy as np
from numpy.linalg import matrix_rank, qr

def pre_check_matrix(A, tol=1e-8):
    """
    Perform basic checks on matrix A for zero rows/columns and linear dependencies.
    
    Parameters:
    -----------
    A : ndarray
        Constraint coefficients matrix
    tol : float
        Tolerance for zero comparison
    
    Returns:
    --------
    valid_rows : list
        Indices of valid rows to keep
    """
    m, n = A.shape
    valid_rows = list(range(m))
    
    # Check for zero rows
    zero_rows = np.where(np.all(np.abs(A) < tol, axis=1))[0]
    valid_rows = [i for i in valid_rows if i not in zero_rows]
    multiple_rows = []

    # Check for multiple rows
    for i in range(len(valid_rows)-1):
        row_i = A[valid_rows[i]]
        norm_i = np.linalg.norm(row_i)
        if norm_i < tol:
            continue
            
        for j in range(i+1, len(valid_rows)):
            row_j = A[valid_rows[j]]
            norm_j = np.linalg.norm(row_j)
            if norm_j < tol:
                continue
                
            # Find non-zero elements in row_i to compute ratio
            non_zero_idx = np.where(np.abs(row_i) > tol)[0]
            if len(non_zero_idx) > 0:
                # Compute ratio using first non-zero element
                ratio = row_j[non_zero_idx[0]] / row_i[non_zero_idx[0]]
                # Check if row_j = ratio * row_i
                if np.all(np.abs(row_j - ratio * row_i) < tol):
                    multiple_rows.append(valid_rows[j])
    # delete multiple rows
    valid_rows = [i for i in valid_rows if i not in multiple_rows]
    
    return valid_rows

def qr_rank_check(A, tol=1e-8):
    """
    Perform QR decomposition and remove dependent rows/columns.
    
    Parameters:
        A (ndarray): Input matrix (m x n).
        tol (float): Tolerance for identifying zero diagonal elements.
        
    Returns:
        Q_reduced (ndarray): Reduced Q matrix with independent columns.
        R_reduced (ndarray): Reduced R matrix with independent rows.
    """
    # Perform QR decomposition on the transpose of A
    Q, R = np.linalg.qr(A.T)
    
    # Find the diagonal elements of R
    diag_R = np.abs(np.diag(R))
    
    # Identify indices of independent rows/columns (diagonal elements > tol)
    independent_indices = np.where(diag_R > tol)[0]
    
    return independent_indices

def check_feasibility(A, b, tol=1e-8):
    """
    Check if the system Ax = b has any feasible solutions.
    A system is infeasible if b is not in the column space of A.
    """
    # Get the projection of b onto the column space of A
    Q, R = np.linalg.qr(A)
    
    # Find rank by counting non-zero diagonal elements in R
    r = np.sum(np.abs(np.diag(R)) > tol)
    
    # Use only the first r columns of Q for projection
    Q_r = Q[:, :r]
    b_proj = Q_r @ Q_r.T @ b
    
    # Check if b equals its projection (within tolerance)
    if np.linalg.norm(b - b_proj) > tol:
        return False
    
    return True

def check_and_fix_rank(c, A, b, tol=1e-8):
    """
    Check feasibility and fix rank of constraint matrix A.
    
    Parameters:
    -----------
    c : ndarray
        Objective function coefficients
    A : ndarray
        Constraint coefficients matrix
    b : ndarray
        Right-hand side values
    tol : float
        Tolerance for zero comparison
    
    Returns:
    --------
    feasible : bool
        True if the system has feasible solutions
    c : ndarray
        Original objective coefficients
    A_full : ndarray
        Constraint matrix with full row rank
    b_full : ndarray
        Updated right-hand side values
    """
    # First check feasibility
    if not check_feasibility(A, b, tol):
        print("System is infeasible: b is not in the column space of A")
        return False, c, A, b
    
    # Continue with rank fixing if feasible
    valid_rows = pre_check_matrix(A, tol)
    if len(valid_rows) < A.shape[0]:
        A = A[valid_rows]
        b = b[valid_rows]
    
    if A.shape[0] == 0:
        print("A is a zero matrix, we are dealing with an unconstrained problem.")
        return True, c, A, b
    
    independent_rows = qr_rank_check(A, tol)
    
    if len(independent_rows) < A.shape[0]:
        A_full = A[independent_rows]
        b_full = b[independent_rows]
        return True, c, A_full, b_full
    
    return True, c, A, b



"""
# Example usage
c = np.array([1, -2, 3])
A = np.array([[1, 0, 1],
              [2, 0, 2],  # This row is 2 times the first row
              [0, 0, 0],  # Zero row
              [0, 1, 1],
              [1, 1, 2]])
b = np.array([5, 10, 0, 3, 1])

feasibility, c_full, A_full, b_full = check_and_fix_rank(c, A, b)
print(f"c: {c_full}\n A: {A_full}\n b: {b_full}")
"""
