"""
This module verifies the solution found by the simplex method against the expected output.
It compares both the objective value and the solution vector, accounting for potential
numerical errors in floating-point calculations.
"""

import numpy as np
from pathlib import Path

def verify_solution(objective_value, solution, output_file="LP_expected_output.out"):
    """
    Verify the computed solution against the expected output
    
    Parameters:
    -----------
    objective_value : float
        Computed objective value from simplex method
    solution : ndarray
        Computed solution vector from simplex method
    output_file : str
        Path to the expected output file (default: LP_expected_output.out)
        
    Returns:
    --------
    bool
        True if solution matches expected output within tolerance
    dict
        Detailed verification results
    """
    # Tolerance for floating-point comparisons
    TOL = 1e-3
    
    # Construct path relative to project root
    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / output_file
    
    try:
        with open(output_path, 'r') as f:
            # First line: expected objective value
            expected_obj = float(f.readline())
            # Second line: expected solution vector
            expected_sol = np.array([float(x) for x in f.readline().split()])
            
        # Check dimensions
        if len(solution) != len(expected_sol):
            return False, {
                'status': 'error',
                'message': f'Solution dimension mismatch: got {len(solution)}, expected {len(expected_sol)}'
            }
            
        # Check objective value
        obj_diff = abs(objective_value - expected_obj)
        obj_match = obj_diff < TOL
        
        # Check solution vector
        sol_diff = np.max(np.abs(solution - expected_sol))
        sol_match = sol_diff < TOL
        
        # Prepare detailed results
        results = {
            'status': 'success' if (obj_match and sol_match) else 'mismatch',
            'objective_match': obj_match,
            'objective_difference': obj_diff,
            'solution_match': sol_match,
            'solution_difference': sol_diff,
            'expected_objective': expected_obj,
            'computed_objective': objective_value,
            'expected_solution': expected_sol,
            'computed_solution': solution
        }
        
        return obj_match and sol_match, results
        
    except FileNotFoundError:
        return False, {
            'status': 'error',
            'message': f'Expected output file not found: {output_file}'
        }
    except Exception as e:
        return False, {
            'status': 'error',
            'message': f'Error during verification: {str(e)}'
        }
    