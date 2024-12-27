# this file tests all cases given in the slides

import unittest
import numpy as np
from LP_solver.modules.simplex import solve_simplex
from LP_solver.modules.big_m_initialization import initialize_tableau
from LP_solver.modules.feasibility_check import *

class TestSimplex(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters"""
        self.tol = 1e-6

    def test_feasible_solution(self):
        """Test case with known feasible solution"""
        c = np.array([-10, -12, -12, 0, 0, 0])
        A = np.array([
            [1, 2, 2, 1, 0, 0],
            [2, 1, 2, 0, 1, 0],
            [2, 2, 1, 0, 0, 1]
        ])
        b = np.array([20, 20, 20])
        
        # Convert to standard form (already done in the example)
        initial_tableau = initialize_tableau(c, A, b)
        final_tableau, status, basic_vars, solution = solve_simplex(initial_tableau)

        self.assertEqual(status, 'optimal')
        self.assertIsNotNone(solution)
        np.testing.assert_array_almost_equal(solution, [4, 4, 4, 0, 0, 0], decimal=5)
        self.assertAlmostEqual(-final_tableau[0, -1], -136.0, places=5)

    def test_redundant_constraints(self):
        """Test case with redundant constraints"""
        c = np.array([1, 2, 3])
        A = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [1, 1, 1]
        ])
        b = np.array([6, 12, 3])
        is_feasible, c, A, b = check_and_fix_rank(c, A, b)
        initial_tableau = initialize_tableau(c, A, b)
        final_tableau, status, basic_vars, solution = solve_simplex(initial_tableau)
        
        self.assertEqual(status, 'optimal')
        self.assertIsNotNone(solution)
        np.testing.assert_array_almost_equal(solution, [0, 3, 0], decimal=5)
        self.assertAlmostEqual(-final_tableau[0, -1], 6.0, places=5)

    def test_unbounded(self):
        """Test case with unbounded solution"""
        c = np.array([-1, 0])
        A = np.array([[1, -1],
                      [-1, 1]
                      ])
        b = np.array([0, 0])

        initial_tableau = initialize_tableau(c, A, b)
        final_tableau, status, basic_vars, solution = solve_simplex(initial_tableau)
        
        self.assertEqual(status, 'unbounded')
        self.assertIsNone(solution)

    def test_infeasible(self):
        """Test case with infeasible solution"""
        c = np.array([1, 1, 1])
        A = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 1]
                      ])
        b = np.array([2, 2, 2, 2])
        is_feasible, c, A, b = check_and_fix_rank(c, A, b)
        self.assertFalse(is_feasible)

    def test_blands(self):
        initial_tableau = np.array([
        [-3/4, 20, -1/2, 6, 0, 0, 0, 3],
        [1/4, -8, -1, 9, 1, 0, 0, 0],
        [1/2, -12, -1/2, 3, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1]
    ])
        final_tableau, status, basic_vars, solution = solve_simplex(initial_tableau)
        self.assertEqual(status, 'optimal')
        self.assertIsNotNone(solution)
        self.assertNotEqual(-final_tableau[0, -1], -3)
        
    
if __name__ == '__main__':
    unittest.main() 