import unittest
import numpy as np
from LP_solver.modules.feasibility_check import *

class TestFeasibilityCheck(unittest.TestCase):
    def setUp(self):
        """Set up any common test fixtures"""
        self.tol = 1e-8

    def test_zero_matrix(self):
        """Test case for zero matrix"""
        c = np.array([1, 2])
        A = np.zeros((2, 2))
        b = np.array([0, 0])
        
        feasible, c_out, A_out, b_out = check_and_fix_rank(c, A, b)
        self.assertTrue(feasible)
        self.assertEqual(A_out.shape[0], 0)  # Should remove all zero rows

    def test_linearly_dependent_rows(self):
        """Test case for matrix with linearly dependent rows"""
        c = np.array([1, -2, 3])
        A = np.array([
            [1, 0, 1],
            [2, 0, 2],  # 2 times first row
            [0, 1, 1]
        ])
        b = np.array([5, 10, 3])
        
        feasible, c_out, A_out, b_out = check_and_fix_rank(c, A, b)
        self.assertTrue(feasible)
        self.assertEqual(A_out.shape[0], 2)  # Should remove one dependent row

    def test_infeasible_system(self):
        """Test case for infeasible system"""
        c = np.array([1, 1])
        A = np.array([
            [1, 1],
            [1, 1]
        ])
        b = np.array([1, 2])  # Contradictory constraints
        
        feasible, _, _, _ = check_and_fix_rank(c, A, b)
        print(A)
        print(b)
        self.assertFalse(feasible)

    def test_full_rank_matrix(self):
        """Test case for matrix already in full rank"""
        c = np.array([1, 2, 3])
        A = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        b = np.array([1, 1])
        
        feasible, c_out, A_out, b_out = check_and_fix_rank(c, A, b)
        self.assertTrue(feasible)
        np.testing.assert_array_equal(A, A_out)  # Should remain unchanged

    def test_multiple_dependencies(self):
        """Test case for matrix with multiple dependencies"""
        c = np.array([1, 2, 3])
        A = np.array([
            [1, 0, 1],
            [2, 0, 2],  # 2 times first row
            [1, 0, 1],  # Same as first row
            [0, 1, 0]
        ])
        b = np.array([1, 2, 1, 3])
        
        feasible, c_out, A_out, b_out = check_and_fix_rank(c, A, b)
        self.assertTrue(feasible)
        self.assertEqual(A_out.shape[0], 2)  # Should keep only independent rows

    def test_zero_rows(self):
        """Test case for matrix with zero rows"""
        c = np.array([1, 2])
        A = np.array([
            [1, 1],
            [0, 0],  # Zero row
            [2, 2]
        ])
        b = np.array([1, 0, 2])
        
        feasible, c_out, A_out, b_out = check_and_fix_rank(c, A, b)
        self.assertTrue(feasible)
        self.assertEqual(A_out.shape[0], 1)  # Should remove zero and dependent rows

    def test_numerical_stability(self):
        """Test case for numerical stability with nearly dependent rows"""
        c = np.array([1, 2])
        A = np.array([
            [1, 1],
            [1, 1 + 1e-10]  # Nearly dependent row
        ])
        b = np.array([1, 1])
        
        feasible, c_out, A_out, b_out = check_and_fix_rank(c, A, b)
        self.assertTrue(feasible)
        self.assertEqual(A_out.shape[0], 1)  # Should identify near dependency

if __name__ == '__main__':
    unittest.main() 