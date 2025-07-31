import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

# Import your functions
# from your_module import abs_diff_set, smallest_non_divisor, qft_projectors, find_zero_sum_combination, caratheodory_reduce, angles

class TestLinearAlgebraHelpers(unittest.TestCase):
    def test_abs_diff_set(self):
        K = [1, 3, 5, 7]
        expected = np.array([2, 4, 6])
        result = abs_diff_set(K)
        assert_array_equal(result, expected)

    def test_smallest_non_divisor(self):
        S = [6, 10, 15]
        self.assertEqual(smallest_non_divisor(S), 7)
        S2 = [2, 4, 8]
        self.assertEqual(smallest_non_divisor(S2), 3)

    def test_qft_projectors_shape_and_type(self):
        K = [0, 1, 2]
        q = 3
        projectors = qft_projectors(K, q)
        self.assertEqual(projectors.shape, (9, 3))
        self.assertTrue(np.iscomplexobj(projectors))

    def test_find_zero_sum_combination(self):
        V = np.array([[1, 2, 3], [4, 5, 6]])
        lam = find_zero_sum_combination(V)
        if lam is not None:
            assert_allclose(np.dot(V, lam), np.zeros(V.shape[0]), atol=1e-12)
            self.assertAlmostEqual(np.sum(lam), 0, places=12)

    def test_caratheodory_reduce(self):
        X = np.array([[0, 1, 0], [0, 0, 1]]).astype(float)
        p = np.array([0.5, 0.25, 0.25])
        Xr, pr, idx = caratheodory_reduce(X, p)
        self.assertEqual(Xr.shape[1], 3)  # Since d=2, expect d+1=3
        self.assertAlmostEqual(np.sum(pr), 1.0, places=12)
        assert_allclose(np.dot(Xr, pr), np.dot(X, p), atol=1e-12)

    def test_angles_basic(self):
        K = [2, 5, 7]
        ang, r = angles(K)
        self.assertTrue(len(ang) <= len(K)+1)
        self.assertTrue(np.all(ang >= 0) and np.all(ang < 1))

if __name__ == "__main__":
    unittest.main()