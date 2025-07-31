"""
Finitely-Supported Canonical Phase Measurements

This module provides efficient routines for convex decomposition of finitely-supported
canonical phase measurements. 

If you use this code in your research, please cite:

> Your Name et al. (2025). "Title of Your Paper." *Journal/Conference Name*. [arXiv/DOI link]

Author: Mohammad Alhejji and Marco Rodríguez
License: MIT
"""

import numpy as np
from scipy.linalg import null_space
from typing import List, Union, Tuple, Optional

def abs_diff_set(K: Union[List[int], np.ndarray]) -> np.ndarray:
    """
    Compute all unique absolute differences |a - b| for a ≠ b, where a, b are elements in K.

    Parameters:
        K (list or np.ndarray): List or array of integer eigenvalues.

    Returns:
        np.ndarray: 1D array of unique absolute differences.
    """
    K = np.asarray(K)
    if K.size == 0:
        raise ValueError("Input list K must not be empty.")
    diffs = np.abs(K[:, None] - K[None, :])
    # Exclude diagonal (zero differences), flatten, and get unique values
    D = np.unique(diffs[np.triu_indices(len(K), k=1)])
    return D

def smallest_non_divisor(S: Union[List[int], np.ndarray]) -> int:
    """
    Find the smallest integer m >= 2 that does not divide any element of S.

    Parameters:
        S (list or np.ndarray): Input array of integers.

    Returns:
        int: The smallest integer m >= 2 that does not divide any element in S.
    """
    S = np.asarray(S).ravel()
    if S.size == 0:
        raise ValueError("Input list S must not be empty.")
    m = 2
    while True:
        if not np.any(S % m == 0):
            return m
        m += 1

def qft_projectors(K: Union[List[Union[int, float]], np.ndarray], q: int) -> np.ndarray:
    """
    Compute QFT projectors for the given basis K and dimension q.

    Parameters:
        K (list or np.ndarray): Input array (integers or floats).
        q (int): Dimension for the QFT.

    Returns:
        np.ndarray: Array of shape (n**2, q) where each column is a flattened projector.
    """
    K = np.asarray(K).ravel()
    n = len(K)
    if n == 0 or q < 1:
        raise ValueError("K must have elements and q must be >= 1.")
    projectors = np.zeros((n**2, q), dtype=np.complex128)
    for j in range(q):
        psi = np.exp(-2j * np.pi * j * K / q)
        norm = np.linalg.norm(psi)
        if norm < 1e-14:
            raise ValueError("Psi vector norm is zero, check input K and q.")
        psi = psi / norm
        P = np.outer(psi, np.conj(psi))
        projectors[:, j] = P.ravel()
    return projectors

def find_zero_sum_combination(V: np.ndarray) -> Optional[np.ndarray]:
    """
    Find a nontrivial vector lambda such that V @ lambda = 0 and sum(lambda) = 0.

    Parameters:
        V (np.ndarray): A (d, L) array.

    Returns:
        np.ndarray or None: A null-space vector (L,) if exists, else None.
    """
    V = np.asarray(V)
    if V.ndim != 2 or V.shape[1] == 0:
        raise ValueError("V must be a 2D array with at least one column.")
    d, L = V.shape
    A = np.vstack([V, np.ones((1, L))])  # (d+1, L)
    N = null_space(A)  # Each column is a null vector
    if N.size == 0:
        return None
    else:
        # Return the first nonzero solution
        return N[:, 0]

def caratheodory_reduce(
    X: np.ndarray,
    p: Union[List[float], np.ndarray],
    tol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Reduces a convex combination to at most d+1 points (Carathéodory's theorem).

    Parameters:
        X (np.ndarray): d x L matrix (columns are points in R^d)
        p (list or np.ndarray): 1D array of convex weights, shape (L,), sum(p) == 1
        tol (float): Tolerance for zero detection.

    Returns:
        X_reduced (np.ndarray): d x (d+1) array of points.
        p_reduced (np.ndarray): 1D array, shape (d+1,), convex weights.
        indices (list): List of column indices of X retained.
    """
    X = np.asarray(X)
    p = np.asarray(p).flatten()
    d, L = X.shape
    if L == 0 or p.size != L:
        raise ValueError("Invalid shapes: X must have L columns and p must have length L.")
    indices = list(range(L))

    while len(indices) > d + 1:
        X_sub = X[:, indices]
        p_sub = p[indices]
        lambda_vec = find_zero_sum_combination(X_sub)
        if lambda_vec is None or np.all(np.abs(lambda_vec) < tol):
            # No nontrivial combination found, break
            break
        neg_idx = np.where(lambda_vec < -tol)[0]
        if neg_idx.size == 0:
            break
        t_vals = -p_sub[neg_idx] / lambda_vec[neg_idx]
        t = np.min(t_vals)
        min_idx = np.argmin(t_vals)
        i_zero = neg_idx[min_idx]
        p_sub_updated = p_sub + t * lambda_vec
        # Ensure no negative weights (numerical stability)
        p_sub_updated = np.where(p_sub_updated < tol, 0, p_sub_updated)
        p[indices] = p_sub_updated
        del indices[i_zero]
    X_reduced = X[:, indices]
    p_reduced = p[indices]
    return X_reduced, p_reduced, indices

def angles(K: Union[List[int], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a list of integer eigenvalues K, compute angles whose phase states resolve the identity
    in fractions of 2*pi.

    Parameters:
        K (list or np.ndarray): List or array of integer eigenvalues.

    Returns:
        ang (np.ndarray): 1D array of angles (fractions of 2*pi), shape (<=len(K)+1,)
        r (np.ndarray): 1D array, shape (same as ang), (length of K) * p_reduced
    """
    K = np.asarray(K)
    if K.size == 0:
        raise ValueError("K must not be empty.")
    k_min = np.min(K)
    K_z = K - k_min  # anchor to zero
    D = abs_diff_set(K_z)
    q = smallest_non_divisor(D)
    X = qft_projectors(K_z, q)  # Should return (n**2, q)
    p = (1 / q) * np.ones(q)
    _, p_reduced, indices = caratheodory_reduce(X, p)
    ang = np.array(indices) / q
    r = len(K_z) * p_reduced
    return ang, r

if __name__ == "__main__":
    # Example usage
    K = [2, 5, 7]
    ang, r = angles(K)
    print("Angles (fractions of 2*pi):", ang)
    print("r:", r)