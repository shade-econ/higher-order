"""Reuse the factorization in core.interp when fitting new values on fixed grids.

core.interp(x, y) builds and solves A q = y for the B-spline coefficients q.
The matrix A depends only on x. Here factor_grid prepares its factorization,
and solve reuses it for each new array of values, giving the same coefficients
up to roundoff. A single grid or separate grids for different splines can be used.

The Spline class in fixed_grid.py retains the factors between calls.
"""

from typing import NamedTuple

import numpy as np
from numba import njit

from . import core


class Factorization(NamedTuple):
    """Endpoint equations and factored tridiagonal interiors for fixed grids.

    The final axis indexes grids. The first axis indexes endpoint terms in
    left/right, and the coefficients q[2:-2] in the other three arrays.
    """

    left: np.ndarray  # (b0,b1,b2,b3,left_ratio) per grid
    right: np.ndarray  # (e0,e1,e2,e3,right_ratio) per grid
    inverse_diagonal: np.ndarray  # Reciprocal pivots after elimination.
    upper: np.ndarray  # Upper diagonal of the reduced system.
    elimination: np.ndarray  # Multipliers for subtracting the previous row.


@njit
def factor_grid(grids):
    """Cache the matrix construction and elimination done inside core.interp.

    Recompute only when the grids change. Endpoint equations in left/right
    leave a tridiagonal interior q[2:-2], stored as reusable Thomas factors.
    """
    # q[0]=y[0] and q[-1]=y[-1]. Eliminate q[1] and q[-2] from
    # the neighboring equations, leaving a tridiagonal system for q[2:-2].
    G, N = grids.shape
    M = N-4
    left = np.empty((G, 5))
    right = np.empty((G, 5))
    diagonal = np.empty((G, M))
    upper = np.empty((G, M))
    elimination = np.zeros((G, M))

    for s in range(G):
        grid = grids[s]
        knots = core.make_knots(grid)
        # Equations at grid[2:-2], with lower, diagonal, and upper entries.
        system = np.empty((M, 3))
        # At knots[i], the fourth B-spline basis value is zero.
        for i in range(4, N):
            system[i-4] = core.val_bsplines_scalar_known_i(
                knots, i, knots[i])[:3]

        b0, b1, b2, b3 = core.val_bsplines_scalar_known_i(
            knots, 3, grid[1])
        e0, e1, e2, e3 = core.val_bsplines_scalar_known_i(
            knots, N-1, grid[-2])
        # y[1] = b0*q[0]+b1*q[1]+b2*q[2]+b3*q[3]; the right
        # equation uses e0,...,e3 on q[-4:]. Remove q[1] and q[-2].
        left_ratio = system[0, 0]/b1
        right_ratio = system[-1, -1]/e2
        system[0, 1] -= left_ratio*b2
        system[0, 2] -= left_ratio*b3
        system[-1, 0] -= right_ratio*e0
        system[-1, 1] -= right_ratio*e1

        left[s] = b0, b1, b2, b3, left_ratio
        right[s] = e0, e1, e2, e3, right_ratio
        diagonal[s] = system[:, 1]
        upper[s] = system[:, 2]
        for i in range(1, M):
            elimination[s, i] = system[i, 0]/diagonal[s, i-1]
            diagonal[s, i] -= elimination[s, i]*upper[s, i-1]

    return Factorization(left.T.copy(), right.T.copy(), (1/diagonal).T.copy(),
                         upper.T.copy(), elimination.T.copy())


@njit
def solve(factor, values):
    """Fit a tuple of K value arrays using the supplied grid factorization.

    values is a tuple of K arrays, each with shape (S, N).
    For example, solve(factor, (y,)) fits one array; solve(factor, (y, z))
    fits two. Each row is fitted independently on its corresponding grid.

    The returned coefficients have shape (K, S, N). For row s using grid x,
    q[k, s] agrees with core.interp(x, values[k][s]).
    """
    K = len(values)
    S, N = values[0].shape
    # Start with the right-hand sides y, then overwrite them with coefficients.
    # The inner loops run over s, so store q[k, i, :] contiguously in memory.
    q = np.empty((K, N, S), dtype=values[0].dtype)
    # by/ey hold the endpoint right-hand sides after subtracting the known
    # first and last coefficients.
    by = np.empty((K, S), dtype=values[0].dtype)
    ey = np.empty_like(by)
    common_grid = factor.inverse_diagonal.shape[1] == 1
    for k in range(K):
        for i in range(N):
            for s in range(S):
                q[k, i, s] = values[k][s, i]

    # Eliminate the two near-endpoint coefficients from the interior system.
    for s in range(S):
        g = 0 if common_grid else s
        b0, b1, b2, b3, bratio = factor.left[:, g]
        e0, e1, e2, e3, eratio = factor.right[:, g]
        for k in range(K):
            by[k, s] = q[k, 1, s]-b0*q[k, 0, s]
            ey[k, s] = q[k, -2, s]-e3*q[k, -1, s]
            q[k, 2, s] -= bratio*by[k, s]
            q[k, -3, s] -= eratio*ey[k, s]

    # Forward elimination for the interior coefficients q[:, 2:-2, :].
    for i in range(1, N-4):
        for k in range(K):
            for s in range(S):
                g = 0 if common_grid else s
                q[k, i+2, s] -= factor.elimination[i, g]*q[k, i+1, s]

    # Back substitution, starting with the last interior coefficient.
    for k in range(K):
        for s in range(S):
            g = 0 if common_grid else s
            q[k, -3, s] *= factor.inverse_diagonal[-1, g]
    for i in range(N-6, -1, -1):
        for k in range(K):
            for s in range(S):
                g = 0 if common_grid else s
                q[k, i+2, s] = (
                    q[k, i+2, s]-factor.upper[i, g]*q[k, i+3, s]
                )*factor.inverse_diagonal[i, g]

    # Recover the two near-endpoint coefficients.
    for s in range(S):
        g = 0 if common_grid else s
        b0, b1, b2, b3, bratio = factor.left[:, g]
        e0, e1, e2, e3, eratio = factor.right[:, g]
        for k in range(K):
            q[k, 1, s] = (by[k, s]-b2*q[k, 2, s]-b3*q[k, 3, s])/b1
            q[k, -2, s] = (ey[k, s]-e0*q[k, -4, s]-e1*q[k, -3, s])/e2
    return np.ascontiguousarray(q.transpose(0, 2, 1))
