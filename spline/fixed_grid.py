"""Fit and differentiate cubic splines repeatedly on the same grids.

Spline stores the interpolation factors and derivative weights, so these do
not need to be recomputed when the supplied values change. Its methods return
B-spline coefficients or derivatives at the original grid points, matching
the corresponding operations in core.py up to roundoff.

A value array has one row per spline and one column per grid point: (S, N).
The S splines are independent and can share a grid or have separate grids.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit

from . import core
from .cached_solve import Factorization, factor_grid, solve


class DerivativeWeights(NamedTuple):
    """Precomputed derivatives of the cubic B-spline basis functions.

    For a spline f(x) = sum_k q[k] B_k(x), its derivatives are weighted sums
    of the same coefficients q. At each grid point, four consecutive basis
    functions contribute. For grid g with points x and coefficients q:

        f'(x[j]) = first[g, j] @ q[i:i+4]
        f''(x[j]) = second[g, j] @ q[i:i+4]

    Here i = min(max(j - 2, 0), N - 4), using core.py's not-a-knot convention.
    Both arrays have shape (G, N, 4): grids, evaluation points, coefficients.
    G is 1 for a common grid or S for a separate grid per spline.
    """

    first: np.ndarray
    second: np.ndarray


@dataclass
class Spline:
    """Prepare fixed grids once, then fit or differentiate new arrays of values.

    Each input row contains N values of one spline. The object retains
    grid-dependent work for reuse across calls.
    """

    grids: np.ndarray
    factor: Factorization
    weights: DerivativeWeights
    _coefficient_matrix: np.ndarray | None = None

    @classmethod
    def from_grid(cls, grid):
        """Fix a common grid (N,) or one grid per spline (S, N)."""
        grids = grid[None, :] if grid.ndim == 1 else grid
        return cls(grids, factor_grid(grids), grid_derivative_weights(grids))

    def coefficient_matrix(self):
        """Return the matrix mapping grid values to spline coefficients.

        For each grid x, C @ y agrees with core.interp(x, y). Each column
        is the fit to a unit vector, constructed only on the first call.
        C has shape (N, N) for a common grid or (S, N, N) for separate grids.
        """
        if self._coefficient_matrix is None:
            G, N = self.grids.shape
            C = np.empty((G, N, N))
            unit = np.zeros((G, N))
            for j in range(N):
                unit[:, j] = 1.
                C[:, :, j] = solve(self.factor, (unit,))[0]
                unit[:, j] = 0.
            self._coefficient_matrix = C[0] if G == 1 else C
        return self._coefficient_matrix

    def coefficients(self, values):
        """Fit each row of values and return its B-spline coefficients.

        For row s with grid x, this agrees with core.interp(x, values[s]).
        """
        return solve(self.factor, (values,))[0]

    def coefficients_three(self, first, second, third):
        """Return the coefficient arrays for three inputs of shape (S, N).

        This groups three calls to coefficients into one compiled solve.
        """
        q = solve(self.factor, (first, second, third))
        return q[0], q[1], q[2]

    def derivative(self, values):
        """Return the fitted spline's first derivative at the grid points.

        For row s with grid x, this agrees with core.val_der(q, x, x),
        where q = core.interp(x, values[s]).
        """
        return spline_derivative(self.factor, self.weights, values)

    def first_second_derivatives(self, values):
        """Return the fitted spline's first and second derivatives on the grid.

        For row s with grid x and q = core.interp(x, values[s]), these
        agree with core.val_monotonic_with_der2(q, x, x)[1:].
        """
        return spline_first_second_derivatives(self.factor, self.weights, values)

    def derivatives_at_lower_bound(self, coefficients):
        """Return first and second derivatives at each grid's first point.

        For a row of coefficients q on grid x, this agrees with
        core.val_monotonic_with_der2(q, x, x[:1])[1:]. Only q[:4] contributes.
        Each returned array has one entry per spline.
        """
        q = coefficients[:, :4]
        first = np.sum(self.weights.first[:, 0]*q, axis=1)
        second = np.sum(self.weights.second[:, 0]*q, axis=1)
        return first, second


@njit
def spline_derivative(factor, weights, values):
    """Fit coefficients q, then sum B'_k(x_j)*q_k at each grid point."""
    coefficients = solve(factor, (values,))[0]
    first_weights = weights.first
    derivative = np.empty_like(values)

    for s in range(len(values)):
        g = 0 if len(first_weights) == 1 else s
        for j in range(values.shape[1]):
            derivative[s, j] = 0.0
            start = min(max(j - 2, 0), values.shape[1] - 4)
            for k in range(4):
                derivative[s, j] += first_weights[g, j, k]*coefficients[s, start+k]
    return derivative


@njit
def spline_first_second_derivatives(factor, weights, values):
    """Fit q once, then sum B'_k(x_j)*q_k and B''_k(x_j)*q_k."""
    coefficients = solve(factor, (values,))[0]
    first_weights, second_weights = weights.first, weights.second
    first = np.empty_like(values)
    second = np.empty_like(values)

    for s in range(len(values)):
        g = 0 if len(first_weights) == 1 else s
        for j in range(values.shape[1]):
            first[s, j] = second[s, j] = 0.0
            start = min(max(j - 2, 0), values.shape[1] - 4)
            for k in range(4):
                coefficient = coefficients[s, start+k]
                first[s, j] += first_weights[g, j, k]*coefficient
                second[s, j] += second_weights[g, j, k]*coefficient
    return first, second


@njit
def basis_derivative_weights(unit_coefficients, knots, interval, x):
    """Return first and second derivatives of the four basis functions at x.

    interval = core.locate(knots, x) selects coefficient indices interval-3
    through interval. Setting one coefficient to 1 and the rest to 0 makes
    core.py evaluate that basis function alone.

    unit_coefficients is a zero array of length N, reused to avoid allocating
    one at every evaluation point. Each changed entry is reset to zero.
    """
    first = np.empty(4)
    second = np.empty(4)
    for local in range(4):
        index = interval-3+local
        unit_coefficients[index] = 1.0
        _, first[local], second[local] = core.val_scalar_with_der2_known_i(
            unit_coefficients, knots, interval, x)
        unit_coefficients[index] = 0.0
    return first, second


@njit
def grid_derivative_weights(grids):
    """Cache B'_k(x_j) and B''_k(x_j) at the original grid points."""
    G, N = grids.shape
    first = np.empty((G, N, 4))
    second = np.empty((G, N, 4))
    unit_coefficients = np.zeros(N)

    for g in range(G):
        knots = core.make_knots(grids[g])
        for j in range(N):
            x = grids[g, j]
            interval = core.locate(knots, x)
            first[g, j], second[g, j] = basis_derivative_weights(
                unit_coefficients, knots, interval, x)
    return DerivativeWeights(first, second)
