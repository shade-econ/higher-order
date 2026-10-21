"""Replace analytical_simple's six sparse matrices with shared sparse storage.

In analytical_simple/expectations.py, ExpectationMaps holds six separate SciPy
sparse matrices, applied in separate matrix-vector products. Here ExpectationMaps
stores their row and column indices once, with six weights at each location.
perturbation_expectations accumulates the required products together in one
compiled loop, reusing indices and coefficient values across the products.

The quadrature and additional moments functions are imported unchanged from
analytical_simple. See that module for the definitions of the expectations.
"""

from typing import NamedTuple

import numpy as np
from numba import njit

from analytical_simple.expectations import (
    cash_expectation_matrices, asset_expectation_matrices, marginal_value_moments,
    asset_density_moments, lognormal_density_derivatives)


class ExpectationMaps(NamedTuple):
    """The same six expectation matrices, stored with common sparse indices.

    Row i uses entries p=indptr[i]:indptr[i+1]. At each entry, indices[p]
    selects a coefficient and weights[p] contains the six matrix weights in
    the simple variant's order: value, first, z_first, second, z_second, z2_second.
    shape=(S,M) restores the query dimensions after applying the matrices.
    """
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    shape: tuple


def sparse_expectation_maps(matrices):
    """Store the quadrature builders' (6,S,M,N) array as shared sparse maps.

    Each s is an independent M-by-N block: query (s,i) uses coefficients
    (s,j). Retain a location if any of its six weights is nonzero.
    """
    _, S, M, N = matrices.shape
    # np.nonzero visits entries in (s,i,j) order, keeping each row together.
    s, i, j = np.nonzero(np.any(matrices != 0, axis=0))
    rows = s*M+i
    indptr = np.concatenate((np.array([0]), np.bincount(rows, minlength=S*M).cumsum()))
    indices = s*N+j
    weights = np.ascontiguousarray(matrices[:, s, i, j].T)
    return ExpectationMaps(indptr, indices, weights, (S, M))


# Allow fused multiply-add without reassociating the sums.
@njit(fastmath={'contract'})
def perturbation_expectations(maps, q1, q2, q3=None):
    """Integrate perturbations dg,d2g,d3g with spline coefficients q1,q2,q3.

    Each coefficient array has shape (S,N). Return expectations with shape
    (10,S,M), ordered as:
        E[dg], E[d2g], E[d3g],
        E[dg_x], E[z*dg_x], E[d2g_x], E[z*d2g_x],
        E[dg_xx], E[z*dg_xx], E[z²*dg_xx].

    Without q3, return shape (4,S,M): E[dg], E[d2g], E[dg_x], E[z*dg_x].
    d differentiates the perturbation parameter at fixed x; subscripts x
    differentiate the spline. The caller applies the chain rule for changes in x.
    """
    S, M = maps.shape
    q1, q2 = q1.ravel(), q2.ravel()
    third_order = q3 is not None
    if third_order:
        q3_flat = q3.ravel()
    n_values = 3 if third_order else 2
    out = np.empty((10 if third_order else 4, S*M))
    for row in range(S*M):
        E_dg = E_d2g = E_d3g = 0.0
        E_dg_x = E_z_dg_x = E_d2g_x = E_z_d2g_x = 0.0
        E_dg_xx = E_z_dg_xx = E_z2_dg_xx = 0.0
        # Accumulate every product for this row using the same coefficient columns.
        for p in range(maps.indptr[row], maps.indptr[row+1]):
            column = maps.indices[p]
            c1, c2 = q1[column], q2[column]
            value, first, z_first, second, z_second, z2_second = maps.weights[p]
            E_dg += value*c1
            E_d2g += value*c2
            E_dg_x += first*c1
            E_z_dg_x += z_first*c1
            if third_order:
                E_d3g += value*q3_flat[column]
                E_d2g_x += first*c2
                E_z_d2g_x += z_first*c2
                E_dg_xx += second*c1
                E_z_dg_xx += z_second*c1
                E_z2_dg_xx += z2_second*c1

        out[0, row], out[1, row] = E_dg, E_d2g
        out[n_values, row], out[n_values+1, row] = E_dg_x, E_z_dg_x
        if third_order:
            out[2, row] = E_d3g
            out[5, row], out[6, row] = E_d2g_x, E_z_d2g_x
            out[7, row], out[8, row], out[9, row] = E_dg_xx, E_z_dg_xx, E_z2_dg_xx
    return out.reshape(out.shape[0], S, M)
