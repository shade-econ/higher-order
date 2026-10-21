"""Precomputed matrices for taking expectations of functions represented by splines.

A spline with coefficients q is f(x)=sum_j q[j]*B_j(x). Taking expectations
of each basis function B_j gives a matrix that integrates f:

    M[i,j] = E[B_j(X_i)]
    (M @ q)[i] = E[f(X_i)]

Here X_i is a shifted and scaled lognormal random variable, with its
distribution specified by the function constructing the matrix.

Each row i computes an expectation under one distribution of X_i; each
column corresponds to a spline basis function. With the knots and distributions
fixed, the same matrix works for any coefficients q. ExpectationMaps also
includes matrices for weighted first and second derivatives. When an integral
is restricted to an interval, E[f(X)] means E[f(X)*1{X in interval}], without
dividing by the probability of that interval.

The additional moments routines evaluate two other integrals for fixed steady-state
functions: a nonlinear transformation of the consumption spline, and CDF
derivatives with a density weight absent from the six matrices above.
"""

from typing import NamedTuple

import numpy as np
from numba import njit
from scipy.sparse import block_diag, csc_matrix

from smooth_sim import quadrature
from spline import core as spline
from spline.fixed_grid import basis_derivative_weights
from smooth_sim import utils


"""Building matrices that map spline coefficients to weighted expectations."""

class ExpectationMaps(NamedTuple):
    """Matrices taking spline coefficients q to weighted expectations.

    For f(x)=sum_j q[j]*B_j(x), the six products with q give
    E[f], E[f'], E[z*f'], E[f''], E[z*f''], and E[z²*f''].
    The integration routine specifies the random location x and weight z.
    Integrals over a restricted interval are not renormalized.
    """

    value: csc_matrix
    first: csc_matrix
    z_first: csc_matrix
    second: csc_matrix
    z_second: csc_matrix
    z2_second: csc_matrix


@njit
def cash_expectation_matrices(m_grid, b, Z, mu, sigma, rule=quadrature.start(40)):
    """Build spline expectation matrices for m=b+Z*eta, log(eta)~N(mu[s],sigma²).

    Integrate only m >= m_grid[s,0], using eta as the multiplier z.
    Each s uses its own grid m_grid[s] and log mean mu[s]; b, Z, and sigma
    are shared. Return (6,S,M,N) in ExpectationMaps order, with M values
    of b and N spline coefficients per grid. No spline values are needed.
    """
    S, N = m_grid.shape
    matrices = np.zeros((6, S, len(b), N))
    for s in range(S):
        knots = spline.make_knots(m_grid[s])
        bounds = np.concatenate((m_grid[s, :-1], np.array([np.inf])))
        for j in range(len(b)):
            # Build the rows for this b[j] by adding integrals over spline pieces.
            # Within each piece the basis functions and their derivatives are smooth.
            for piece in range(len(bounds)-1):
                w, m = utils.integrate_lognormal_interval(
                    b[j], mu[s]+np.log(Z), sigma, bounds[piece], bounds[piece+1], rule)
                if len(m) == 0:
                    continue
                # Recover eta at these quadrature points for the eta-weighted expectations.
                eta = (m-b[j])/Z
                # Convert this piece's quadrature to coefficient weights and add them
                # to the six rows computing expectations at b[j].
                matrices[:, s, j] += quadrature_to_spline_weights(knots, m, w, eta)
    return matrices


@njit
def asset_expectation_matrices(grid, a_shift, lam, mu, sigma, rule=quadrature.start(40)):
    """Build spline expectation matrices for a=a_shift-lam*eps, log(eps)~N(mu[s],sigma²).

    At query (s,j), a=a_shift[s,j]-lam[s]*eps, with lam[s] > 0.
    Integrate only grid[0] <= a <= grid[-1], using a as the multiplier z.
    Return (6,S,M,N) in ExpectationMaps order: S states, M queries per
    state, and N coefficients on the common grid.
    """
    S, M = a_shift.shape
    matrices = np.zeros((6, S, M, len(grid)))
    knots = spline.make_knots(grid)
    for s in range(S):
        for j in range(M):
            # Build the rows for this a_shift[s,j] by adding integrals over spline pieces.
            # Within each piece the basis functions and their derivatives are smooth.
            for piece in range(len(grid)-1):
                # a decreases with eps, reversing the integration limits.
                eps_low = max(0.0, (a_shift[s, j]-grid[piece+1])/lam[s])
                eps_high = (a_shift[s, j]-grid[piece])/lam[s]
                w, eps = utils.integrate_lognormal_interval(
                    0.0, mu[s], sigma, eps_low, eps_high, rule)
                if len(eps) == 0:
                    continue
                # Recover a at these quadrature points for the a-weighted expectations.
                a = a_shift[s, j]-lam[s]*eps
                # Convert this piece's quadrature to coefficient weights and add them
                # to the six rows computing expectations at a_shift[s,j].
                matrices[:, s, j] += quadrature_to_spline_weights(knots, a, w, a)
    return matrices


def sparse_expectation_maps(maps):
    """Convert six arrays of shape (S,M,N) into the sparse ExpectationMaps.

    Each array holds S matrices of shape (M,N). Placing them along the
    diagonal gives one (S*M,S*N) matrix acting on all S splines at once.
    The array order is value, first, z_first, second, z_second, z2_second.
    """
    return ExpectationMaps(*(
        block_diag([csc_matrix(block) for block in matrix], format="csc")
        for matrix in maps))


@njit(inline="always")
def quadrature_to_spline_weights(knots, x, w, z):
    """Convert a quadrature rule into weights on cubic B-spline coefficients.

    The caller supplies nodes x and weights w so that w @ f(x) approximates
    the desired integral of f. This function expresses those quadrature sums
    in terms of f's spline coefficients, also for derivatives and products.

    Write f(t)=sum_j q[j]*B_j(t), using core.py's knots from make_knots(grid)
    and coefficients q from interp(grid,y). Return a matrix A such that:

        A @ q = [w @ f(x),
                 w @ f'(x),   w @ (z*f'(x)),
                 w @ f''(x),  w @ (z*f''(x)),  w @ (z**2*f''(x))]

    where z is an additional multiplier at each node. A has shape (6,N),
    with one column per basis function: A[0,j]=sum_n w[n]*B_j(x[n]).
    """
    matrices = np.zeros((6, len(knots)-4))
    value, first, z_first, second, z_second, z2_second = matrices
    unit_coefficients = np.zeros(matrices.shape[1])
    for node in range(len(x)):
        interval = spline.locate(knots, x[node])
        values = spline.weights_known_i(knots, interval, x[node])
        d1, d2 = basis_derivative_weights(unit_coefficients, knots, interval, x[node])

        # values, d1, d2 contain B_j, B_j', B_j'' at x[node],
        # for the four indices j=interval-3,...,interval in that order.
        for local in range(4):
            column = interval-3+local
            value[column] += w[node]*values[local]
            first[column] += w[node]*d1[local]
            z_first[column] += w[node]*d1[local]*z[node]
            second[column] += w[node]*d2[local]
            z_second[column] += w[node]*d2[local]*z[node]
            z2_second[column] += w[node]*d2[local]*z[node]**2
    return matrices


"""Additional moments of fixed steady-state functions."""

@njit
def marginal_value_moments(q_c, m_grid, b, Z, mu, sigma, gamma,
                           rule=quadrature.start(40)):
    """Integrate derivatives of g(m)=c(m)**(-gamma), with m=b+Z*eta.

    log(eta) ~ N(mu[s],sigma²). Below the first endogenous cash point
    m_grid[s,0], the borrowing constraint binds and c(m)=m. Above it,
    consumption follows the unconstrained spline q_c[s]. Return:

        E_g_m[k]   = E[eta**k * g_m],    k=0,1
        E_g_mm[k]  = E[eta**k * g_mm],   k=0,1,2
        E_g_mmm[k] = E[eta**k * g_mmm],  k=0,1,2,3

    These moments integrate both regions and have shape (powers,S,M),
    with S consumption splines and M values of b. Subscripts m denote
    derivatives with respect to cash on hand within each region; borrowing-
    threshold corrections are computed separately in Backward.
    """
    # One set of moments for each consumption spline and each value of b.
    S, M = m_grid.shape[0], len(b)
    E_g_m = np.zeros((2, S, M))
    E_g_mm = np.zeros((3, S, M))
    E_g_mmm = np.zeros((4, S, M))
    for s in range(S):
        # Prepend the constrained interval [0, m_grid[s,0]] to the cash spline
        # pieces. As above, the final spline piece extends to infinity.
        bounds = np.concatenate((np.array([0.]), m_grid[s, :-1], np.array([np.inf])))
        for j in range(M):
            # Build the moments for this b[j] by adding each region's integral.
            # Split the unconstrained region at knots because c''' jumps there.
            for piece in range(len(bounds)-1):
                w, m = utils.integrate_lognormal_interval(
                    b[j], mu[s]+np.log(Z), sigma, bounds[piece], bounds[piece+1], rule)
                if len(m) == 0:
                    continue

                # Evaluate marginal-value derivatives within this region.
                if piece == 0:
                    # Constrained: save zero, consume m, so g(m)=m**(-gamma).
                    g_m = -gamma*m**(-gamma-1)
                    g_mm = gamma*(gamma+1)*m**(-gamma-2)
                    g_mmm = -gamma*(gamma+1)*(gamma+2)*m**(-gamma-3)
                else:
                    # Unconstrained: evaluate the consumption spline and apply
                    # the chain rule to g(m)=c(m)**(-gamma).
                    c, c_m, c_mm, c_mmm = spline.val_monotonic_with_der3(
                        q_c[s], m_grid[s], m)
                    g_c = -gamma*c**(-gamma-1)
                    g_cc = gamma*(gamma+1)*c**(-gamma-2)
                    g_ccc = -gamma*(gamma+1)*(gamma+2)*c**(-gamma-3)
                    g_m = g_c*c_m
                    g_mm = g_cc*c_m**2+g_c*c_mm
                    g_mmm = g_ccc*c_m**3+3*g_cc*c_m*c_mm+g_c*c_mmm

                # Weight these derivatives by powers of eta and add this piece.
                eta = (m-b[j])/Z
                for power in range(2):
                    E_g_m[power, s, j] += w @ (g_m*eta**power)
                for power in range(3):
                    E_g_mm[power, s, j] += w @ (g_mm*eta**power)
                for power in range(4):
                    E_g_mmm[power, s, j] += w @ (g_mmm*eta**power)
    return E_g_m, E_g_mm, E_g_mmm


@njit
def asset_density_moments(grid, coefficients, a_shift, lam, mu, sigma, rule=quadrature.start(40)):
    """Return E[a**k * f_aa(a) * p'(eps)/p(eps)] for k=0,1,2,3.

    At query (s,j), a=a_shift[s,j]-lam[s]*eps, log(eps)~N(mu[s],sigma²),
    and lam[s] > 0. The spline f uses coefficients[s] on grid; f_aa is
    its second derivative with respect to a. Here p is the density of eps,
    and p' is its derivative with respect to eps.

    Used for the steady-state CDF in the forward step's integration by parts.

    Integrate only grid[0] <= a <= grid[-1], without renormalizing.
    The result has shape (4,S,M), with S splines and M queries per spline.
    """
    S, M = a_shift.shape
    E_f_aa_dlogp = np.zeros((4, S, M))
    for s in range(S):
        for j in range(M):
            # Build the moments for this a_shift[s,j] by adding integrals over spline pieces.
            # Within each piece f_aa is linear; its slope can jump at knots.
            for piece in range(len(grid)-1):
                # a decreases with eps, reversing the integration limits.
                eps_low = max(0.0, (a_shift[s, j]-grid[piece+1])/lam[s])
                eps_high = (a_shift[s, j]-grid[piece])/lam[s]
                w, eps = utils.integrate_lognormal_interval(
                    0.0, mu[s], sigma, eps_low, eps_high, rule)
                if len(eps) == 0:
                    continue
                a = a_shift[s, j]-lam[s]*eps

                # Evaluate in increasing a, then restore the quadrature order.
                _, _, f_aa = spline.val_monotonic_with_der2(coefficients[s], grid, a[::-1])
                f_aa = f_aa[::-1]

                # Weight by p'/p and powers of a, then add this piece's moments.
                dlogp = -(1+(np.log(eps)-mu[s])/sigma**2)/eps
                for power in range(4):
                    E_f_aa_dlogp[power, s, j] += w @ (f_aa*dlogp*a**power)
    return E_f_aa_dlogp


"""Lognormal density and its derivatives, used by both differentiation steps."""

@njit
def lognormal_density_derivatives(x, mu, sigma):
    """Return p(x), p'(x), p''(x) for log(X)~N(mu,sigma²).

    Derivatives are with respect to x, holding mu and sigma fixed.
    The backward and forward steps use them at borrowing thresholds.
    """
    p = utils.lognormal_density(x, mu, sigma)
    if p == 0.0:
        return 0.0, 0.0, 0.0
    ell = 1+(np.log(x)-mu)/sigma**2
    return p, -p*ell/x, p*(ell**2+ell-1/sigma**2)/x**2
