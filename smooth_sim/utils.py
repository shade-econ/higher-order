"""Quadrature and distribution weights for the smooth model.

The expectation integrators use the Gaussian rules below. integrate_*_interval
returns unnormalized probability weights: w @ f(x) approximates the integral
on the requested interval, including its probability mass.
"""

import numpy as np
from numba import njit
import math

from . import quadrature
from spline import core as spline
from discrete_sim.sim_steady_state import stationary_markov
Leg = quadrature.start(40)
Leg_sm = quadrature.start(3)


@njit
def integrate_lognormal_interval(a, mu, sigma, x_l, x_h, rule=Leg):
    """Return weights w and nodes x for a shifted lognormal integral.

        x        = a + exp(Y),     Y ~ N(mu, sigma**2)
        w @ F(x) = E[F(x) 1{x_l <= x <= x_h}]  (quadrature approximation).

    Integrate in log income using the normal rule below, truncated at 8 sigma.
    An interval outside the distribution's support returns empty arrays.
    """
    if x_h <= a:
        return np.empty(0), np.empty(0)

    logy_l, logy_h = log_with_inf(x_l - a), log_with_inf(x_h - a)
    w, logy = integrate_normal_interval(mu, sigma, logy_l, logy_h, rule)
    x = a + np.exp(logy)
    return w, x


@njit
def integrate_normal_interval(mu, sigma, x_l, x_h, rule=Leg):
    """Return w,x with w @ F(x) approximating E[F(X) 1{x_l <= X <= x_h}].

    X ~ N(mu, sigma**2). Use Gauss-Legendre quadrature (40 points by default) on the
    interval intersected with mu +/- 8*sigma, weighted by the normal density.
    """
    aquad = max(mu - 8*sigma, x_l)
    bquad = min(mu + 8*sigma, x_h)
    if aquad >= bquad:
        return np.empty(0), np.empty(0)
    else:
        w, x = quadrature.interval(rule, aquad, bquad)
        return w * normal_pdf(x, mu, sigma), x


@njit
def integrate_normal_common_lower(mu, sigma, x_l, x_h):
    """Return normal quadrature rules for intervals sharing a lower bound.

    Column i of w,x integrates E[F(X) 1{x_l <= X <= x_h[i]}], with
    X ~ N(mu, sigma**2), using the same rule as integrate_normal_interval.
    Empty intervals have zero weights and retain their column positions.
    """
    aquad = max(mu - 8*sigma, x_l)
    bquad = np.minimum(mu + 8*sigma, x_h)

    # Keep empty intervals as zero-weight columns corresponding to x_h.
    bquad = np.maximum(aquad, bquad)
    w, x = quadrature.interval(Leg, aquad, bquad[:, np.newaxis])

    # Transpose so column i contains the rule for [x_l, x_h[i]].
    w = (w * normal_pdf(x, mu, sigma)).T.copy()
    x = x.T.copy()
    return w, x


@njit
def log_with_inf(x):
    """Convert a lognormal integration bound to log space; x<=0 means -inf."""
    return np.log(x) if x > 0 else -np.inf


@njit
def normal_pdf(x, mu, sigma):
    """Evaluate the N(mu, sigma**2) density at scalar or array x."""
    return np.exp(-((x-mu)/sigma)**2/2)/np.sqrt(2*np.pi)/sigma


@njit
def lognormal_density(x, mu, sigma):
    """Density of exp(N(mu,sigma^2)); quadrature truncation is separate."""
    if x <= 0:
        return 0.0
    log_x = np.log(x)
    return normal_pdf(log_x, mu, sigma)/x


@njit
def normal_cdf(x, mu, sigma):
    """Evaluate Pr(X<=x) for X ~ N(mu, sigma**2), with scalar x."""
    return 0.5*(1 + math.erf((x-mu)/sigma/np.sqrt(2)))


def get_Pi_F(Pi):
    """Return Pi_F[s_next, s] = Pr(s_t=s | s_{t+1}=s_next) under stationary state shares."""
    pi = stationary_markov(Pi)
    return Pi.T * pi / pi[:, np.newaxis]


@njit
def integration_weights_spline(xgrid, qF):
    """Return w such that w @ qG = integral G(x) dF(x).

    qF and qG are cubic-spline coefficients on xgrid. Include the mass F(x_min)
    at the lower endpoint. Three Gaussian nodes integrate the degree-5 pieces.
    """
    t = spline.make_knots(xgrid)

    weights = np.zeros(len(qF))

    # iterate over each piecewise interval
    for i in range(3, len(t)-4):
        # integrate over [t[i], t[i+1]]
        a, b = t[i], t[i+1]
        # get 3 quadrature points and weights on this interval
        w, x = quadrature.interval(Leg_sm, a, b)
        # find weights on qG corresponding to each quadrature point
        # and multiply by dF
        for j in range(3):
            weights[i-3:i+1] += (w[j] * spline.weights_known_i(t, i, x[j])
                    * spline.val_scalar_with_der_known_i(qF, t, i, x[j])[1])

    # if F starts above 0 (mass point), need integration weight on first point too
    weights[0] += qF[0]

    return weights
