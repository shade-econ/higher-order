"""Check simulation, moments, and GIRFs using a small, invented response of a series X.

We choose derivatives through third order and evaluate the resulting finite
Volterra series directly. Its expectations can be computed by Gaussian quadrature,
so we can check the stochastic calculations without solving a household model.
"""

from itertools import permutations, product

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sequence_space.stochastic import girf, mean_correction, covariance_coefficients, simulate, simulate_simple


@pytest.fixture
def volterra_series():
    """Choose derivatives of X for a short history of shocks."""
    rng = np.random.default_rng(419)
    T, L = 3, 3
    H = T+L-1

    # second[i,j] differentiates X in shocks i and j periods ago.
    # third[i,j,k] does the same for three shocks; their order cannot matter.
    first = np.zeros(H)
    first[:T] = rng.normal(size=T)
    second = rng.normal(size=(H, H))
    second = (second+second.T)/2
    third = rng.normal(size=(H, H, H))
    third = sum(third.transpose(p) for p in permutations(range(3)))/6

    # Drop terms whose most recent shock is T or more periods ago,
    # or whose earliest and latest shocks are L or more periods apart.
    for ages in np.ndindex(second.shape):
        if min(ages) >= T or max(ages)-min(ages) >= L:
            second[ages] = 0.
    for ages in np.ndindex(third.shape):
        if min(ages) >= T or max(ages)-min(ages) >= L:
            third[ages] = 0.

    # stochastic.py stores second[t,t+l] as X2[t,l], and similarly for X3.
    X2 = np.array([[second[t, t+l] for l in range(L)] for t in range(T)])
    X3 = np.array([[[third[t, t+l, t+m] for m in range(L)]
                    for l in range(L)] for t in range(T)])
    X_risk, X_sigma_sigma = .37, rng.normal(size=T)
    return first, second, third, X2, X3, X_risk, X_sigma_sigma


def series_coefficients(first, second, third, history, X_risk, X_sigma_sigma):
    """Return x1, x2, x3 in X = X_ss + sigma*x1 + sigma^2*x2 + sigma^3*x3.

    history[i] is the shock i periods ago, divided by sigma.
    """
    # Evaluate the linear, quadratic, and cubic sums, adding the risk terms.
    x1 = first @ history
    x2 = (history @ second @ history+X_risk)/2
    x3 = (np.einsum('ijk,i,j,k', third, history, history, history)/6
          +X_sigma_sigma @ history[:len(X_sigma_sigma)]/2)
    return np.array([x1, x2, x3])


def normal_quadrature(n):
    """Nodes and weights for n independent standard normal shocks.

    Three nodes per shock integrate the moments needed through sigma^4 exactly.
    """
    nodes = (-np.sqrt(3.), 0., np.sqrt(3.))
    weights = (1/6, 2/3, 1/6)
    for indices in product(range(3), repeat=n):
        yield np.array([nodes[i] for i in indices]), np.prod([weights[i] for i in indices])


def test_simulation(volterra_series):
    """Convolution and direct Volterra summation give the same paths for X."""
    first, _, _, X2, X3, X_risk, X_sigma_sigma = volterra_series
    X1 = first[:len(X2)]
    epsbar = np.array([.7, -.2, 1.1, .4, -.9, .3, -.5, .8])
    expected = simulate_simple(epsbar, X1, X2, X3,
                               X_risk, X_sigma_sigma)
    assert_allclose(simulate(epsbar, X1, X2, X3, X_risk, X_sigma_sigma), expected, rtol=2e-13, atol=2e-14)
    assert_allclose(simulate(epsbar, X1, X2, X3, X_risk, X_sigma_sigma, burnin=3), expected[:, 3:], rtol=2e-13, atol=2e-14)


def test_girf_conditional_expectation(volterra_series):
    """The GIRF equals the change in conditional expectations after a surprise."""
    first, second, third, X2, X3, X_risk, X_sigma_sigma = volterra_series
    T = len(X2)
    sigma, nu = .2, -.31
    eps = np.array([.5, -.2, .17, -.09, .12, .08])
    H = len(first)

    def response(history):
        # Convert shocks to units of sigma for the series calculation.
        terms = series_coefficients(first, second, third, history/sigma, X_risk, X_sigma_sigma)
        # Return the first-, second-, and third-order approximations to X.
        return .83+np.cumsum(terms*sigma**np.arange(1, 4))

    # Look h periods ahead: today's shock is history[h], and past shocks follow it.
    expected = np.zeros((3, T))
    for h in range(T):
        base = np.zeros(H)
        n = min(len(eps), H-h-1)
        base[h+1:h+1+n] = eps[-n:][::-1]
        after = np.zeros(3)
        before = np.zeros(3)

        # After observing today's shock nu, average over the h future shocks.
        for draws, weight in normal_quadrature(h):
            history = base.copy()
            history[:h] = sigma*draws
            history[h] = nu
            after += weight*response(history)

        # Before observing it, also average over what today's shock might be.
        for draws, weight in normal_quadrature(h+1):
            history = base.copy()
            history[:h+1] = sigma*draws
            before += weight*response(history)
        expected[:, h] = after-before

    assert_allclose(girf(eps, nu, sigma, first[:T], X2, X3, X_sigma_sigma=X_sigma_sigma, horizon=T), expected, rtol=2e-13, atol=3e-14)


def test_covariance_direct_expectations(volterra_series):
    """Covariance coefficients through sigma^4 agree with Gaussian integration."""
    first, second, third, X2, X3, X_risk, X_sigma_sigma = volterra_series
    T = len(X2)
    H = len(first)
    # Only squares of shocks have nonzero expectations in the quadratic sum.
    mean2 = (X_risk+np.trace(second))/2

    # For X = X_ss + sigma*x1 + sigma^2*x2 + sigma^3*x3, covariance through sigma^4
    # uses x1*x1, centered x2*x2, and the two x1*x3 terms.
    cov2, cov4 = [], []
    for h in (0, 1):
        linear, fourth = 0., 0.
        # The two dates share all but h shocks; h=0 gives variance, h=1 autocovariance.
        for draws, weight in normal_quadrature(H+h):
            x1, x2, x3 = series_coefficients(first, second, third, draws[:H], X_risk, X_sigma_sigma)
            x1_lag, x2_lag, x3_lag = series_coefficients(first, second, third, draws[h:h+H], X_risk, X_sigma_sigma)
            linear += weight*x1*x1_lag
            fourth += weight*((x2-mean2)*(x2_lag-mean2)+x1*x3_lag+x3*x1_lag)
        cov2.append(linear)
        cov4.append(fourth)

    Sigma2, Sigma4 = covariance_coefficients(first[:T], X2, X3, X_sigma_sigma=X_sigma_sigma)
    assert_allclose(Sigma2, cov2, rtol=2e-13, atol=2e-14)
    assert_allclose(Sigma4, cov4, rtol=2e-13, atol=2e-14)


def test_mean_direct_expectation(volterra_series):
    """Independent unit-variance shocks leave only the diagonal quadratic terms in the mean."""
    _, second, _, X2, _, X_risk, _ = volterra_series
    expected = (np.trace(second)+X_risk)/2
    assert_allclose(mean_correction(X2[:, 0], X_risk), expected, rtol=2e-13, atol=2e-14)


def test_missing_and_zero_cubic():
    """Unavailable cubic information omits Sigma4; known zero cubic terms still allow it."""
    X1, X2, X3 = np.array([1.]), np.array([[2.]]), np.zeros((1, 1, 1))
    assert_allclose(covariance_coefficients(X1, lags=(0,)), ([1.], [0.]))
    assert_allclose(covariance_coefficients(X1, X2, lags=(0,)), ([1.], [0.]))
    assert_allclose(covariance_coefficients(X1, X2, X3, lags=(0,)), ([1.], [2.]))
    assert mean_correction() == 0.
