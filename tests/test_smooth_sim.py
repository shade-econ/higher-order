"""Check income integration, expectation identities, and calibration targets."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import lognorm

from smooth_sim import core, utils
from spline import core as spline


"""Simple income integrals"""

def test_expected_marginal_utility():
    """Integrate 1/c when c=m below the constraint and c=(1+m)/2 above it."""
    sigma = 0.2
    mu = -sigma**2/2
    income = lognorm(s=sigma, scale=np.exp(mu))
    m_endog = np.linspace(1, 10, 20)
    q = spline.interp(m_endog, (1+m_endog)/2)
    b = np.array([0., 0.5, 1., 2.])
    expected = []
    for assets in b:
        cutoff = max(1-assets, 0.)
        constrained = quad(lambda y: income.pdf(y)/(assets+y), 0, cutoff, epsabs=1e-12)[0]
        unconstrained = quad(lambda y: 2*income.pdf(y)/(1+assets+y),
                             cutoff, np.inf, epsabs=1e-12)[0]
        expected.append(constrained+unconstrained)
    actual = core.expectation_lognormal_margutil(b, m_endog, q, mu, sigma, eis=1.)
    assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_expected_cash():
    """Use f(m)=m to check income integration and both below-grid extensions:
    hold f at its boundary value 1, or set it to zero, for m<1.
    """
    sigma = 0.2
    mu = -sigma**2/2
    income = lognorm(s=sigma, scale=np.exp(mu))
    m_endog = np.linspace(1, 10, 20)
    q = spline.interp(m_endog, m_endog)
    b = np.array([0., 0.5, 1., 2.])
    cutoff = np.maximum(1-b, 0.)
    expected = [quad(lambda y: (assets+y)*income.pdf(y), low, np.inf, epsabs=1e-12)[0]
                for assets, low in zip(b, cutoff)]
    for constrained in (False, True):
        actual = core.expectation_lognormal_m(b, m_endog, q, mu, sigma, constrained)
        assert_allclose(actual, np.array(expected)+constrained*income.cdf(cutoff), rtol=1e-11, atol=1e-12)


def test_cash_distribution():
    """Cash on hand is uniform assets on [0,10] plus lognormal income."""
    sigma = 0.2
    mu = -sigma**2/2
    income = lognorm(s=sigma, scale=np.exp(mu))
    b = np.linspace(0, 10, 20)
    qF = spline.interp(b, b/10)
    m = np.linspace(0.1, 8, 20)
    # Pr(assets+income <= m) = integral_0^m (m-y)/10 * income_density(y) dy.
    expected = [quad(lambda y: (cash-y)/10*income.pdf(y), 0, cash, epsabs=1e-12)[0]
                for cash in m]
    actual = core.forward_lognormal_cdf(qF, b, m, mu, sigma)
    assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


"""Expectations, marginal expectations, and forward/backward duality"""

def test_marginal_expectation(household):
    """Hold the saving policy fixed and shift cash through a transfer.
    Check that the derivative of E[f(a')] is E[f'(a') * (1-mpc)].
    """
    hh, ss = household
    a, m = hh.a_grid, ss['m_endog']
    r, Z, Tr = ss['r'], ss['Z'], ss['Tr']
    f = np.broadcast_to(np.log1p(a), ss['F'].shape).copy()
    df = np.broadcast_to(1/(1+a), f.shape).copy()
    h = 1e-4
    plus = hh.expectation_iteration(f, m, r, Z, Tr+h)
    minus = hh.expectation_iteration(f, m, r, Z, Tr-h)
    numerical = (plus-minus)/(2*h)  # A transfer shifts current cash one for one.
    # Constrained households keep saving zero, so their marginal contribution is zero.
    marginal = hh.expectation_iteration(df, m, r, Z, Tr, mult=1-ss['mpc'], constrained=False)

    # Different spline approximations agree on average, with larger local grid errors.
    error = np.abs(marginal-numerical)
    assert error.mean() < 1e-4*np.abs(numerical).mean()
    assert error.max() < 3e-3*np.abs(numerical).max()


def test_marginal_value_propagation(household):
    """Perturb next period's Va while holding aggregate inputs fixed.
    Compare today's numerical Va response with beta*(1+r)*E[dVa(a')*(1-mpc)].
    """
    hh, ss = household
    r, Z, Tr = ss['r'], ss['Z'], ss['Tr']
    dVa = ss['Va']/(1+hh.a_grid)  # A smooth shock that fades at high assets.
    h = 1e-4
    plus = hh.backward_iteration(ss['Va']+h*dVa, r, Z, Tr)[0]
    minus = hh.backward_iteration(ss['Va']-h*dVa, r, Z, Tr)[0]
    numerical = (plus-minus)/(2*h)
    # Constrained households consume all their cash, so changing future Va has no effect.
    marginal = hh.beta*(1+r)*hh.expectation_iteration(
        dVa, ss['m_endog'], r, Z, Tr, mult=1-ss['mpc'], constrained=False)

    error = np.abs(marginal-numerical)
    assert error.mean() < 1e-4*np.abs(numerical).mean()
    assert error.max() < 3e-3*np.abs(numerical).max()


def test_forward_backward_duality(household):
    """Hold policies fixed and find the asset response to an initial redistribution
    in two ways: move the distributions forward, or integrate expected future
    assets against their initial difference.
    """
    hh, ss = household
    r, Z, Tr = ss['r'], ss['Z'], ss['Tr']
    dF = 0.01*ss['F']*(1-ss['F'])  # Redistribute assets within each persistent state.
    baseline, perturbed = ss['F'], ss['F']+dF
    expected_assets = np.broadcast_to(hh.a_grid, dF.shape).copy()
    # These weights integrate a spline against the initial CDF change dF.
    weights = [utils.integration_weights_spline(hh.a_grid, spline.interp(hh.a_grid, row))
               for row in dF]

    for t in range(1, 6):
        baseline = hh.forward_iteration(baseline, ss['m_endog'], r, Z, Tr)
        perturbed = hh.forward_iteration(perturbed, ss['m_endog'], r, Z, Tr)
        direct = hh.aggregate_assets(perturbed)-hh.aggregate_assets(baseline)
        expected_assets = hh.expectation_iteration(expected_assets, ss['m_endog'], r, Z, Tr)
        predicted = sum(hh.pi[s]*(weights[s] @ spline.interp(hh.a_grid, expected_assets[s]))
                        for s in range(len(hh.e)))
        assert_allclose(direct, predicted, rtol=1e-5, atol=1e-10, err_msg=f"horizon {t}")


"""Paper calibration"""


def test_paper_calibration(household):
    """The fitted discount factors match annual wealth/GDP and the income-weighted MPC."""
    hh, ss = household
    assert_allclose(ss["A"]/4, 1., rtol=0, atol=1e-10)  # Quarterly GDP is one.

    # dZ distributes income as e*xi; divide by the aggregate income increase.
    h = 1e-4
    Cp = hh.transition(ss, dZ=np.array([h, 0.]))["C"][0]
    Cm = hh.transition(ss, dZ=np.array([-h, 0.]))["C"][0]
    income_weighted_mpc = (Cp-Cm)/(2*h*hh.e_mean)  # E[xi]=1.
    assert_allclose(income_weighted_mpc, 0.2, rtol=0, atol=1e-10)
