"""Check sequence_space.solution using numerical derivatives of the nonlinear model.

Size effects are checked in PE and GE; shock histories and anticipated-risk
terms are checked in PE. Risk enters by perturbing expected marginal value.
"""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import root

from sequence_space import stochastic
from sequence_space.solution import solve_pe, solve_ge


@pytest.fixture(scope='module')
def pe_case(household):
    """A short PE solution with persistent income, including nonzero TVP."""
    hh, ss = household
    rho = .6
    return hh, ss, rho, solve_pe(hh, ss, T=12, L=3, rho_Z=rho)


def surprises(hh, ss, shocks, rho, T):
    """Restart after each surprise, including T income dates for each innovation.
    Return the transition starting at the last surprise.
    """
    # At each date, add the new shock and carry forward the asset distribution.
    F = ss['F']
    for date in range(len(shocks)):
        income = np.zeros(T+len(shocks)-1-date)
        for when in range(date+1):
            ages = date-when+np.arange(len(income))
            active = ages < T
            income[active] += shocks[when]*rho**ages[active]
        result = hh.transition(ss, dZ=income, F0=F, return_F=True)
        F = result['F'][1]
    return result


@pytest.mark.parametrize('equilibrium', ['PE', 'GE'])
def test_size_effects(pe_case, equilibrium):
    """Size derivatives through third order agree with nonlinear PE and GE paths."""
    hh, ss, rho, solution = pe_case
    T = solution.T
    if equilibrium == 'GE':
        solution = solve_ge(hh, ss, T=T, L=3, rho_B=rho)

    # The same innovation changes income in PE and debt supply in GE.
    def X(x):
        shock = x*rho**np.arange(T)
        if equilibrium == 'PE':
            return hh.transition(ss, dZ=shock)['C']
        # Income clears the asset market for the prescribed debt path.
        def residual(income):
            return hh.transition(ss, dZ=income)['A']-ss['A']-shock
        income = root(residual, np.zeros(T), tol=1e-10).x
        assert np.max(abs(residual(income))) < 1e-10
        # Fixed government spending makes output equal consumption plus G.
        G = 1.-ss['C']
        return hh.transition(ss, dZ=income)['C']+G

    assert_allclose(solution.X_ss, X(0.), atol=1e-10)
    assert_allclose(solution.ergodic_mean(0.), X(0.), atol=1e-10)

    # Centered differences; larger steps at higher orders limit cancellation.
    h1, h2, h3 = 1e-4, 3e-4, .002
    first = (X(h1)-X(-h1))/(2*h1)
    second = (X(h2)-2*X(0.)+X(-h2))/h2**2
    third = (X(-3*h3)-8*X(-2*h3)+13*X(-h3)-13*X(h3)+8*X(2*h3)-X(3*h3))/(8*h3**3)

    assert_allclose(solution.X1, first, rtol=2e-5, atol=1e-7)
    assert_allclose(solution.X2_own, second, rtol=5e-4, atol=2e-5)
    assert_allclose(solution.X3[:, 0, 0], third, rtol=.003, atol=2e-4)


def test_mixed_histories_nonlinear(pe_case):
    """PE history effects agree with nonlinear calculations restarted at each surprise."""
    hh, ss, rho, solution = pe_case
    T, h = solution.T, .002

    # x arrives one period before y; z, when present, arrives one period after y.
    def C(x, y):
        return surprises(hh, ss, (x, y), rho, T)['C']

    # Signed differences isolate interactions between the successive surprises.
    xy = sum(x*y*C(x*h, y*h) for x, y in product((-1, 1), repeat=2))/(4*h*h)
    xyy = sum(x*(C(x*h, h)-2*C(x*h, 0.)+C(x*h, -h)) for x in (-1, 1))/(2*h**3)
    xxy = sum(y*(C(h, y*h)-2*C(0., y*h)+C(-h, y*h)) for y in (-1, 1))/(2*h**3)
    xyz = sum(x*y*z*surprises(hh, ss, (x*h, y*h, z*h), rho, T)['C']
              for x, y, z in product((-1, 1), repeat=3))/(8*h**3)

    # Tensor indices give older shock lags relative to the newest shock.
    assert_allclose(solution.X2[:, 1], xy, rtol=.003, atol=2e-5)
    assert_allclose(solution.X3[:, 1, 0], xyy, rtol=.006, atol=3e-4)
    assert_allclose(solution.X3[:, 1, 1], xxy, rtol=.006, atol=3e-4)
    assert_allclose(solution.X3[:, 2, 1], xyz, rtol=.006, atol=3e-4)


def test_risk_perturbed_steady_states(pe_case):
    """Add the marginal-value forcing from anticipated risk to the nonlinear model.
    Check its effects on steady-state consumption and the income impulse response.
    """
    hh, ss, rho, solution = pe_case
    T, h, delta = solution.T, 3e-4, 1e-4
    income = rho**np.arange(T)

    # Find the curvature of marginal value in the size of an income innovation.
    def Va(x):
        return hh.transition(ss, dZ=x*income)['Va0']
    Va00 = (Va(h)-2*Va(0.)+Va(-h))/h**2

    # Risk shifts expected Va by sigma^2*Va00/2. Use ±delta*Va00 to differentiate
    # in its coefficient; X_risk is the response before multiplying by sigma^2/2.
    stationary = [hh.steady_state(ss['r'], ss['Z'], ss['Tr'], Va_forcing=s*delta*Va00,
                                 backward_tol=1e-11, forward_tol=1e-12) for s in (-1, 1)]
    risk = (stationary[1]['C']-stationary[0]['C'])/(2*delta)

    # Keep Va00 fixed and compare income responses around the two perturbed steady states.
    responses = []
    for sign, steady in zip((-1, 1), stationary):
        forcing = np.broadcast_to(sign*delta*Va00, (T,)+Va00.shape)
        plus = hh.transition(steady, dZ=h*income, Va_forcing=forcing)['C']
        minus = hh.transition(steady, dZ=-h*income, Va_forcing=forcing)['C']
        responses.append((plus-minus)/(2*h))
    rss = (responses[1]-responses[0])/(2*delta)

    assert_allclose(solution.X_risk, risk, rtol=.002, atol=1e-6)
    assert_allclose(solution.X_rss, rss, rtol=.004, atol=2e-4)


def test_tvp_future_expectations(pe_case):
    """Check how today's income shock changes the effect of future income risk.
    Differentiate future marginal-value curvature, then measure consumption's
    response to that change in expectations.
    """
    hh, ss, rho, solution = pe_case
    T, L, h, delta = solution.T, solution.L, .002, 1e-4

    # Differentiate once in today's shock x and twice in the future shock y.
    # For risk l periods ahead, the forcing enters expectations at date l-1.
    forcing = np.zeros((T,)+ss['Va'].shape)
    for l in range(1, L):
        def Va(x, y):
            # At the future innovation's date, policies depend on the remaining
            # income path; the inherited distribution does not enter marginal value.
            income = y*rho**np.arange(T)
            income[:T-l] += x*rho**np.arange(l, T)
            return hh.transition(ss, dZ=income)['Va0']
        forcing[l-1] = sum(x*(Va(x*h, h)-2*Va(x*h, 0.)+Va(x*h, -h))
                           for x in (-1, 1))/(2*h**3)

    # Vary the coefficient of this expectation forcing to obtain X_tvp,
    # again before multiplying by sigma^2/2.
    plus = hh.transition(ss, dZ=np.zeros(T), Va_forcing=delta*forcing)['C']
    minus = hh.transition(ss, dZ=np.zeros(T), Va_forcing=-delta*forcing)['C']
    tvp = (plus-minus)/(2*delta)
    assert np.max(abs(solution.X_tvp)) > 1e-4
    assert_allclose(solution.X_tvp, tvp, rtol=.008, atol=2e-4)


@pytest.mark.parametrize('order', [1, 2, 3])
def test_solution_moments(household, order):
    """Omitting histories unnecessary for moments preserves the mean and covariances."""
    hh, ss = household
    full = solve_pe(hh, ss, T=12, L=3, rho_Z=.6, order=order)
    reduced = solve_pe(hh, ss, T=12, L=3, rho_Z=.6, order=order, only_moments=True)
    assert_allclose(reduced.mean_correction(), full.mean_correction())
    assert_allclose(reduced.covariance_coefficients(), full.covariance_coefficients())

    # Simulation and GIRFs need the full set of size and history effects.
    if order > 1:
        with pytest.raises(ValueError, match='all size and history effects'):
            reduced.simulate(np.zeros(6))
        with pytest.raises(ValueError, match='all size and history effects'):
            reduced.girf(np.zeros(6), .03, .07, horizon=4)


def test_solution_simulation(pe_case):
    """The assembled solution supplies its derivatives and risk terms to simulation."""
    _, _, _, solution = pe_case
    epsbar = np.array([.2, -.4, .7, .1, -.6, .3])
    paths = stochastic.simulate(epsbar, solution.X1, solution.X2, solution.X3,
                               solution.X_risk, solution.X_rss+solution.X_tvp)
    assert_allclose(solution.simulate(epsbar), paths)
