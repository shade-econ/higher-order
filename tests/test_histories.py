"""Check sequence_space.histories against market clearing and nonlinear paths.

Also check serial/parallel agreement and the derivatives needed for moments.
"""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import lu_factor, lu_solve

from sequence_space.histories import ge_derivatives, second_histories, third_histories, shifted
from sequence_space.jacobian import income_jacobian_analytical


@pytest.fixture(scope='module')
def histories(fast_derivatives):
    b, f = fast_derivatives
    T, L, rho = 12, 3, 0.95
    J_lu = lu_factor(income_jacobian_analytical(b, f, T))
    u = np.zeros((T, 3))
    u[:, 1] = lu_solve(J_lu, rho**np.arange(T))
    quadratic = second_histories(b, f, u, L, J_lu=J_lu)
    cubic, Va_xyy = third_histories(b, f, u, L, d2Z=quadratic, J_lu=J_lu)
    return b, f, u, L, J_lu, quadratic, cubic, Va_xyy


def test_ge_path_market_clearing(fast_derivatives):
    """The order-by-order GE path solve matches the prescribed debt response."""
    b, f = fast_derivatives
    T, rho = 12, 0.95
    J_lu = lu_factor(income_jacobian_analytical(b, f, T))
    debt = rho**np.arange(T)
    response = ge_derivatives(b, f, debt, J_lu)
    assert_allclose(response['dA'], debt, atol=1e-11)
    assert_allclose(response['d2A'], 0., atol=1e-11)
    assert_allclose(response['d3A'], 0., atol=1e-10)


@pytest.mark.parametrize('only_moments', [False, True])
def test_parallel_histories(histories, only_moments):
    """Both execution modes give the same cubic and future-risk derivatives."""
    b, f, u, L, J_lu, quadratic, cubic, Va_xyy = histories
    if only_moments:
        cubic, Va_xyy = third_histories(b, f, u, L, d2Z=quadratic, J_lu=J_lu,
                                       only_moments=True)
    parallel, parallel_Va = third_histories(b, f, u, L, d2Z=quadratic, J_lu=J_lu,
                                           workers=2, only_moments=only_moments)
    assert_allclose(parallel, cubic, rtol=1e-13, atol=1e-13)
    assert_allclose(parallel_Va, Va_xyy, rtol=1e-13, atol=1e-13)


def test_moment_histories(histories):
    """Skipping three distinct shock dates preserves repeated-date and risk derivatives."""
    b, f, u, L, J_lu, quadratic, cubic, Va_xyy = histories
    reduced, reduced_Va = third_histories(b, f, u, L, d2Z=quadratic, J_lu=J_lu,
                                         only_moments=True)
    for j in range(L):
        for k in range(L):
            # Ages are t, t+j, t+k: a zero lag or equal lags repeat a shock date.
            expected = cubic[:, j, k] if j == 0 or k == 0 or j == k else 0.
            assert_allclose(reduced[:, j, k], expected, rtol=1e-13, atol=1e-13)
    assert_allclose(reduced_Va, Va_xyy, rtol=1e-13, atol=1e-13)


def test_reused_histories_nonlinear(household, histories):
    """Feed first- and second-order income responses into nonlinear household paths.
    The remaining mixed asset response determines the cubic income correction.

    Each first- and second-order income response is zero after its T dates.
    The distribution it leaves behind continues to propagate after that cutoff.
    """
    hh, ss = household
    b, f, inputs, L, J_lu, quadratic, cubic, Va_xyy = histories
    T = len(inputs)
    u = inputs[:, 1]
    own, cross1, cross2 = quadratic[:, 0], quadratic[:, 1], quadratic[:, 2]

    # Supply income responses through second order and inherit F at each surprise.
    def asset_path(x, y, z):
        """Return assets from the third surprise onward; x, y, z arrive at dates 0, 1, 2."""
        # Date 0: only x is known. Carry its distribution forward to date 1.
        income = x*shifted(u, 0, T+2)+x*x*shifted(own, 0, T+2)/2
        old = hh.transition(ss, dZ=income, return_F=True)
        F1 = old['F'][1]

        # Date 1: y arrives, interacting with the one-period-old shock x.
        income = (x*shifted(u, 1, T+1)+y*shifted(u, 0, T+1)
                  +(x*x*shifted(own, 1, T+1)+y*y*shifted(own, 0, T+1))/2
                  +x*y*shifted(cross1, 0, T+1))
        middle = hh.transition(ss, dZ=income, F0=F1, return_F=True)
        F2 = middle['F'][1]

        # Date 2: z arrives; its gaps to x and y are two and one periods.
        income = (x*shifted(u, 2, T)+y*shifted(u, 1, T)+z*u
                  +(x*x*shifted(own, 2, T)+y*y*shifted(own, 1, T)+z*z*own)/2
                  +x*y*shifted(cross1, 1, T)+x*z*cross2+y*z*cross1)
        return hh.transition(ss, dZ=income, F0=F2)['A']

    errors = []
    for step in (0.004, 0.002):
        mixed = sum(x*y*z*asset_path(step*x, step*y, step*z)
                    for x, y, z in product((-1, 1), repeat=3))/(8*step**3)
        # Third-order income must offset the mixed asset response to these inputs.
        numerical = lu_solve(J_lu, -mixed)
        # Relative to the newest shock, the earlier shocks arrived 2 and 1 dates ago.
        errors.append(np.max(abs(numerical-cubic[:, 2, 1])))
    assert errors[1] < errors[0]/2
    assert errors[1] < 0.003*np.max(abs(cubic[:, 2, 1]))
