"""Check fake-news Jacobians against direct income-shock experiments."""

import numpy as np
from numpy.testing import assert_allclose

from sequence_space.jacobian import income_jacobian_analytical, income_jacobian_numerical


def test_income_jacobian(household, simple_derivatives):
    """Compare analytical and numerical fake-news Jacobians, then check three
    columns against finite differences of complete nonlinear transitions.
    """
    hh, ss = household
    T, h = 12, 1e-4
    analytical = income_jacobian_analytical(*simple_derivatives, T)
    numerical = income_jacobian_numerical(hh, ss, T, h=h)
    # Expectation-function and CDF propagation have slightly different discretization errors.
    assert_allclose(numerical, analytical, rtol=0, atol=3e-5)

    # Both Jacobians share fake-news assembly; direct transitions check that step too.
    # Use an immediate shock, an anticipated shock, and the final-date shock.
    for s in (0, T//2, T-1):
        dZ = np.zeros(T)
        dZ[s] = h
        plus = hh.transition(ss, dZ=dZ)['A']
        minus = hh.transition(ss, dZ=-dZ)['A']
        direct = (plus-minus)/(2*h)
        assert_allclose(analytical[:, s], direct, rtol=0, atol=3e-5)
        assert_allclose(numerical[:, s], direct, rtol=0, atol=3e-5)
