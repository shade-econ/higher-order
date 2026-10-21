"""Cached spline operations agree with core; core agrees with SciPy's default."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import CubicSpline

from spline import core
from spline.fixed_grid import Spline, basis_derivative_weights


@pytest.mark.parametrize('n', [6, 25])
def test_core_against_scipy(n):
    x = np.linspace(0, 2, n)**1.3
    y = np.sin(3*x) + .2*x**3
    q = core.interp(x, y)

    # Compare values and first two derivatives, including knots and extrapolation.
    xs = np.sort(np.r_[x, np.linspace(x[0]-.1, x[-1]+.1, 101)])
    reference = CubicSpline(x, y)  # Default not-a-knot boundary conditions.
    actual = core.val_monotonic_with_der2(q, x, xs)
    for order in range(3):
        assert_allclose(actual[order], reference(xs, order), rtol=1e-10, atol=1e-10)

    # Third derivatives jump at knots: core selects the left side, SciPy the right.
    midpoints = (x[:-1]+x[1:])/2
    third = core.val_monotonic_with_der3(q, x, midpoints)[3]
    assert_allclose(third, reference(midpoints, 3), rtol=1e-10, atol=1e-10)

    # Interpolation must reproduce the supplied values at the grid points.
    assert_allclose(core.val(q, x, x), y, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('n', [6, 25])
@pytest.mark.parametrize('shared_grid', [True, False])
@pytest.mark.parametrize('strided', [False, True])
def test_cached_against_core(n, shared_grid, strided):
    # Three splines, using either one common grid or separate grids.
    x = np.linspace(0, 2, n)**1.3
    grids = np.array([x, 1.2*x+.1, np.linspace(0, 2, n)**1.7-.2])
    if shared_grid:
        grids[:] = x
    spline = Spline.from_grid(x if shared_grid else grids)

    # Three value arrays shaped (spline, grid point), also testing noncontiguous views.
    rng = np.random.default_rng(17)
    fields = rng.normal(size=(3, 3, 2*n))
    fields = fields[:, :, ::2] if strided else fields[:, :, :n].copy()

    grouped = spline.coefficients_three(*fields)
    C = spline.coefficient_matrix()

    for y, grouped_q in zip(fields, grouped):
        q = spline.coefficients(y)
        d1, d2 = spline.first_second_derivatives(y)
        lower1, lower2 = spline.derivatives_at_lower_bound(q)

        # Grouped and separate calls must give the same answers.
        assert_allclose(grouped_q, q, rtol=1e-13, atol=1e-13)
        assert_allclose(spline.derivative(y), d1, rtol=1e-13, atol=1e-13)

        # Compare each row with an ordinary, uncached spline fit and evaluation.
        for s, grid in enumerate(grids):
            reference_q = core.interp(grid, y[s])
            _, reference_d1, reference_d2 = core.val_monotonic_with_der2(reference_q, grid, grid)

            assert_allclose(q[s], reference_q, rtol=1e-12, atol=1e-12)
            assert_allclose((C if shared_grid else C[s]) @ y[s], reference_q, rtol=1e-12, atol=1e-12)

            assert_allclose(d1[s], reference_d1, rtol=1e-11, atol=1e-10)
            assert_allclose(d2[s], reference_d2, rtol=1e-11, atol=1e-9)
            assert_allclose([lower1[s], lower2[s]], [reference_d1[0], reference_d2[0]], rtol=1e-11, atol=1e-9)


def test_basis_derivative_weights():
    """Check that weighting the four relevant spline coefficients gives
    the same first and second derivatives as SciPy.
    """
    x = np.linspace(0, 2, 25)**1.3
    y = np.random.default_rng(18).normal(size=len(x))
    q = core.interp(x, y)
    reference = CubicSpline(x, y)
    knots = core.make_knots(x)
    scratch = np.zeros(len(x))

    # Include grid points, points between them, and extrapolation on both sides.
    for point in np.r_[x, (x[:-1]+x[1:])/2, x[0]-.1, x[-1]+.1]:
        interval = core.locate(knots, point)
        d1, d2 = basis_derivative_weights(scratch, knots, interval, point)
        local_q = q[interval-3:interval+1]
        actual = [d1 @ local_q, d2 @ local_q]
        expected = [reference(point, 1), reference(point, 2)]
        assert_allclose(actual, expected, rtol=1e-11, atol=1e-9)
        np.testing.assert_array_equal(scratch, 0.)  # Safe to reuse for the next point.
