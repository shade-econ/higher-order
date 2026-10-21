"""Tests for expectations.py in analytical_simple and analytical_fast.

First check the simple routines against direct spline evaluation and numerical
integration with SciPy. Then check that the fast routines give the same answers
as the simple ones.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.stats import lognorm
from scipy.sparse import csr_matrix

from analytical_simple import expectations as simple
from analytical_fast import expectations as fast
from spline import core as spline
from smooth_sim import quadrature


# Simple expectation formulas: direct spline sums and independent integrals.

def test_weighted_spline_rows():
    """Check the conversion from quadrature nodes and weights to spline weights.

    We make up nodes and weights here: the conversion should work for any
    weighted sum, whether or not it comes from an actual quadrature rule.
    """
    # Irregular grid and evaluation points, including both extrapolation regions.
    grid = np.linspace(0, 2, 12)**1.3
    y = np.sin(grid)
    q = spline.interp(grid, y)
    f = CubicSpline(grid, y)
    x = np.array([2.5, -0.1, 0.3, 1.2, 0.8])
    w = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
    z = 1+x**2

    knots = spline.make_knots(grid)
    rows = simple.quadrature_to_spline_weights(knots, x, w, z)
    expected = [w @ f(x), w @ f(x, 1), w @ (z*f(x, 1)),
                w @ f(x, 2), w @ (z*f(x, 2)), w @ (z**2*f(x, 2))]
    assert_allclose(rows @ q, expected, rtol=2e-12, atol=2e-12)


def test_marginal_value_moments():
    """Check expected marginal-utility derivatives for a simple consumption rule.

    Consumption is linear on each side of the borrowing cutoff, so we can
    write the derivatives of g(m)=c(m)**(-gamma) explicitly and integrate
    them over income shocks with SciPy.
    """
    # Choose c=m below m=1 and c=1+0.4*(m-1) above it; fit the latter with a spline.
    grid = np.linspace(1, 5, 12)
    q = spline.interp(grid, 1+0.4*(grid-1))[None, :]

    # Cash is m=b+Z*eta, where log(eta) ~ N(0, sigma^2).
    b, Z, sigma, gamma = 0.3, 0.8, 0.3, 2.0
    shock = lognorm(s=sigma)
    moments = simple.marginal_value_moments(
        q, grid[None, :], np.array([b]), Z, np.array([0.0]), sigma, gamma)

    # For derivative orders n=1,2,3, the routine returns E[eta**k * g^(n)(m)],
    # k=0,...,n. Powers of eta arise from differentiating in Z, since dm/dZ=eta.
    cutoff = (grid[0]-b)/Z  # Income draw at which m=1.
    for order, actual in enumerate(moments, start=1):
        # Since c is linear in each region, d^order g/dm^order is
        # coefficient * c**(-gamma-order) * (dc/dm)**order.
        coefficient = (-1)**order*np.prod(gamma+np.arange(order))
        for power in range(order+1):
            def integrand(eta):
                m = b+Z*eta
                c, slope = (m, 1.0) if m < 1 else (1+0.4*(m-1), 0.4)
                return eta**power*coefficient*c**(-gamma-order)*slope**order*shock.pdf(eta)

            # Integrate the constrained and unconstrained formulas separately.
            expected = (quad(integrand, 0, cutoff, epsabs=1e-11)[0]
                        + quad(integrand, cutoff, np.inf, epsabs=1e-11)[0])
            assert_allclose(actual[power, 0, 0], expected, rtol=2e-10, atol=2e-11)


def test_asset_lognormal_expectation():
    """Integrate f(2-eps), where f(x)=x^2, over 0 < eps < 2.

    log(eps) ~ N(0, 0.3^2). This is an unnormalized integral over that interval.
    """
    def f(x):
        return x**2

    shock = lognorm(s=0.3)
    expected, _ = quad(lambda eps: f(2-eps)*shock.pdf(eps), 0, 2)

    # On this shock interval, x=2-eps lies in [0,2]. Represent f on that grid.
    grid = np.linspace(0, 2, 12)
    q = spline.interp(grid, f(grid))

    # The routine takes a batch of splines; here there is just one, at a_shift=2.
    matrices = simple.asset_expectation_matrices(
        grid, a_shift=np.array([[2.0]]),
        lam=np.array([1.0]), mu=np.array([0.0]), sigma=0.3)
    M = matrices[0, 0]  # The value-expectation matrix for the only spline.
    actual = (M @ q)[0]
    assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_asset_density_moments():
    """Check integrals of a**k * f''(a) * p'(eps) against SciPy integration.

    Here a=2-eps, f(a)=a^2, and p is the lognormal shock density.
    """
    # f(a)=a^2 gives f_aa=2. Include an unreachable query at a_shift=-1.
    grid = np.linspace(0, 2, 12)
    q = spline.interp(grid, grid**2)[None, :]
    sigma = 0.3
    shock = lognorm(s=sigma)
    actual = simple.asset_density_moments(
        grid, q, np.array([[-1., 2.]]), np.array([1.]), np.array([0.]), sigma)

    # At the first query, a=-1-eps is always below the asset grid.
    assert_allclose(actual[:, 0, 0], 0, atol=0)
    for power in range(4):
        def integrand(eps):
            a = 2-eps
            dlogp = -(1+np.log(eps)/sigma**2)/eps  # p'(eps)/p(eps)
            return a**power*2*dlogp*shock.pdf(eps)

        expected = quad(integrand, 0, 2, epsabs=1e-11)[0]
        assert_allclose(actual[power, 0, 1], expected, rtol=1e-9, atol=1e-11)


def test_asset_matrices_and_density_moments_across_spline_pieces():
    """Compare spline integrals with SciPy when curvature varies across knots,
    and check that doubling the quadrature nodes changes little.
    """
    # The spline's second derivative has different slopes between knots.
    grid = np.linspace(0, 2, 12)**1.3
    values = np.sin(3*grid)
    q = spline.interp(grid, values)
    f = CubicSpline(grid, values)
    # Integrate at a=a_shift-lam*eps, keeping only a inside the asset grid.
    # Use two shock distributions and shifts below, inside, and above the grid.
    a_shift = np.array([[-0.2, 0.8, 3.5], [-0.2, 1.1, 4.0]])
    lam, mu, sigma = np.array([0.7, 1.3]), np.array([-0.1, 0.2]), 0.5
    # Doubling nodes per piece checks quadrature convergence for both routines.
    matrices = simple.asset_expectation_matrices(grid, a_shift, lam, mu, sigma)
    refined = simple.asset_expectation_matrices(grid, a_shift, lam, mu, sigma, quadrature.start(80))
    assert_allclose(matrices, refined, rtol=1e-10, atol=1e-11)
    coefficients = np.tile(q, (len(mu), 1))
    moments = simple.asset_density_moments(grid, coefficients, a_shift, lam, mu, sigma)
    refined_moments = simple.asset_density_moments(
        grid, coefficients, a_shift, lam, mu, sigma, quadrature.start(80))
    assert_allclose(moments, refined_moments, rtol=1e-10, atol=1e-11)

    # (Derivative order, asset power) in ExpectationMaps order.
    orders_and_powers = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
    for s in range(len(mu)):
        shock = lognorm(s=sigma, scale=np.exp(mu[s]))
        for j, query in enumerate(a_shift[s]):
            for row, (order, power) in enumerate(orders_and_powers):
                # Integrate in a rather than log(eps): the density is p(eps)/lam.
                # Split at grid points and restrict to a < query so eps stays positive.
                def integrand(a):
                    eps = (query-a)/lam[s]
                    return a**power*f(a, order)*shock.pdf(eps)/lam[s]

                expected = sum(quad(integrand, lower, min(upper, query), epsabs=1e-11)[0]
                               for lower, upper in zip(grid[:-1], grid[1:]) if lower < query)
                assert_allclose(matrices[row, s, j] @ q, expected, rtol=1e-9, atol=1e-10)

            # The additional moments replace p by p', keeping the same change of variable.
            for power in range(4):
                def density_integrand(a):
                    eps = (query-a)/lam[s]
                    dlogp = -(1+(np.log(eps)-mu[s])/sigma**2)/eps
                    return a**power*f(a, 2)*dlogp*shock.pdf(eps)/lam[s]

                expected = sum(quad(density_integrand, lower, min(upper, query), epsabs=1e-11)[0]
                               for lower, upper in zip(grid[:-1], grid[1:]) if lower < query)
                assert_allclose(moments[power, s, j], expected, rtol=1e-9, atol=1e-10)


def test_cash_expectation_matrices():
    """Compare matrix-based expectations of m^2 and its derivatives with SciPy integration."""
    # Integrate f(m)=m^2 only above the cash cutoff, including spline extrapolation.
    grid = np.linspace(1, 5, 12)
    q = spline.interp(grid, grid**2)
    b, Z, sigma = 0.3, 0.8, 0.3
    shock = lognorm(s=sigma)
    matrices = simple.cash_expectation_matrices(
        grid[None, :], np.array([b]), Z, np.array([0.0]), sigma)
    actual = matrices[:, 0, 0] @ q

    # The six rows integrate f, f', eta*f', f'', eta*f'', eta^2*f''.
    orders_and_powers = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
    for row, (order, power) in enumerate(orders_and_powers):
        def integrand(eta):
            m = b+Z*eta
            derivative = m**2 if order == 0 else 2*m if order == 1 else 2
            return eta**power*derivative*shock.pdf(eta)

        expected = quad(integrand, (grid[0]-b)/Z, np.inf, epsabs=1e-11)[0]
        assert_allclose(actual[row], expected, rtol=1e-9, atol=1e-11)


# Fast expectation products: shared sparse storage versus the simple matrices.

def test_shared_storage():
    """Check that the fast storage recovers each original matrix, including
    entries that are nonzero in only one matrix.
    """
    matrices = np.zeros((6, 2, 3, 4))  # Six matrices, each with two 3-by-4 blocks.
    matrices[0, 0, 0, 1] = 2
    matrices[1, 0, 0, 2] = 3
    matrices[5, 1, 2, 3] = 4
    shared = fast.sparse_expectation_maps(matrices)

    # Each column of weights recovers one of the original sparse matrices.
    for k, original in enumerate(simple.sparse_expectation_maps(matrices)):
        recovered = csr_matrix((shared.weights[:, k], shared.indices, shared.indptr),
                               shape=original.shape)
        assert_array_equal(recovered.toarray(), original.toarray())


@pytest.mark.parametrize('kind', ['cash', 'asset'])
def test_expectation_products(kind):
    """Check that the fast routine gives the same expectations as separate
    matrix products for the first three spline perturbations.
    """
    # Two distributions, three queries each, and twelve coefficients per spline.
    grid = np.linspace(0, 4, 12)
    mu, sigma = np.array([-0.1, 0.2]), 0.3
    if kind == 'cash':
        grids = np.stack((grid+0.2, grid*1.1+0.3))
        matrices = simple.cash_expectation_matrices(grids, np.array([0., 0.5, 2.]), 0.8, mu, sigma)
    else:
        a_shift = np.array([[-1., 1., 3.], [-1., 2., 4.]])
        matrices = simple.asset_expectation_matrices(grid, a_shift, np.array([0.8, 1.2]), mu, sigma)
    maps = simple.sparse_expectation_maps(matrices)
    shared = fast.sparse_expectation_maps(matrices)
    # Arbitrary coefficients stand in for the three perturbations: we are
    # checking the matrix products, so no model derivatives need to be computed.
    q1, q2, q3 = np.random.default_rng(9).normal(size=(3, 2, 12))
    c1, c2, c3 = q1.ravel(), q2.ravel(), q3.ravel()

    # Second order needs two value expectations and two weighted derivatives.
    expected = np.array([maps.value @ c1, maps.value @ c2,
                         maps.first @ c1, maps.z_first @ c1]).reshape(4, 2, 3)
    actual = fast.perturbation_expectations(shared, q1, q2)
    assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    # Third order adds d3g, derivatives of d2g, and second derivatives of dg.
    expected = np.array([maps.value @ c1, maps.value @ c2, maps.value @ c3,
                         maps.first @ c1, maps.z_first @ c1,
                         maps.first @ c2, maps.z_first @ c2,
                         maps.second @ c1, maps.z_second @ c1, maps.z2_second @ c1]).reshape(10, 2, 3)
    actual = fast.perturbation_expectations(shared, q1, q2, q3)
    assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
