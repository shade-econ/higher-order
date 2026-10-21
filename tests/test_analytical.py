"""Check analytical_simple and analytical_fast backward/forward derivatives.

Compare one-period steps and the transitions built by sequence_space.paths
against each other and finite differences of the nonlinear household model.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sequence_space import paths


def numerical_derivatives(f, scale=1):
    """Return the first three derivatives at zero using centered differences.

    Higher orders use larger steps to limit cancellation; scale lets callers
    check sensitivity to halving or doubling these steps.
    """
    h1, h2, h3 = scale*np.array([1e-4, 2e-4, 1e-3])
    first = (f(h1)-f(-h1))/(2*h1)
    second = (f(h2)-2*f(0.)+f(-h2))/h2**2
    third = (f(-3*h3)-8*f(-2*h3)+13*f(-h3)-13*f(h3)+8*f(2*h3)-f(3*h3))/(8*h3**3)
    return first, second, third


def test_analytical(household, simple_derivatives, fast_derivatives):
    """Differentiate one backward/forward step with all its inputs perturbed jointly.

    Inputs are aggregate inputs (r, Z, Tr), forward-looking marginal
    value Va_next, and the backward-looking distribution F_in inherited from
    the previous period. Compare output derivatives through third order across
    the simple and fast formulas and against nonlinear finite differences.
    """
    hh, ss = household
    simple_b, simple_f = simple_derivatives
    fast_b, fast_f = fast_derivatives

    # Give each input first, second, and third derivatives with respect to theta.
    # Thus the check includes both interactions and curvature of the inputs.
    u = np.array([ss['r'], ss['Z'], ss['Tr']])
    du = np.array([0.05, 1., -0.25])  # Joint direction in (r, Z, Tr).
    inputs = du, -0.2*du, 0.1*du

    # Perturb next period's marginal value, tapering the change at high assets.
    continuation = tuple(k*ss['Va']/(1+hh.a_grid) for k in (0.5, -0.1, 0.05))
    # Perturb the entering CDF, with changes vanishing where F=0 or F=1.
    inherited = tuple(k*ss['F']*(1-ss['F']) for k in (0.5, -0.1, 0.05))

    # The backward step gives current Va and the endogenous cash grid;
    # the forward step uses that grid and the entering CDF to give next period's F.
    Va, m_endog = simple_b.step_3rd(*continuation, *inputs)
    F = simple_f.step_3rd(*inherited, *m_endog, *inputs)
    fast_Va, fast_m_endog = fast_b.step_3rd(*continuation, *inputs)
    fast_F = fast_f.step_3rd(*inherited, *fast_m_endog, *inputs)

    # Compiled and simple formulas should agree almost to roundoff at each order.
    for actual, expected in zip((fast_Va, fast_m_endog, fast_F), (Va, m_endog, F)):
        for a, e in zip(actual, expected):
            assert_allclose(a, e, rtol=0, atol=2e-13*np.max(np.abs(e)))

    def nonlinear_step(theta):
        """Return nonlinear Va, cash, and F; finite differences in theta check their analytical derivatives."""
        # Realize the same joint perturbation in levels, for each input x:
        # x(theta)=x0+theta*dx+theta^2*d2x/2+theta^3*d3x/6.
        weights = theta, theta**2/2, theta**3/6
        Va_next = ss['Va'] + sum(w*d for w, d in zip(weights, continuation))
        F_in = ss['F'] + sum(w*d for w, d in zip(weights, inherited))
        r, Z, Tr = u + sum(w*d for w, d in zip(weights, inputs))
        Va_out, (_, cash) = hh.backward_iteration(Va_next, r, Z, Tr)
        F_out = hh.forward_iteration(F_in, cash, r, Z, Tr)
        return np.array([Va_out, cash, F_out])

    # (Mean, maximum) relative-error bounds, in first/second/third-order columns.
    # Cash is algebraic; Va and F also incur spline and quadrature error.
    tolerances = {
        'Va':   ((6e-5, 1e-3), (2.5e-3, 1.5e-2), (1.5e-2, 5e-2)),
        'cash': ((2e-8, 3e-8), (1e-5, 3e-5),     (1e-3, 2e-3)),
        'F':    ((4e-6, 2e-5), (7e-5, 1e-3),     (1.2e-3, 8e-3)),
    }
    analytical = {'Va': Va, 'cash': m_endog, 'F': F}
    for scale in (0.5, 1., 2.):
        # Group numerical derivatives by variable, in the same order as analytical.
        first, second, third = numerical_derivatives(nonlinear_step, scale)
        numerical = dict(zip(analytical, zip(first, second, third)))
        for name in analytical:
            for order, (exact, numeric) in enumerate(zip(analytical[name], numerical[name])):
                mean_tol, max_tol = tolerances[name][order]
                error = np.abs(exact-numeric)
                # Normalize by the mean/peak response; absolute errors cannot cancel.
                assert error.mean() < mean_tol*np.abs(exact).mean(), (name, order+1, scale)
                assert error.max() < max_tol*np.abs(exact).max(), (name, order+1, scale)


@pytest.mark.parametrize("shift_terminal", [False, True])
def test_paths(household, simple_derivatives, fast_derivatives, shift_terminal):
    """Check derivatives of an entire transition path rather than one backward/forward step.

    Compare aggregate asset responses to an income impulse across the simple and
    fast analytical implementations and against numerical differentiation.
    The shock arrives at date zero; initial F stays at steady state. Also check
    a changing terminal marginal value, including its second and third derivatives.
    """
    hh, ss = household
    du = np.zeros((20, 3))
    du[0, 1] = 1.  # Unit income impulse in partial equilibrium.

    terminal = tuple(k*ss['Va']/(1+hh.a_grid) for k in (0.5, -0.1, 0.05))

    def nonlinear_path(theta):
        Va_terminal = ss['Va']
        if shift_terminal:
            weights = theta, theta**2/2, theta**3/6
            Va_terminal = Va_terminal + sum(w*d for w, d in zip(weights, terminal))
        return hh.transition(ss, *(theta*du).T, Va_terminal=Va_terminal)['A']

    # Check sensitivity to halving and doubling the finite-difference steps.
    numerical = [numerical_derivatives(nonlinear_path, scale) for scale in (0.5, 1., 2.)]
    # The terminal perturbation has larger spatial approximation errors on this grid;
    # refinement reduces them. Keep the tighter bounds for the income impulse alone.
    tolerances = (3e-6, 2e-5, 1.5e-3) if shift_terminal else (3e-7, 2e-5, 7e-4)
    # Exercise each path routine, comparing aggregate assets at every returned order.
    for solve, keys in ((paths.first_order, ('dA',)),
                        (paths.second_order, ('dA', 'd2A')),
                        (paths.third_order, ('dA', 'd2A', 'd3A'))):
        boundary = None
        if shift_terminal:
            boundary = terminal[0] if len(keys) == 1 else terminal[:len(keys)]
        analytical = solve(*simple_derivatives, du, Va_terminal=boundary)
        compiled = solve(*fast_derivatives, du, Va_terminal=boundary)
        for order, key in enumerate(keys):
            # Scale by the path's peak, so near-zero tail responses remain comparable.
            peak = np.max(np.abs(analytical[key]))
            tight = (5e-13, 5e-11, 5e-11)[order]*peak
            assert_allclose(compiled[key], analytical[key], rtol=0, atol=tight)
            atol = tolerances[order]*peak
            for reference in numerical:
                assert_allclose(reference[order], analytical[key], rtol=0, atol=atol)
