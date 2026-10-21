"""Primitive backward, expectation, and forward operators for smooth simulation."""

import numpy as np
from numba import njit
from scipy import interpolate

from spline import core as spline
from . import utils


"""Backward iteration"""

@njit
def backward_iteration(Va, Pi, a_grid, e, beta, eis, sigma_xi, r, Z, Tr):
    """Compute today's marginal value and consumption policy from tomorrow's Va.
    Cash on hand is m = (1+r)a + Z e_s xi + Tr, with log(xi) ~ N(-sigma_xi**2/2, sigma_xi**2).

    Return Va and (q, m_endog), the consumption spline coefficients and cash grid.
    Saving at m_endog[s,j] is a_grid[j].
    """
    # Discount and average over the next persistent state.
    Wa = beta * Pi @ Va

    # Fit consumption on the endogenous cash-on-hand grid; below m_endog[s, 0], c=m.
    c_endog = Wa**(-eis)
    m_endog = c_endog + a_grid

    q = np.empty_like(Va)
    for s in range(len(e)):
        q[s] = spline.interp(m_endog[s], c_endog[s])

    # Integrate over lognormal income to obtain marginal value on the asset grid.
    b = (1+r)*a_grid + Tr
    mu = -sigma_xi**2/2 + np.log(Z*e)

    Va = np.empty_like(Va)
    for s in range(len(e)):
        Va[s] = expectation_lognormal_margutil(b, m_endog[s], q[s], mu[s], sigma_xi, eis)

    # Convert marginal value of cash to marginal value of assets.
    Va *= (1+r)
    return Va, (q, m_endog)


@njit
def expectation_lognormal_margutil(b, m_endog, q, mu, sigma_xi, eis):
    """Return E[c(b+income)**(-1/eis)], with log(income) ~ N(mu, sigma_xi**2)."""
    EVa = np.zeros_like(b)

    # Separate asset points where the constraint can bind within the integration
    # range (log income within mu +/- 8*sigma_xi) from those where it cannot.
    min_income = np.exp(mu - 8*sigma_xi)
    first_unconstrained = np.searchsorted(b, m_endog[0] - min_income)
    for i in range(first_unconstrained):
        # Split at m_endog[0]: below it c=m; above it consumption follows the spline.
        w, m = utils.integrate_lognormal_interval(b[i], mu, sigma_xi, 0, m_endog[0])
        EVa[i] = w @ m**(-1/eis)

        w, m = utils.integrate_lognormal_interval(b[i], mu, sigma_xi, m_endog[0], np.inf)
        c = spline.val_monotonic(q, m_endog, m)
        EVa[i] += w @ c**(-1/eis)

    # For the remaining asset points, the constraint never binds over this income
    # range, so all integrals use the same income nodes.
    # At each income node, m increases with assets, allowing a fast, monotonic spline pass.
    # Gauss-Hermite would be more efficient here, but switching quadrature rules
    # introduces small mismatches that numerical differentiation then amplifies.
    if first_unconstrained < len(b):
        w, income = utils.integrate_lognormal_interval(0.0, mu, sigma_xi, 0.0, np.inf)
        for j in range(len(income)):
            m = b[first_unconstrained:] + income[j]
            c = spline.val_monotonic(q, m_endog, m)
            EVa[first_unconstrained:] += w[j] * c**(-1/eis)

    return EVa


"""General expectation iteration (mostly parallels backward iteration)"""

@njit
def expectation_iteration(f, m_endog, Pi, a_grid, e, sigma_xi, r, Z, Tr,
                          mult=None, constrained=True):
    """Compute today's conditional expectation of tomorrow's f under today's policy.
    Use m_endog and the same cash-on-hand process as backward_iteration.

    Optional mult multiplies the expected f at the cash-on-hand stage.
    Below m_endog[s, 0], use the first gridpoint's value if constrained, else zero.
    With mult=mps (the marginal propensity to save) and constrained=False,
    this propagates marginal expectations.
    """
    # Average over the next persistent state and apply the multiplier.
    f = Pi @ f
    if mult is not None:
        f *= mult

    # At m_endog[s,k], today's policy saves a_grid[k], so fit the expected f
    # at those asset points as a function of cash on hand m.
    q = np.empty_like(f)
    for s in range(len(e)):
        q[s] = spline.interp(m_endog[s], f[s])

    # Integrate over lognormal income to obtain expectations on the asset grid.
    b = (1+r)*a_grid + Tr
    mu = -sigma_xi**2/2 + np.log(Z*e)

    Ef = np.empty_like(f)
    for s in range(len(e)):
        Ef[s] = expectation_lognormal_m(b, m_endog[s], q[s], mu[s], sigma_xi, constrained)

    return Ef


@njit
def expectation_lognormal_m(b, m_endog, q, mu, sigma_xi, constrained=False):
    """Integrate spline q at b+income, with log(income) ~ N(mu, sigma_xi**2).

    Below m_endog[0], use q[0] if constrained=True, otherwise zero.
    """
    Ef = np.zeros_like(b)

    # As above, separate asset points where the constraint can bind from those
    # where it cannot within the integration range.
    min_income = np.exp(mu - 8*sigma_xi)
    first_unconstrained = np.searchsorted(b, m_endog[0] - min_income)
    for i in range(first_unconstrained):
        # Split at m_endog[0]: spline above; constant q[0] or zero below.
        w, m = utils.integrate_lognormal_interval(b[i], mu, sigma_xi, m_endog[0], np.inf)
        Ef[i] = w @ spline.val_monotonic(q, m_endog, m)
        if constrained:
            # When constrained, saving is a_grid[0], so use its expected value.
            # This equals the spline value at m_endog[0], namely q[0].
            Ef[i] += (1 - w.sum()) * q[0]

    # For the remaining asset points, share income nodes and a monotonic spline pass.
    if first_unconstrained < len(b):
        w, income = utils.integrate_lognormal_interval(0.0, mu, sigma_xi, 0.0, np.inf)
        for j in range(len(income)):
            m = b[first_unconstrained:] + income[j]
            Ef[first_unconstrained:] += w[j] * spline.val_monotonic(q, m_endog, m)

    return Ef


"""Forward iteration"""

def forward_iteration(F, m_endog, Pi_F, a_grid, e, sigma_xi, r, Z, Tr):
    """Iterate the conditional asset CDF forward once."""
    # Apply saving within each current state, then mix conditional CDFs using
    # Pi_F[s_next,s] = Pr(s today | s_next tomorrow).
    return Pi_F @ forward_policy(F, m_endog, a_grid, e, sigma_xi, r, Z, Tr)


@njit
def forward_policy(F, m_endog, a_grid, e, sigma_xi, r, Z, Tr):
    """Advance the asset CDF through income and saving within each persistent state."""
    b = (1+r)*a_grid + Tr
    mu = -sigma_xi**2/2 + np.log(Z*e)

    Fnew = np.empty_like(F)
    for s in range(len(e)):
        # Index the inherited CDF by b to include asset returns and transfers.
        qF = spline.interp(b, F[s])
        # Saving <= a_grid[i] iff cash on hand m <= m_endog[s,i].
        Fnew[s] = forward_lognormal_cdf(qF, b, m_endog[s], mu[s], sigma_xi)

    return Fnew


@njit
def forward_lognormal_cdf(qF, b, m_endog, mu, sigma_xi):
    """Return E[F(m_endog - income)], where log(income) ~ N(mu, sigma_xi**2).

    qF represents the CDF on b; extend F as zero below b[0] and one above b[-1].
    The result is the cash-on-hand CDF evaluated at m_endog.
    """
    Fnew = np.zeros_like(m_endog)
    log_min, log_max = mu - 8*sigma_xi, mu + 8*sigma_xi
    income_min = np.exp(log_min)

    # Before lo, cash thresholds are unreachable, so Fnew=0. From hi onward,
    # some queries exceed b[-1]; mid:hi keeps all queries inside the CDF grid.
    lo = np.searchsorted(m_endog, b[0] + income_min)
    hi = np.searchsorted(m_endog, b[-1] + income_min, side="right")
    mid = min(hi, np.searchsorted(m_endog, b[0] + np.exp(log_max)))

    # lo:mid: integrate up to income=m-b[0], where the CDF drops to zero.
    # Each w,x below is a vector across m, with a separate income interval for each m.
    # Batch by quadrature-node index so spline queries m-exp(x) are increasing.
    m = m_endog[lo:mid]
    weights, log_income = utils.integrate_normal_common_lower(
        mu, sigma_xi, log_min, np.log(m-b[0]))
    for w, x in zip(weights, log_income):
        Fnew[lo:mid] += w * spline.val_monotonic(qF, b, m-np.exp(x))

    # mid:hi: share one rule over the integration range; w,x are now scalars.
    m = m_endog[mid:hi]
    weights, log_income = utils.integrate_normal_interval(mu, sigma_xi, log_min, log_max)
    for w, x in zip(weights, log_income):
        Fnew[mid:hi] += w * spline.val_monotonic(qF, b, m-np.exp(x))

    # From hi onward, split at income=m-b[-1]: F=1 below it, spline above it.
    # Enforce F=1 beyond the grid: spline extrapolation can destabilize repeated iterations.
    for i in range(hi, len(m_endog)):
        m_i = m_endog[i]
        log_lower = np.log(m_i-b[-1])
        log_upper = np.log(m_i-b[0])
        Fnew[i] = (utils.normal_cdf(min(log_lower, log_max), mu, sigma_xi)
                   - utils.normal_cdf(log_min, mu, sigma_xi))
        w, x = utils.integrate_normal_interval(mu, sigma_xi, log_lower, log_upper)
        Fnew[i] += w @ spline.val(qF, b, m_i-np.exp(x))

    return Fnew


def aggregate_assets_by_state(F, a_grid):
    """Integrate 1-F over a_grid to obtain assets conditional on each state."""
    return interpolate.CubicSpline(a_grid, 1-F, axis=1).integrate(0, a_grid[-1])


def get_mpc(q_policy, m_endog):
    """Evaluate dc/dm on the endogenous cash grid for each state.

    At the first gridpoint, use the derivative from the unconstrained side.
    """
    mpc = np.empty_like(q_policy)
    for i in range(len(q_policy)):
        mpc[i] = spline.val_der(q_policy[i], m_endog[i], m_endog[i])
    return mpc
