"""Assemble perturbation solutions and provide simulation, GIRFs, and moments.

solve_pe and solve_ge bring together the household derivatives, size and history
effects, and anticipated-risk calculations. solve_ge also adjusts income to clear
the asset market. They return a Solution whose methods use these derivatives to
simulate paths and calculate GIRFs and moments, as in notebook 3.

X denotes consumption in PE and output in GE.
The size and history effects follow the paper's notation:
X1[j] = X_j, X2[j,l] = X_{j,j+l}, X3[j,l,lp] = X_{j,j+l,j+lp}.
Here j counts periods since the most recent shock, and l and lp are lags to
earlier shocks. X2_own stores the size effects X_{j,j} used in the ergodic mean.

X_ss is the deterministic steady state. Anticipated risk changes its level
through X_risk = X_{sigma,sigma} and shock transmission through
X_rss + X_tvp = X_{sigma,sigma,j}, the RSS and TVP effects.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from analytical_fast.backward import Backward
from analytical_fast.forward import Forward
from . import stochastic
from .jacobian import income_jacobian_analytical
from .histories import second_histories, third_histories
from .paths import first_order, second_order
from .risk_terms import stationary_response, rss_response, tvp_response


@dataclass
class Solution:
    """A perturbation solution for X through the requested order.

    The solution combines size, history, and risk effects to calculate simulated
    paths, GIRFs, and ergodic moments. With only_moments=True, the solvers calculate
    just the derivatives needed for moments, omitting history effects needed for
    simulation and GIRFs at second and third order.
    """
    X_ss: float
    X1: np.ndarray
    T: int
    L: int
    order: int
    only_moments: bool
    X2_own: np.ndarray | None = None
    X2: np.ndarray | None = None
    X3: np.ndarray | None = None
    X_rss: np.ndarray | None = None
    X_tvp: np.ndarray | None = None
    X_risk: float = 0.

    def simulate(self, epsbar, burnin=0):
        """Return x in X_t = X_ss + sigma*x[0,t] + sigma^2*x[1,t] + ...

        epsbar contains standardized innovations. Output rows give coefficients
        through the solved order, with the first burnin dates removed.
        """
        if self.only_moments and self.order > 1:
            raise ValueError("Simulation requires all size and history effects; use only_moments=False")
        X_sigma_sigma = self.X_rss+self.X_tvp if self.order == 3 else None
        return stochastic.simulate(epsbar, self.X1, self.X2, self.X3,
                                   self.X_risk, X_sigma_sigma, burnin)

    def girf(self, eps, nu, sigma, horizon=20):
        """Return the GIRF after today's innovation nu, given past innovations eps.

        eps runs from oldest to newest and is in levels (eps=sigma*epsbar).
        Output rows give successive approximations, each including lower orders.
        Output columns are horizons 0,...,horizon-1.
        """
        if self.only_moments and self.order > 1:
            raise ValueError("GIRFs require all size and history effects; use only_moments=False")
        X_sigma_sigma = self.X_rss+self.X_tvp if self.order == 3 else None
        return stochastic.girf(eps, nu, sigma, self.X1, self.X2, self.X3,
                               X_sigma_sigma=X_sigma_sigma, horizon=horizon)

    def mean_correction(self):
        """Return the coefficient of sigma^2 in the ergodic mean.

        At order=1, return zero because the correction was not computed.
        """
        return stochastic.mean_correction(self.X2_own, self.X_risk)

    def covariance_coefficients(self, lags=(0, 1)):
        """Return coefficients of sigma^2 and sigma^4 in the requested lagged covariances.

        The sigma^4 coefficient requires the cubic terms from an order=3 solution.
        Lower-order solutions return zero for this correction, rather than the
        incomplete correction obtained from the quadratic terms alone.
        """
        X_sigma_sigma = self.X_rss+self.X_tvp if self.order == 3 else None
        return stochastic.covariance_coefficients(
            self.X1, self.X2, self.X3, X_sigma_sigma=X_sigma_sigma, lags=lags)

    def ergodic_mean(self, sigma=0.05):
        """Return E[X] through sigma^2, or just X_ss when order=1."""
        return self.X_ss + sigma**2*self.mean_correction()

    def covariances(self, sigma=0.05, lags=(0, 1)):
        """Return Cov(X_t,X_{t-h}) through sigma^4 for order=3, otherwise through sigma^2."""
        Sigma2, Sigma4 = self.covariance_coefficients(lags)
        return sigma**2*Sigma2 + sigma**4*Sigma4


def solve_ge(hh, ss, T=300, L=100, rho_B=0.95, order=3,
              only_moments=False, workers=1):
    """Return an IKC output solution through the requested order for debt innovations.

    Debt follows rho_B**t after an innovation. Income adjusts to clear the asset
    market, with steady-state output normalized to one and government spending fixed.

    T is the number of response dates after the newest shock. L limits the lags
    to earlier shocks to 0,...,L-1. Set only_moments=True to calculate just the
    mean at order=2, or the mean and corrected covariances at order=3.
    """
    if order not in (1, 2, 3) or not 1 <= L <= T:
        raise ValueError("Use order 1, 2, or 3 and 1 <= L <= T")
    backward, forward = Backward(hh, ss), Forward(hh, ss)
    # Factor J[t,s] = dA[t]/dZ[s] once to solve for income at each order.
    J_lu = lu_factor(income_jacobian_analytical(backward, forward, T))

    # Solve J @ Z1 = B1 to clear the asset market at first order.
    # Output follows from the government's budget: Y1 = Z1 + (1+r)*B1_lag - B1.
    debt = rho_B**np.arange(T)
    u = np.zeros((T, 3))
    u[:, 1] = lu_solve(J_lu, debt)
    Y1 = u[:, 1]+(1+ss['r'])*np.concatenate(([0.], debt[:-1]))-debt
    result = Solution(T=T, L=L, order=order, only_moments=only_moments,
                  X_ss=1., X1=Y1)
    if order == 1:
        return result

    # The second-order mean needs size effects but no history effects.
    if only_moments and order == 2:
        own = second_order(backward, forward, u)
        Y2_own = lu_solve(J_lu, -own['d2A'])
    else:
        Y2 = second_histories(backward, forward, u, L, J_lu=J_lu)
        result.X2 = Y2
        Y2_own = Y2[:, 0]
    result.X2_own = Y2_own

    # Debt has no higher derivatives, so higher-order income and output responses coincide.
    u2 = np.zeros_like(u)
    u2[:, 1] = Y2_own
    equilibrium2 = second_order(backward, forward, u, u2)
    # Va00 measures curvature in the debt innovation along the equilibrium income
    # path. Its expectation effect determines the risky steady state below.
    Va00 = equilibrium2['d2Va0']

    # Measure how assets respond to a permanent income increase and to anticipated
    # risk. Choose the income change that offsets the risk-induced asset demand.
    income_ss = stationary_response(backward, forward, np.array([0., 1., 0.]),
                                     np.zeros(backward.shape))
    direct_ss = stationary_response(backward, forward, np.zeros(3), Va00)
    Y_risk = -direct_ss['dA']/income_ss['dA']
    result.X_risk = Y_risk
    if order == 2:
        return result

    # Third-order size and history effects use the second-order income responses.
    Y3, Va_xyy = third_histories(backward, forward, u, L, d2Z=Y2, J_lu=J_lu,
                                 workers=workers, only_moments=only_moments)
    result.X3 = Y3

    # RSS: how starting from the risky steady state changes the shock response.
    rss_du = np.array([0., Y_risk, 0.])
    stationary = stationary_response(backward, forward, rss_du, Va00)
    rss = rss_response(backward, forward, u, stationary, Va00, rss_du=rss_du)
    # Adjust income to offset the additional asset response and clear the market.
    result.X_rss = lu_solve(J_lu, -rss['dA'])

    # TVP: how the shock changes the precautionary response to future risk.
    tvp = tvp_response(backward, forward, Va_xyy, T)
    result.X_tvp = lu_solve(J_lu, -tvp['dA'])
    return result


def solve_pe(hh, ss, T=300, L=100, rho_Z=0., order=3,
             only_moments=False, workers=1):
    """Solve consumption for aggregate income innovations at fixed r and transfers.

    Income follows rho_Z**t after an innovation. The solution format and options
    match solve_ge, with consumption in place of output.
    """
    if order not in (1, 2, 3) or not 1 <= L <= T:
        raise ValueError("Use order 1, 2, or 3 and 1 <= L <= T")
    backward, forward = Backward(hh, ss), Forward(hh, ss)

    # With income given, consumption follows from the household budget.
    u = np.zeros((T, 3))
    u[:, 1] = rho_Z**np.arange(T)
    first = first_order(backward, forward, u)
    C1 = u[:, 1]*hh.e_mean+consumption_from_assets(first['dA'], ss['r'])
    result = Solution(T=T, L=L, order=order, only_moments=only_moments,
                  X_ss=ss['C'], X1=C1)
    if order == 1:
        return result

    # Income is linear in the shock, so its higher derivatives are zero.
    own = second_order(backward, forward, u)
    # The second-order mean needs size effects but no history effects.
    if only_moments and order == 2:
        A2_own = own['d2A']
    else:
        A2 = second_histories(backward, forward, u, L)
        result.X2 = consumption_from_assets(A2, ss['r'])
        A2_own = A2[:, 0]
    result.X2_own = consumption_from_assets(A2_own, ss['r'])

    # Curvature of marginal value in the income innovation determines the effect
    # of anticipated risk on the steady state.
    Va00 = own['d2Va0']

    # Anticipated risk raises steady-state assets and hence interest income and consumption.
    stationary = stationary_response(backward, forward, np.zeros(3), Va00)
    result.X_risk = ss['r']*stationary['dA']
    if order == 2:
        return result

    # Calculate third-order size and history effects without GE income corrections.
    A3, Va_xyy = third_histories(backward, forward, u, L,
                                 workers=workers, only_moments=only_moments)
    result.X3 = consumption_from_assets(A3, ss['r'])

    # RSS: how starting from the risky steady state changes the shock response.
    rss = rss_response(backward, forward, u, stationary, Va00)
    result.X_rss = consumption_from_assets(rss['dA'], ss['r'])

    # TVP: how the shock changes the precautionary response to future risk.
    tvp = tvp_response(backward, forward, Va_xyy, T)
    result.X_tvp = consumption_from_assets(tvp['dA'], ss['r'])
    return result


def consumption_from_assets(dA, r):
    """Consumption response from asset accumulation: (1+r)*dA[t-1] - dA[t].

    Axis zero is the response date. Any remaining axes index lags between shock dates.
    Add the income response separately at first order.
    """
    dC = -dA.copy()
    dC[1:] += (1+r)*dA[:-1]
    return dC
