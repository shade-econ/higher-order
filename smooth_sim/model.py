"""High-level household workflows built from the smooth simulation core."""

import numpy as np

from . import core, utils
from discrete_sim.sim_steady_state import discretize_assets, discretize_income, stationary_markov


class Household:
    """Smooth incomplete-markets household with a zero borrowing limit."""

    def __init__(self, Pi, a_grid, e, beta, eis, sigma_xi):
        self.Pi = Pi
        self.Pi_F = utils.get_Pi_F(Pi)
        self.pi = stationary_markov(Pi)
        self.a_grid = a_grid
        self.e = e
        self.e_mean = self.pi @ e
        self.beta = beta
        self.eis = eis
        self.sigma_xi = sigma_xi

    def backward_iteration(self, Va, r, Z, Tr):
        """Apply one backward step to marginal value Va at levels r, Z, and Tr."""
        return core.backward_iteration(
            Va, self.Pi, self.a_grid, self.e, self.beta, self.eis, self.sigma_xi,
            r, Z, Tr)

    def expectation_iteration(self, f, m_endog, r, Z, Tr, mult=None, constrained=True):
        """Take one backward expectation of f through the household policy."""
        return core.expectation_iteration(
            f, m_endog, self.Pi, self.a_grid, self.e, self.sigma_xi,
            r, Z, Tr, mult, constrained)

    def forward_iteration(self, F, m_endog, r, Z, Tr):
        """Advance conditional asset CDF F one period at levels r, Z, and Tr."""
        return core.forward_iteration(
            F, m_endog, self.Pi_F, self.a_grid, self.e, self.sigma_xi,
            r, Z, Tr)

    def policy_ss(self, r, Z, Tr, Va_forcing=None, tol=1E-9, maxit=10_000):
        """Solve the stationary policy with an optional (S,N) Va forcing.

        Va_forcing is added to next-period Va before every Markov expectation.
        """
        c_init = 0.05 * ((Z*self.e)[:, np.newaxis] + Tr + self.a_grid)
        Va = c_init**(-1/self.eis)

        for it in range(maxit):
            Va_next = Va if Va_forcing is None else Va+Va_forcing
            Va, (q, m_endog) = self.backward_iteration(Va_next, r, Z, Tr)
            if it > 0 and np.max(np.abs(m_endog - m_endog_old)) < tol:
                return Va, (q, m_endog)
            m_endog_old = m_endog

        raise ValueError(f"Policy failed to converge after {maxit} iterations")

    def distribution_ss(self, m_endog, r, Z, Tr, tol=1E-11, maxit=10_000):
        """Solve the stationary conditional asset CDF."""
        F = np.ones_like(m_endog)

        for it in range(maxit):
            Fnew = self.forward_iteration(F, m_endog, r, Z, Tr)
            if it > 0 and np.max(np.abs(Fnew - F)) < tol:
                return Fnew
            F = Fnew

        raise ValueError(f"Distribution failed to converge after {maxit} iterations")

    def aggregate_assets(self, F):
        """Aggregate a conditional asset CDF across persistent states."""
        return self.pi @ core.aggregate_assets_by_state(F, self.a_grid)

    def steady_state(self, r, Z, Tr, Va_forcing=None,
                     backward_tol=1E-9, forward_tol=1E-11):
        """Solve the stationary household problem with an optional Va forcing."""
        Va, (q, m_endog) = self.policy_ss(
            r, Z, Tr, Va_forcing=Va_forcing, tol=backward_tol)
        F = self.distribution_ss(m_endog, r, Z, Tr, tol=forward_tol)

        A = self.aggregate_assets(F)
        C = r*A + Z*self.e_mean + Tr
        mpc = core.get_mpc(q, m_endog)

        return dict(Va=Va, q_policy=q, m_endog=m_endog, F=F,
                    A=A, C=C, r=r, Z=Z, Tr=Tr, mpc=mpc)

    def backward_transition(self, ss, r, Z, Tr, Va_forcing=None, Va_terminal=None):
        """Solve policies backward along level paths r, Z, and Tr.

        Va_terminal is the marginal value after the final transition date.
        Returns date-zero Va and m_endog[t,s,k].
        """
        T = len(r)
        Va = ss["Va"].copy() if Va_terminal is None else Va_terminal.copy()
        m_endog = np.empty((T,) + ss["m_endog"].shape)

        for t in range(T-1, -1, -1):
            Va_next = Va if Va_forcing is None else Va+Va_forcing[t]
            Va, (_, m_endog[t]) = self.backward_iteration(
                Va_next, r[t], Z[t], Tr[t])
        return Va, m_endog

    def transition(
            self, ss, dr=None, dZ=None, dTr=None,
            Va_forcing=None, Va_terminal=None, F0=None,
            return_F=False):
        """Compute asset and consumption paths for deviations dr, dZ, dTr from ss.

        Va_forcing[t] is added to next period's marginal value in the backward step;
        Va_terminal sets its terminal value. F0 sets the initial conditional asset
        CDF, keeping state shares at self.pi. Both boundaries default to ss.

        Returns end-of-period A, current C, and date-zero Va0, all in levels.
        Optional F[t] is the beginning-of-period CDF; F[t+1] determines A[t].
        """
        paths = [path for path in (dr, dZ, dTr, Va_forcing) if path is not None]
        if not paths:
            raise ValueError("At least one input or Va-forcing path is required")

        T = len(paths[0])
        if any(len(path) != T for path in paths[1:]):
            raise ValueError("Transition paths must have the same length")

        r = np.full(T, ss["r"]) if dr is None else ss["r"] + dr
        Z = np.full(T, ss["Z"]) if dZ is None else ss["Z"] + dZ
        Tr = np.full(T, ss["Tr"]) if dTr is None else ss["Tr"] + dTr

        # Solve policies backward from the terminal marginal value.
        Va, m_endog = self.backward_transition(ss, r, Z, Tr, Va_forcing, Va_terminal)

        # Propagate the distribution and aggregate assets forward.
        F = ss["F"].copy() if F0 is None else F0.copy()
        A = np.empty(T+1)
        A[0] = ss["A"] if F0 is None else self.aggregate_assets(F)
        if return_F:
            F_path = np.empty((T+1,) + F.shape)
            F_path[0] = F

        for t in range(T):
            F = self.forward_iteration(F, m_endog[t], r[t], Z[t], Tr[t])
            A[t+1] = self.aggregate_assets(F)
            if return_F:
                F_path[t+1] = F

        C = Z*self.e_mean + (1+r)*A[:-1] + Tr - A[1:]
        result = dict(A=A[1:], C=C, Va0=Va)
        if return_F:
            result["F"] = F_path
        return result


def paper_household():
    """Return the paper-calibrated household and its steady state as (hh, ss)."""
    p = paper_calibration()
    hh = Household(p["Pi"], p["a_grid"], p["e"], p["beta"], p["eis"], p["sigma_xi"])
    ss = hh.steady_state(p["r"], p["Z"], p["Tr"],
                         backward_tol=1e-11, forward_tol=1e-12)
    return hh, ss


def paper_calibration():
    """Paper calibration targeting income-weighted MPC=0.2 and assets=100% of annual GDP."""
    # Match log-income s.d. 0.92 and quarterly-change s.d. 0.24;
    # the persistent component has annual persistence 0.91.
    rho_z = 0.91**0.25
    sig_z = np.sqrt((0.92**2-0.5*0.24**2)/rho_z)
    sigma_xi = np.sqrt(0.92**2-sig_z**2)
    z, _, Pi_z = discretize_income(rho_z, sig_z, 7)

    # Fitted discount factors target A=4 (quarterly GDP=1) and impact MPC=0.2,
    # measured by a two-sided dZ=1e-4 income difference.
    beta_types = np.array([0.9600600537357907, 0.9904871775270647])
    Pi_beta = np.array([[0.995, 0.005], [0.005, 0.995]])
    # Independent beta and income transitions, with income states within each beta type.
    Pi = np.kron(Pi_beta, Pi_z)
    e = np.tile(z, len(beta_types))

    # The column shape makes beta*Pi discount each current-state row.
    beta = np.repeat(beta_types, len(z))[:, np.newaxis]
    a_grid = discretize_assets(0, 1000, 200)

    r = 1.02**0.25-1  # Quarterly rate corresponding to 2% annually.
    Tr = 0.14
    Z = 1-r*4-0.17-Tr  # GDP less interest on bonds, government spending, and transfers.
    return dict(Pi=Pi, a_grid=a_grid, e=e, beta=beta, eis=1,
                sigma_xi=sigma_xi, r=r, Z=Z, Tr=Tr)
