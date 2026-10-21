"""Same equations and interface as analytical_simple.backward, with faster evaluation.

Expectation products share sparse indices and run in one compiled loop.
Some key operations are extracted into compiled Numba routines, and repeated
squares and cubes are computed once and reused. See analytical_simple/backward.py
and analytical_simple/backward.md for the derivation.
"""

import numpy as np
from numba import njit, vectorize

from spline.fixed_grid import Spline
from .expectations import (cash_expectation_matrices, marginal_value_moments,
                           sparse_expectation_maps, perturbation_expectations,
                           lognormal_density_derivatives)


class Backward:
    """Differentiate one backward household step around a fixed steady state."""

    def step_1st(self, dVa_next, du):
        """Return first derivatives of marginal value and cash using prepared linear maps.

        Return current dVa and dm_endog. An optional final axis evaluates several
        directions together, as needed for news about each household input.
        """
        single = dVa_next.ndim == 2
        if single:
            dVa_next, du = dVa_next[..., None], du[:, None]
        S, N, Q = dVa_next.shape
        dWa = (self.beta_Pi @ dVa_next.reshape(S, N*Q)).reshape(S, N, Q)
        dVa = self.dVa_dWa @ dWa+self.dVa_du @ du
        dm_endog = self.dm_dWa[..., None]*dWa
        return (dVa[..., 0], dm_endog[..., 0]) if single else (dVa, dm_endog)

    def step_2nd(self, dVa_next, d2Va_next, du, d2u):
        """Return first and second derivatives of marginal value and the cash grid.

        dVa_next and d2Va_next perturb next period's marginal value on the
        asset grid. du and d2u perturb current inputs, ordered as (r,Z,Tr).
        Return (dVa,d2Va) and (dm_endog,d2m_endog). Each array has shape (state,asset).
        """
        # Continuation value and the endogenous grid
        # Differentiate m_endog=a_grid+Wa**(-eis).
        dWa = self.beta_Pi @ dVa_next
        d2Wa = self.beta_Pi @ d2Va_next
        dm_endog = self.dm_dWa*dWa
        d2m_endog = self.dm_dWa*d2Wa+self.d2m_dWa2*dWa**2

        # Marginal value at fixed cash
        # At steady state, saving at cash m_endog[s,j] is a_grid[j], so
        # asset and cash responses line up pointwise. The spline grids stay
        # fixed; changes in m_endog are returned separately.
        # Differentiate u_c(m-a)=Wa(a) at fixed m.
        # mps is the marginal propensity to save, da/dm.
        mps = self.mps
        da = -mps*dWa/self.u_cc
        da_m = self.m_spline.derivative(da)
        dWaa = self.a_spline.derivative(dWa)
        dv_m = mps*dWa
        d2v_m = mps*d2Wa+da_m*dWa+mps*da*dWaa

        # Integrate while current cash changes
        # First integrate the responses with the borrowing cutoff fixed.
        q1 = self.m_spline.coefficients(dv_m)
        q2 = self.m_spline.coefficients(d2v_m)
        E_dv_m, E_d2v_m, E_dv_mm, E_eta_dv_mm = perturbation_expectations(self.maps, q1, q2)

        # Apply the chain rule with dm=db+eta*dZ. The cached V partials
        # include constrained and unconstrained households, with the cutoff fixed.
        dr, dZ, dTr = du
        d2r, d2Z, d2Tr = d2u
        a = self.a_grid[None, :]
        db, d2b = dr*a+dTr, d2r*a+d2Tr
        Vbb, VbZ = self.barV_grad
        Vbbb, VbbZ, VbZZ = self.barV_hess_fixed
        dbarV_b = E_dv_m+Vbb*db+VbZ*dZ
        d2barV_b = (E_d2v_m+2*db*E_dv_mm+2*dZ*E_eta_dv_mm
                    + Vbbb*db**2+2*VbbZ*db*dZ+VbZZ*dZ**2+Vbb*d2b+VbZ*d2Z)

        # Borrowing-limit correction
        # Delta_v_mm is the unconstrained-minus-constrained jump in v_mm.
        # Continuity of v_m removes the first-order boundary term. dDelta changes
        # the marginal-value difference at fixed eta_star. Keeping v_m equal at
        # the moving cutoff gives deta_star = -dDelta/(Z*Delta_v_mm), which yields
        # the second-order correction below.
        Delta_v_mm = self.Delta_v_mm
        dDelta = dv_m[:, :1]+Delta_v_mm*(db+self.eta_star*dZ)
        d2barV_b += self.p_star*dDelta**2/Delta_v_mm

        # Marginal value of assets
        # Multiply expected marginal value of cash by the changing return R.
        dVa = dr*self.barV_b+self.R*dbarV_b
        d2Va = d2r*self.barV_b+2*dr*dbarV_b+self.R*d2barV_b
        return (dVa, d2Va), (dm_endog, d2m_endog)

    def step_3rd(self, dVa_next, d2Va_next, d3Va_next, du, d2u, d3u):
        """Differentiate current marginal value and the cash grid through order three.

        The dVa_next arguments perturb next period's marginal value; du, d2u,
        and d3u perturb current inputs (r,Z,Tr). Supply zeros for derivatives
        that are absent. Return (dVa,d2Va,d3Va) and (dm_endog,d2m_endog,d3m_endog).
        Each array has shape (state,asset).
        """
        # Continuation value and the endogenous grid
        # Differentiate m_endog=a_grid+Wa**(-eis).
        dWa = self.beta_Pi @ dVa_next
        d2Wa = self.beta_Pi @ d2Va_next
        d3Wa = self.beta_Pi @ d3Va_next
        dWa_squared = dWa*dWa
        dm_endog = self.dm_dWa*dWa
        d2m_endog = self.dm_dWa*d2Wa+self.d2m_dWa2*dWa_squared
        d3m_endog = (self.dm_dWa*d3Wa+3*self.d2m_dWa2*dWa*d2Wa
                    + self.d3m_dWa3*(dWa_squared*dWa))

        # Marginal value at fixed cash
        # At steady state, saving at cash m_endog[s,j] is a_grid[j], so
        # asset and cash responses line up pointwise. The spline grids stay
        # fixed; changes in m_endog are returned separately.
        # Differentiate u_c(m-a)=Wa(a) and v_m=Wa(a) at fixed m.
        # mps=da/dm converts marginal-value changes into saving responses.
        mps, u_cc = self.mps, self.u_cc
        dWaa, dWaaa = self.a_spline.first_second_derivatives(dWa)
        d2Waa = self.a_spline.derivative(d2Wa)
        da = -mps*dWa/u_cc
        da_m = self.m_spline.derivative(da)
        dv_m = mps*dWa
        dv_mm = self.mps_m*dWa+mps**2*dWaa
        d2v_m = mps*d2Wa+da_m*dWa+mps*da*dWaa
        d2v_mm = self.m_spline.derivative(d2v_m)
        # Differentiate v_m=u_c(m-a) again to obtain d2a, d2a_m, and d3v_m.
        d3v_m = marginal_value_at_fixed_cash_3rd(
            dWa, d2Wa, d3Wa, dWaa, dWaaa, d2Waa,
            da, da_m, d2v_m, d2v_mm, mps, self.mpc, u_cc, self.u_ccc, self.u_cccc)

        # Integrate while current cash changes
        # First integrate the responses with the borrowing cutoff fixed.
        q1, q2, q3 = self.m_spline.coefficients_three(dv_m, d2v_m, d3v_m)
        expectations = perturbation_expectations(self.maps, q1, q2, q3)

        # Compose with changing cash, correct the borrowing kink, and apply Va=R*barV_b.
        dVa, d2Va, d3Va = compose_marginal_value_3rd(
            expectations, (dv_m[:, 0], d2v_m[:, 0], dv_mm[:, 0]), du, d2u, d3u,
            self.a_grid, self.R, self.Z,
            (self.barV_b, self.barV_grad, self.barV_hess_fixed, self.barV_third_fixed),
            (self.eta_star, self.Delta_v_mm, self.Delta_v_mmm, self.p_star, self.p_m_star))
        return (dVa, d2Va, d3Va), (dm_endog, d2m_endog, d3m_endog)


    def __init__(self, hh, ss):
        """Prepare steady-state utility derivatives, splines, and income expectations."""
        self.R, self.Z = 1+ss["r"], ss["Z"]
        self.beta_Pi, self.a_grid = hh.beta*hh.Pi, hh.a_grid
        self.Va, self.m_endog = ss["Va"], ss["m_endog"]
        self.shape = self.Va.shape
        S, N = self.shape

        # Steady-state preparation
        # Utility and endogenous cash-grid derivatives.
        Wa = self.beta_Pi @ self.Va
        c = Wa**(-hh.eis)
        gamma = 1/hh.eis
        self.u_cc = -gamma*c**(-gamma-1)
        self.u_ccc = gamma*(gamma+1)*c**(-gamma-2)
        self.u_cccc = -gamma*(gamma+1)*(gamma+2)*c**(-gamma-3)
        self.dm_dWa = -hh.eis*Wa**(-hh.eis-1)
        self.d2m_dWa2 = hh.eis*(hh.eis+1)*Wa**(-hh.eis-2)
        self.d3m_dWa3 = -hh.eis*(hh.eis+1)*(hh.eis+2)*Wa**(-hh.eis-3)
        self.mpc, self.mps = ss["mpc"], 1-ss["mpc"]
        self.m_spline = Spline.from_grid(self.m_endog)
        self.a_spline = Spline.from_grid(hh.a_grid)
        self.mps_m = self.m_spline.derivative(self.mps)

        # Prepare E[v_mm], E[eta*v_mm], and the higher derivative moments,
        # including both constrained and unconstrained households.
        b = self.R*hh.a_grid+ss["Tr"]
        mu = -hh.sigma_xi**2/2+np.log(hh.e)  # Log mean of eta=e_s*eps.
        q_c = ss["q_policy"]  # Consumption spline coefficients.
        self.barV_grad, self.barV_hess_fixed, self.barV_third_fixed = (
            marginal_value_moments(q_c,self.m_endog,b,self.Z,mu,hh.sigma_xi,gamma))

        # Expectations of changing marginal-value splines, independent of q_c.
        matrices = cash_expectation_matrices(self.m_endog,b,self.Z,mu,hh.sigma_xi)
        self.maps = sparse_expectation_maps(matrices)
        self.barV_b = self.Va/self.R

        # Borrowing-threshold derivatives
        # Jumps are unconstrained minus constrained.
        # v_mm=u_cc*c_m and v_mmm=u_ccc*c_m**2+u_cc*c_mm.
        # Constrained consumption has c_m=1 and c_mm=0 at m_star.
        m_star = self.m_endog[:, 0]
        c_m, c_mm = self.m_spline.derivatives_at_lower_bound(q_c)
        u_cc = -gamma*m_star**(-gamma-1)
        u_ccc = gamma*(gamma+1)*m_star**(-gamma-2)
        self.Delta_v_mm = (u_cc*(c_m-1))[:, None]
        self.Delta_v_mmm = (u_ccc*(c_m**2-1)+u_cc*c_mm)[:, None]
        self.eta_star = (m_star[:, None]-b)/self.Z

        # Cash minus b is Z*eta, with log mean mu+log(Z).
        self.p_star, self.p_m_star = np.empty((S,N)), np.empty((S,N))
        for s in range(S):
            for j in range(N):
                self.p_star[s,j], self.p_m_star[s,j], _ = lognormal_density_derivatives(
                    m_star[s]-b[j],mu[s]+np.log(self.Z),hh.sigma_xi)

        # First-order maps
        # For each state, dVa/dWa = R * M @ C @ diag(mps):
        # C converts grid values to coefficients, and M takes expectations.
        C = self.m_spline.coefficient_matrix().reshape(S, N, N)
        self.dVa_dWa = np.empty((S, N, N))
        for s in range(S):
            M = matrices[0, s]
            self.dVa_dWa[s] = self.R*(M @ C[s])*self.mps[s][None, :]
        Vbb, VbZ = self.barV_grad
        dVa_dr = self.barV_b+self.R*Vbb*self.a_grid
        dVa_dZ = self.R*VbZ
        dVa_dTr = self.R*Vbb
        self.dVa_du = np.stack((dVa_dr, dVa_dZ, dVa_dTr), axis=-1)


@njit
def compose_marginal_value_3rd(expectations, cutoff_responses, du, d2u, d3u,
                               a_grid, R, Z, barV, cutoff):
    """Compose Va=R*E[v_m(b+Z*eta)] through third order, including the moving kink.

    expectations has shape (10,S,N); cutoff_responses contains the (S,) arrays
    (dv_m_min,d2v_m_min,dv_mm_min) at the borrowing cutoff.
    barV groups the level and fixed-cutoff partials; cutoff groups eta_star,
    the two derivative jumps, and cash densities. Return three (S,N) arrays.
    """
    dr, dZ, dTr = du
    d2r, d2Z, d2Tr = d2u
    d3r, d3Z, d3Tr = d3u
    barV_b, barV_grad, barV_hess_fixed, barV_third_fixed = barV
    eta_star, Delta_v_mm_grid, Delta_v_mmm_grid, p_star, p_m_star = cutoff
    dv_m_min, d2v_m_min, dv_mm_min = cutoff_responses
    dVa, d2Va, d3Va = (np.empty_like(barV_b), np.empty_like(barV_b), np.empty_like(barV_b))

    for s in range(barV_b.shape[0]):
        Delta_v_mm, Delta_v_mmm = Delta_v_mm_grid[s, 0], Delta_v_mmm_grid[s, 0]
        for i in range(barV_b.shape[1]):
            a = a_grid[i]
            db, d2b, d3b = dr*a+dTr, d2r*a+d2Tr, d3r*a+d3Tr
            db_squared = db*db
            (E_dv_m, E_d2v_m, E_d3v_m,
             E_dv_mm, E_eta_dv_mm, E_d2v_mm, E_eta_d2v_mm,
             E_dv_mmm, E_eta_dv_mmm, E_eta2_dv_mmm) = expectations[:, s, i]
            Vbb, VbZ = barV_grad[:, s, i]
            Vbbb, VbbZ, VbZZ = barV_hess_fixed[:, s, i]
            Vbbbb, VbbbZ, VbbZZ, VbZZZ = barV_third_fixed[:, s, i]

            # Apply the chain rule with dm=db+eta*dZ. The cached V partials
            # include constrained and unconstrained households, with the cutoff fixed.
            dbarV_b = E_dv_m+Vbb*db+VbZ*dZ
            d2barV_b = (E_d2v_m+2*db*E_dv_mm+2*dZ*E_eta_dv_mm
                        + Vbbb*db_squared+2*VbbZ*db*dZ+VbZZ*dZ**2+Vbb*d2b+VbZ*d2Z)
            d3barV_b = (
                E_d3v_m+3*db*E_d2v_mm+3*dZ*E_eta_d2v_mm
                + 3*db_squared*E_dv_mmm+6*db*dZ*E_eta_dv_mmm+3*dZ**2*E_eta2_dv_mmm
                + 3*d2b*E_dv_mm+3*d2Z*E_eta_dv_mm
                + Vbbbb*(db_squared*db)+3*VbbbZ*db_squared*dZ+3*VbbZZ*db*dZ**2+VbZZZ*dZ**3
                + 3*Vbbb*db*d2b+3*VbbZ*(db*d2Z+dZ*d2b)+3*VbZZ*dZ*d2Z
                + Vbb*d3b+VbZ*d3Z)

            # Borrowing-limit correction
            # Delta_v_mm and Delta_v_mmm are the jumps in v_mm and v_mmm.
            # Differentiate the marginal-value difference at fixed eta_star.
            # As above, deta_star = -dDelta/(Z*Delta_v_mm). At third order,
            # terms involving d2eta_star cancel by the same matching condition.
            dm = db+eta_star[s, i]*dZ
            d2m = d2b+eta_star[s, i]*d2Z
            dDelta = dv_m_min[s]+Delta_v_mm*dm
            dDelta_squared = dDelta*dDelta
            dDelta_cubed = dDelta_squared*dDelta
            d2Delta = d2v_m_min[s]+2*dv_mm_min[s]*dm+Delta_v_mmm*dm**2+Delta_v_mm*d2m
            dDelta_m = dv_mm_min[s]+Delta_v_mmm*dm
            # p is the density of cash at the kink; p_m is its cash derivative.
            p, p_m = p_star[s, i], p_m_star[s, i]
            d2barV_b += p*dDelta_squared/Delta_v_mm
            d3barV_b += (3*p*dDelta*d2Delta/Delta_v_mm
                         - 3*p*dDelta_squared*(dDelta_m+Delta_v_mm*dZ/Z)/Delta_v_mm**2
                         + p*dDelta_cubed*Delta_v_mmm/Delta_v_mm**3-p_m*dDelta_cubed/Delta_v_mm**2)

            # Marginal value of assets
            # Multiply expected marginal value of cash by the changing return R.
            dVa[s, i] = dr*barV_b[s, i]+R*dbarV_b
            d2Va[s, i] = d2r*barV_b[s, i]+2*dr*dbarV_b+R*d2barV_b
            d3Va[s, i] = (d3r*barV_b[s, i]+3*d2r*dbarV_b
                          + 3*dr*d2barV_b+R*d3barV_b)
    return dVa, d2Va, d3Va


@vectorize
def marginal_value_at_fixed_cash_3rd(
        dWa, d2Wa, d3Wa, dWaa, dWaaa, d2Waa,
        da, da_m, d2v_m, d2v_mm, mps, mpc, u_cc, u_ccc, u_cccc):
    """Third derivative of v_m at fixed cash, from u_c(m-a)=Wa(a).

    Numba applies this scalar formula elementwise to the supplied arrays.
    Subscripts a and m denote asset and cash derivatives; d denotes the shock.
    """
    da_squared = da*da
    d2a = (u_ccc*da_squared-d2v_m)/u_cc
    d2a_m = (u_cccc*mpc*da_squared+2*u_ccc*da*da_m
              - u_ccc*mpc*d2a-d2v_mm)/u_cc
    d3v_m = (d2a_m*dWa+2*da_m*da*dWaa+2*da_m*d2Wa
             + mps*d2a*dWaa+mps*da_squared*dWaaa
             + 2*mps*da*d2Waa+mps*d3Wa)
    return d3v_m
