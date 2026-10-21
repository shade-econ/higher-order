"""Same equations and interface as analytical_simple.forward, with faster evaluation.

Expectation products share sparse indices and run in one compiled loop.
Some key operations are extracted into compiled Numba routines, and repeated
squares and cubes are computed once and reused. See analytical_simple/forward.py
and analytical_simple/forward.md for the derivation.
"""

import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline

from spline.fixed_grid import Spline
from .expectations import (asset_expectation_matrices, asset_density_moments,
                           sparse_expectation_maps, perturbation_expectations,
                           lognormal_density_derivatives)


class Forward:
    """Differentiate the distribution update around one household steady state."""

    def step_1st(self, dF, dm_endog, du):
        """Return the first CDF derivative using prepared linear maps.

        dF and dm_endog change the inherited CDF and target cash grid; du changes
        inputs (r,Z,Tr). A trailing axis may contain several directions.
        """
        S, N = self.shape
        before_markov = (self.dtildeF_dF @ dF.reshape(S, N, -1)
                         + self.dtildeF_dm[..., None]*dm_endog.reshape(S, N, -1)
                         + self.dtildeF_du @ du.reshape(3, -1))
        return (self.Pi_F @ before_markov.reshape(S, -1)).reshape(dF.shape)

    def step_2nd(self, dF, d2F, dm_endog, d2m_endog, du, d2u):
        """Return the first two derivatives of the next-period CDF.

        dF and dm_endog describe changes in the inherited CDF and target cash
        grid. du gives changes in (r,Z,Tr); d2F, d2m_endog, and d2u give their
        second derivatives. Return (dF_next,d2F_next), each with shape
        (state,asset).
        """
        R, Z, m, Ze = self.R, self.Z, self.m_endog, self.Ze
        dr, dZ, dTr = du
        d2r, d2Z, d2Tr = d2u

        # Asset-cutoff derivatives
        # Differentiate the asset cutoff at fixed income draw.
        # Write da=alpha1*a+beta1 and d2a=alpha2*a+beta2 to reuse asset moments.
        alpha1 = dZ/Z-dr/R
        beta1 = (dm_endog-dTr-(m-self.Tr)*dZ/Z)/R
        alpha2 = d2Z/Z-d2r/R-2*dr*alpha1/R
        beta2 = (d2m_endog-d2Tr-(m-self.Tr)*d2Z/Z-2*dr*beta1)/R

        # CDF expectation
        # Integrate the inherited CDF changes and their asset derivatives.
        q1 = self.a_spline.coefficients(dF)
        q2 = self.a_spline.coefficients(d2F)
        E_dF, E_d2F, E_dF_a, E_a_dF_a = perturbation_expectations(self.maps, q1, q2)
        E_Fa, E_a_Fa = self.E_Fa
        E_Faa, E_a_Faa, E_a2_Faa = self.E_Faa
        first = E_dF+alpha1*E_a_Fa+beta1*E_Fa
        second = (E_d2F+2*(alpha1*E_a_dF_a+beta1*E_dF_a)
                  + alpha1**2*E_a2_Faa+2*alpha1*beta1*E_a_Faa+beta1**2*E_Faa
                  + alpha2*E_a_Fa+beta2*E_Fa)

        # Borrowing-limit correction
        # Differentiate a(eps_bar)=a_min with a_min fixed.
        # da_min evaluates the fixed-shock cutoff response at this boundary.
        da_min = alpha1*self.a_grid[0]+beta1
        d2a_min = alpha2*self.a_grid[0]+beta2
        eps1 = R/Ze*da_min
        eps2 = R/Ze*d2a_min-2*alpha1*eps1
        F_min, F_a_min = self.F_min, self.F_a_min
        p, p_eps = self.density[:2]
        # For g=F(a(eps))*p(eps), d holds eps fixed and _eps differentiates
        # the income draw. Add the moving-limit terms from forward.md.
        g = F_min*p
        dg = (dF[:, :1]+F_a_min*da_min)*p
        g_eps = F_a_min*(-Ze/R)*p+F_min*p_eps
        first += g*eps1
        second += 2*dg*eps1+g_eps*eps1**2+g*eps2

        # Persistent-state transition
        return self.Pi_F @ first, self.Pi_F @ second

    def step_3rd(self, dF, d2F, d3F, dm_endog, d2m_endog, d3m_endog, du, d2u, d3u):
        """Return the first three derivatives of the next-period CDF.

        dF and dm_endog change the inherited CDF and target cash grid; du changes
        inputs (r,Z,Tr). The d2 and d3 arguments give their second and third
        derivatives. Return (dF_next,d2F_next,d3F_next), each with shape
        (state,asset).
        """
        # Asset-cutoff derivatives: d^k a=alpha_k*a+beta_k at fixed income draw.
        cutoff = asset_cutoff_derivatives_3rd(
            dm_endog, d2m_endog, d3m_endog, du, d2u, d3u, self.R, self.Z, self.Tr, self.m_endog)

        # CDF expectation
        # Powers of a supply the weights from the cutoff derivatives.
        q1, q2, q3 = self.a_spline.coefficients_three(dF, d2F, d3F)
        dF_a_min, _ = self.a_spline.derivatives_at_lower_bound(q1)
        E_perturb = perturbation_expectations(self.maps, q1, q2, q3)
        boundary_responses = (dF[:, 0], d2F[:, 0], dF_a_min)
        first, second, third = cdf_expectation_derivatives_3rd(
            E_perturb, cutoff, boundary_responses, self.E_Fa, self.E_Faa,
            self.E_Faa_dlogp, (self.F_min, self.F_a_min, self.F_aa_min),
            self.density, self.R, self.Ze, self.a_grid[0])

        # Persistent-state transition
        return self.Pi_F @ first, self.Pi_F @ second, self.Pi_F @ third

    def __init__(self, hh, ss):
        """Prepare CDF expectations, borrowing-threshold derivatives, and linear maps."""
        self.R, self.Z, self.Tr = 1+ss["r"], ss["Z"], ss["Tr"]
        self.a_grid, self.m_endog = hh.a_grid, ss["m_endog"]
        self.F, self.A, self.Pi_F = ss["F"], ss["A"], hh.Pi_F
        self.shape = self.F.shape
        S, N = self.shape
        self.Ze = (self.Z*hh.e)[:, None]
        self.a_spline = Spline.from_grid(hh.a_grid)

        # Steady-state preparation
        # Inherited assets at target cash m are (m-Tr-Z*e*eps)/R.
        # Integrate the CDF derivatives and prepare matrices for its changes.
        a_shift = (self.m_endog-self.Tr)/self.R
        lam = self.Z*hh.e/self.R
        mu = np.full(S,-hh.sigma_xi**2/2)
        matrices = asset_expectation_matrices(hh.a_grid,a_shift,lam,mu,hh.sigma_xi)
        self.maps = sparse_expectation_maps(matrices)

        # Steady-state E[a**j F_a], j=0,1, and E[a**j F_aa], j=0,1,2.
        q_F = self.a_spline.coefficients(self.F)
        self.E_Fa = (matrices[1:3] @ q_F[..., None])[..., 0]
        self.E_Faa = (matrices[3:6] @ q_F[..., None])[..., 0]

        # Third-order integration by parts also needs moments weighted by p'/p.
        self.E_Faa_dlogp = asset_density_moments(hh.a_grid,q_F,a_shift,lam,mu,hh.sigma_xi)

        # Borrowing-threshold derivatives
        self.F_min = self.F[:, :1]
        F_a_min, F_aa_min = self.a_spline.derivatives_at_lower_bound(q_F)
        self.F_a_min, self.F_aa_min = F_a_min[:, None], F_aa_min[:, None]

        eps_bar = (a_shift-hh.a_grid[0])/lam[:, None]
        self.density = np.zeros((3,S,N))
        for s in range(S):
            for j in range(N):
                if eps_bar[s,j] > 0:
                    self.density[:,s,j] = lognormal_density_derivatives(
                        eps_bar[s,j],mu[s],hh.sigma_xi)

        # First-order maps
        # Cash and input derivatives include the moving borrowing threshold.
        E_Fa, E_a_Fa = self.E_Fa
        p = self.density[0]
        E_eta_Fa = ((self.m_endog-self.Tr)*E_Fa-self.R*E_a_Fa)/self.Z
        self.dtildeF_dm = E_Fa/self.R+self.F_min*p/self.Ze
        dtildeF_dr = -E_a_Fa/self.R-hh.a_grid[0]*self.F_min*p/self.Ze
        dtildeF_dZ = -E_eta_Fa/self.R-self.F_min*p*eps_bar/self.Z
        dtildeF_dTr = -self.dtildeF_dm
        self.dtildeF_du = np.stack((dtildeF_dr, dtildeF_dZ, dtildeF_dTr), axis=-1)

        # Compose interpolation with expectation: dtildeF/dF = M @ C.
        C = self.a_spline.coefficient_matrix()
        self.dtildeF_dF = np.empty((S, N, N))
        for s in range(S):
            M = matrices[0, s]
            self.dtildeF_dF[s] = M @ C

        # Mean-asset weights
        # Each column interpolates a unit vector of grid values. Its integral
        # gives the corresponding weight in dA=-sum_s pi_s integral dF_s da.
        unit_splines = CubicSpline(hh.a_grid, np.eye(N), axis=0)
        self.dA_dF = -hh.pi[:, None]*unit_splines.integrate(hh.a_grid[0], hh.a_grid[-1])


    def aggregate_assets(self, dF):
        """Return the change in mean assets implied by the CDF change dF.

        If dF contains direction columns, return one asset response per column.
        """
        return np.einsum("sn,sn...->...", self.dA_dF, dF)


@njit
def asset_cutoff_derivatives_3rd(dm_endog, d2m_endog, d3m_endog, du, d2u, d3u, R, Z, Tr, m):
    """Differentiate R*a=m-Tr-Z*eta at fixed eta: d^k a=alpha_k*a+beta_k.

    Return (alpha1,beta1,alpha2,beta2,alpha3,beta3), with scalar alphas and
    (S,N) betas. dm_endog,d2m_endog,d3m_endog perturb m; du,d2u,d3u perturb (r,Z,Tr).
    """
    dr, dZ, dTr = du
    d2r, d2Z, d2Tr = d2u
    d3r, d3Z, d3Tr = d3u
    alpha1 = dZ/Z-dr/R
    alpha2 = d2Z/Z-d2r/R-2*dr*alpha1/R
    alpha3 = d3Z/Z-d3r/R-3*(d2r*alpha1+dr*alpha2)/R
    betas = np.empty((3, *m.shape))
    for s in range(m.shape[0]):
        for i in range(m.shape[1]):
            beta1 = (dm_endog[s,i]-dTr-(m[s,i]-Tr)*dZ/Z)/R
            beta2 = (d2m_endog[s,i]-d2Tr-(m[s,i]-Tr)*d2Z/Z-2*dr*beta1)/R
            beta3 = (d3m_endog[s,i]-d3Tr-(m[s,i]-Tr)*d3Z/Z-3*d2r*beta1-3*dr*beta2)/R
            betas[0,s,i], betas[1,s,i], betas[2,s,i] = beta1, beta2, beta3
    return alpha1, betas[0], alpha2, betas[1], alpha3, betas[2]


@njit
def cdf_expectation_derivatives_3rd(
        E_perturb, cutoff, boundary_responses, moments_Fa, moments_Faa,
        moments_Faa_dlogp, F_boundary, density, R, Ze_by_state, a_min):
    """Differentiate E[F(a(eps))] through order three, before the Markov step.

    cutoff gives (alpha1,beta1,alpha2,beta2,alpha3,beta3) for d^k a=alpha_k*a+beta_k.
    E_perturb contains the ten moments returned by perturbation_expectations;
    boundary_responses gives (dF_min,d2F_min,dF_a_min) by persistent state.
    F_boundary gives (F_min,F_a_min,F_aa_min); moments and density are evaluated
    at each query (s,i). Return three (S,N) CDF derivative arrays.
    """
    alpha1, beta1_grid, alpha2, beta2_grid, alpha3, beta3_grid = cutoff
    alpha1_squared = alpha1*alpha1
    S, N = beta1_grid.shape
    out = np.empty((3, S, N))
    for s in range(S):
        Ze = Ze_by_state[s, 0]
        F_min, F_a_min, F_aa_min = (F_boundary[0][s, 0], F_boundary[1][s, 0],
                                     F_boundary[2][s, 0])
        dF_min, d2F_min, dF_a_min = (boundary_responses[0][s], boundary_responses[1][s],
                                    boundary_responses[2][s])
        for i in range(N):
            beta1, beta2, beta3 = beta1_grid[s, i], beta2_grid[s, i], beta3_grid[s, i]
            beta1_squared = beta1*beta1
            (E_dF, E_d2F, E_d3F,
             E_dF_a, E_a_dF_a, E_d2F_a, E_a_d2F_a,
             E_dF_aa, E_a_dF_aa, E_a2_dF_aa) = E_perturb[:, s, i]
            E_Fa, E_a_Fa = moments_Fa[:, s, i]
            E_Faa, E_a_Faa, E_a2_Faa = moments_Faa[:, s, i]
            (E_Faa_dlogp, E_a_Faa_dlogp,
             E_a2_Faa_dlogp, E_a3_Faa_dlogp) = moments_Faa_dlogp[:, s, i]
            p, p_eps, p_eps_eps = density[:, s, i]

            # Apply the chain rule to F(a(eps)) with shock integration limits fixed.
            E_Faa_da2 = alpha1_squared*E_a2_Faa+2*alpha1*beta1*E_a_Faa+beta1_squared*E_Faa
            first = E_dF+alpha1*E_a_Fa+beta1*E_Fa
            second = (E_d2F+2*(alpha1*E_a_dF_a+beta1*E_dF_a)
                      + E_Faa_da2+alpha2*E_a_Fa+beta2*E_Fa)
            third = (
                E_d3F+3*(alpha1*E_a_d2F_a+beta1*E_d2F_a)
                + 3*(alpha1_squared*E_a2_dF_aa+2*alpha1*beta1*E_a_dF_aa+beta1_squared*E_dF_aa)
                + 3*(alpha2*E_a_dF_a+beta2*E_dF_a)
                + 3*(alpha1*alpha2*E_a2_Faa+(alpha1*beta2+alpha2*beta1)*E_a_Faa
                     + beta1*beta2*E_Faa)+alpha3*E_a_Fa+beta3*E_Fa)

            # Integration by parts
            # The expression above omits E[F_aaa*(da)^3]. Integration by parts
            # supplies it using F_aa and p_eps/p, avoiding the spline's third
            # derivative, which jumps at knots. Treat the CDF as locally flat
            # at the upper asset endpoint, so its boundary term vanishes.
            da_min = alpha1*a_min+beta1
            da_min_squared = da_min*da_min
            d2a_min = alpha2*a_min+beta2
            d3a_min = alpha3*a_min+beta3
            E_Faa_dlogp_da3 = (
                (alpha1_squared*alpha1)*E_a3_Faa_dlogp+3*alpha1_squared*beta1*E_a2_Faa_dlogp
                + 3*alpha1*beta1_squared*E_a_Faa_dlogp+(beta1_squared*beta1)*E_Faa_dlogp)
            third += -R/Ze*F_aa_min*(da_min_squared*da_min)*p-3*alpha1*E_Faa_da2+R/Ze*E_Faa_dlogp_da3

            # Borrowing-limit correction
            # Differentiate a(eps_bar)=a_min with a_min fixed.
            eps1 = R/Ze*da_min
            eps1_squared = eps1*eps1
            eps2 = R/Ze*d2a_min-2*alpha1*eps1
            eps3 = R/Ze*d3a_min-3*alpha1*eps2-3*alpha2*eps1
            a_eps = -Ze/R
            # For g=F(a(eps))*p(eps), d holds eps fixed and _eps differentiates
            # the income draw. Add the moving-limit terms from forward.md.
            g = F_min*p
            dg = (dF_min+F_a_min*da_min)*p
            d2g = (d2F_min+2*dF_a_min*da_min+F_aa_min*da_min_squared+F_a_min*d2a_min)*p
            g_eps = F_a_min*a_eps*p+F_min*p_eps
            dg_eps = ((dF_a_min+F_aa_min*da_min)*a_eps+F_a_min*alpha1*a_eps)*p
            dg_eps += (dF_min+F_a_min*da_min)*p_eps
            g_eps_eps = F_aa_min*a_eps**2*p+2*F_a_min*a_eps*p_eps+F_min*p_eps_eps
            first += g*eps1
            second += 2*dg*eps1+g_eps*eps1_squared+g*eps2
            third += (3*d2g*eps1+3*dg_eps*eps1_squared+g_eps_eps*(eps1_squared*eps1)
                      + 3*dg*eps2+3*g_eps*eps1*eps2+g*eps3)

            out[0, s, i], out[1, s, i], out[2, s, i] = first, second, third
    return out[0], out[1], out[2]
