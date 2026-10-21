"""Propagate changes in the household's asset distribution through its saving policy.

For target cash on hand m (m_j in forward.md), a=(m-Tr-Z*eta)/R is the
greatest inherited asset level consistent with saving below the corresponding
target asset level.
Thus the probability of saving below that target is E[F(a)], where F is the
inherited CDF. Pi_F then applies the persistent-income transition.

Here R=1+r, eta=e_s*eps is income before scaling by Z, and eps is the iid
income shock. Derivatives are with respect to the shock shifter theta at the
steady state, theta=0. See forward.md for the derivation.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from spline.fixed_grid import Spline
from .expectations import (asset_expectation_matrices, asset_density_moments,
                           sparse_expectation_maps, lognormal_density_derivatives)


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
        # The named matrices apply these expectations to spline coefficients.
        q1 = self.a_spline.coefficients(dF).ravel()
        q2 = self.a_spline.coefficients(d2F).ravel()
        M = self.maps
        E_dF = (M.value @ q1).reshape(self.shape)
        E_d2F = (M.value @ q2).reshape(self.shape)
        E_dF_a = (M.first @ q1).reshape(self.shape)
        E_a_dF_a = (M.z_first @ q1).reshape(self.shape)
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
        R, Z = self.R, self.Z
        dr, dZ, dTr = du
        d2r, d2Z, d2Tr = d2u
        d3r, d3Z, d3Tr = d3u
        m, Ze = self.m_endog, self.Ze

        # Asset-cutoff derivatives
        # Differentiate R*a=m-Tr-Z*eta at fixed income draw.
        # Write d^k a=alpha_k*a+beta_k to reuse asset moments.
        alpha1 = dZ/Z-dr/R
        beta1 = (dm_endog-dTr-(m-self.Tr)*dZ/Z)/R
        alpha2 = d2Z/Z-d2r/R-2*dr*alpha1/R
        beta2 = (d2m_endog-d2Tr-(m-self.Tr)*d2Z/Z-2*dr*beta1)/R
        alpha3 = d3Z/Z-d3r/R-3*(d2r*alpha1+dr*alpha2)/R
        beta3 = (d3m_endog-d3Tr-(m-self.Tr)*d3Z/Z-3*d2r*beta1-3*dr*beta2)/R

        # CDF expectation
        # Powers of a supply the weights from the cutoff derivatives.
        q1, q2, q3 = self.a_spline.coefficients_three(dF, d2F, d3F)
        dF_a_min, _ = self.a_spline.derivatives_at_lower_bound(q1)
        dF_a_min = dF_a_min[:, None]
        q1, q2, q3 = q1.ravel(), q2.ravel(), q3.ravel()
        M = self.maps
        E_dF = (M.value @ q1).reshape(self.shape)
        E_d2F = (M.value @ q2).reshape(self.shape)
        E_d3F = (M.value @ q3).reshape(self.shape)
        E_dF_a = (M.first @ q1).reshape(self.shape)
        E_a_dF_a = (M.z_first @ q1).reshape(self.shape)
        E_d2F_a = (M.first @ q2).reshape(self.shape)
        E_a_d2F_a = (M.z_first @ q2).reshape(self.shape)
        E_dF_aa = (M.second @ q1).reshape(self.shape)
        E_a_dF_aa = (M.z_second @ q1).reshape(self.shape)
        E_a2_dF_aa = (M.z2_second @ q1).reshape(self.shape)
        E_Fa, E_a_Fa = self.E_Fa
        E_Faa, E_a_Faa, E_a2_Faa = self.E_Faa

        # Apply the chain rule to F(a(eps)) with shock integration limits fixed.
        E_Faa_da2 = alpha1**2*E_a2_Faa+2*alpha1*beta1*E_a_Faa+beta1**2*E_Faa
        first = E_dF+alpha1*E_a_Fa+beta1*E_Fa
        second = (E_d2F+2*(alpha1*E_a_dF_a+beta1*E_dF_a)
                  + E_Faa_da2+alpha2*E_a_Fa+beta2*E_Fa)
        third = (
            E_d3F+3*(alpha1*E_a_d2F_a+beta1*E_d2F_a)
            + 3*(alpha1**2*E_a2_dF_aa+2*alpha1*beta1*E_a_dF_aa+beta1**2*E_dF_aa)
            + 3*(alpha2*E_a_dF_a+beta2*E_dF_a)
            + 3*(alpha1*alpha2*E_a2_Faa+(alpha1*beta2+alpha2*beta1)*E_a_Faa
                 + beta1*beta2*E_Faa)+alpha3*E_a_Fa+beta3*E_Fa)

        # Integration by parts
        # The expression above omits E[F_aaa*(da)^3]. Integration by parts
        # supplies it using F_aa and p_eps/p, avoiding the spline's third
        # derivative, which jumps at knots. Treat the CDF as locally flat
        # at the upper asset endpoint, so its boundary term vanishes.
        da_min = alpha1*self.a_grid[0]+beta1
        d2a_min = alpha2*self.a_grid[0]+beta2
        d3a_min = alpha3*self.a_grid[0]+beta3
        F_min, F_a_min, F_aa_min = self.F_min, self.F_a_min, self.F_aa_min
        p, p_eps, p_eps_eps = self.density
        E_Faa_dlogp, E_a_Faa_dlogp, E_a2_Faa_dlogp, E_a3_Faa_dlogp = self.E_Faa_dlogp
        E_Faa_dlogp_da3 = (
            alpha1**3*E_a3_Faa_dlogp+3*alpha1**2*beta1*E_a2_Faa_dlogp
            + 3*alpha1*beta1**2*E_a_Faa_dlogp+beta1**3*E_Faa_dlogp)
        third += -R/Ze*F_aa_min*da_min**3*p-3*alpha1*E_Faa_da2+R/Ze*E_Faa_dlogp_da3

        # Borrowing-limit correction
        # Differentiate a(eps_bar)=a_min with a_min fixed.
        eps1 = R/Ze*da_min
        eps2 = R/Ze*d2a_min-2*alpha1*eps1
        eps3 = R/Ze*d3a_min-3*alpha1*eps2-3*alpha2*eps1
        a_eps = -Ze/R
        # For g=F(a(eps))*p(eps), d holds eps fixed and _eps differentiates
        # the income draw. Add the moving-limit terms from forward.md.
        g = F_min*p
        dg = (dF[:, :1]+F_a_min*da_min)*p
        d2g = (d2F[:, :1]+2*dF_a_min*da_min+F_aa_min*da_min**2+F_a_min*d2a_min)*p
        g_eps = F_a_min*a_eps*p+F_min*p_eps
        dg_eps = ((dF_a_min+F_aa_min*da_min)*a_eps+F_a_min*alpha1*a_eps)*p
        dg_eps += (dF[:, :1]+F_a_min*da_min)*p_eps
        g_eps_eps = F_aa_min*a_eps**2*p+2*F_a_min*a_eps*p_eps+F_min*p_eps_eps
        first += g*eps1
        second += 2*dg*eps1+g_eps*eps1**2+g*eps2
        third += (3*d2g*eps1+3*dg_eps*eps1**2+g_eps_eps*eps1**3
                  + 3*dg*eps2+3*g_eps*eps1*eps2+g*eps3)

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

        # Derivative moments of the steady-state CDF.
        q_F = self.a_spline.coefficients(self.F)
        q = q_F.ravel()
        M = self.maps
        self.E_Fa = np.stack((M.first @ q, M.z_first @ q)).reshape(2,S,N)
        self.E_Faa = np.stack((M.second @ q, M.z_second @ q, M.z2_second @ q)).reshape(3,S,N)

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
            rows = slice(s*N, (s+1)*N)
            M = self.maps.value[rows, rows]
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
