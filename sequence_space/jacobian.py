"""Numerical and analytical asset vs. income Jacobians by fake news algorithm."""

import numpy as np
from spline import core as spline
from smooth_sim.utils import integration_weights_spline


def income_jacobian_numerical(hh, ss, T, *, h=1e-4):
    """Return J[t,s] = dA[t]/dZ[s] using numerical differentiation and expectation functions."""
    shape = ss["F"].shape
    dF_news = np.empty((T,)+shape)
    Vp = Vm = ss["Va"]

    # News changes policies backward; each CDF response holds inherited F fixed.
    for lead in range(T):
        dZ = h if lead == 0 else 0.
        Vp, (_, mp) = hh.backward_iteration(Vp, ss["r"], ss["Z"]+dZ, ss["Tr"])
        Vm, (_, mm) = hh.backward_iteration(Vm, ss["r"], ss["Z"]-dZ, ss["Tr"])
        Fp = hh.forward_iteration(ss["F"], mp, ss["r"], ss["Z"]+dZ, ss["Tr"])
        Fm = hh.forward_iteration(ss["F"], mm, ss["r"], ss["Z"]-dZ, ss["Tr"])
        dF_news[lead] = (Fp-Fm)/(2*h)

    # Expectation functions for assets, starting from a.
    E = np.empty((T,)+shape)
    E[0] = hh.a_grid
    for t in range(1, T):
        E[t] = hh.expectation_iteration(
            E[t-1], ss["m_endog"], ss["r"], ss["Z"], ss["Tr"])

    # Express expectation functions in the spline basis and CDF news as integration weights.
    qE, weights = np.empty_like(E), np.empty_like(dF_news)
    for t in range(T):
        for s in range(shape[0]):
            qE[t,s] = spline.interp(hh.a_grid, E[t,s])
            qF = spline.interp(hh.a_grid, dF_news[t,s])
            weights[t,s] = integration_weights_spline(hh.a_grid, qF)

    # Pair expectation horizon t with news date s, integrating over assets and states.
    F = (qE*hh.pi[None,:,None]).reshape(T,-1) @ weights.reshape(T,-1).T
    return fake_news_to_jacobian(F)


def income_jacobian_analytical(backward, forward, T):
    """Return J[t,s] = dA[t]/dZ[s] using analytical household linear maps."""
    shape = forward.shape
    dVa, zero_F = np.zeros((2,)+shape)
    dF_news = np.empty((T,)+shape)

    # Propagate income news backward, then apply its one-period CDF effect.
    for lead in range(T):
        du = np.array([0., 1. if lead == 0 else 0., 0.])
        dVa, dm_endog = backward.step_1st(dVa, du)
        dF_news[lead] = forward.step_1st(zero_F, dm_endog, du)

    # Propagate the asset functional through the transposed distribution map.
    dA_dF = np.empty((T,)+shape)
    dA_dF[0] = forward.dA_dF
    for t in range(1, T):
        after_markov = forward.Pi_F.T @ dA_dF[t-1]
        dA_dF[t] = np.einsum("sij,si->sj", forward.dtildeF_dF, after_markov)

    F = dA_dF.reshape(T,-1) @ dF_news.reshape(T,-1).T
    return fake_news_to_jacobian(F)


def fake_news_to_jacobian(F):
    """Convert fake news F to Jacobian J via J[t,s] = F[t,s] + J[t-1,s-1]."""
    J = F.copy()
    for t in range(1, len(J)):
        J[t,1:] += J[t-1,:-1]
    return J
