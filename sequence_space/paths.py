"""Household transition derivatives through third order.

These functions differentiate household responses along a specified input path.
Their callers determine the income adjustments needed for general equilibrium.

du, d2u, and d3u are successive derivatives of the input path in shock size.
Their rows are dates and their columns are the inputs (r,Z,Tr).
The returned dictionaries contain derivatives of aggregate assets A, the asset
CDF F, and date-zero marginal value Va0. F[t] is the distribution entering
date t, while A[t] aggregates the end-of-period distribution F[t+1].

The one-period equations are in analytical_simple. See its
backward.md and forward.md for the derivations.
"""

import numpy as np


def first_order(backward, forward, du, *, initial_F=None, Va_forcing=None, Va_terminal=None):
    """Return first-order transition derivatives.

    initial_F is the initial CDF derivative dF0.
    Va_forcing[t] adds to the derivative of expected continuation marginal value at t.
    Va_terminal is the marginal-value derivative at date T.
    """
    T, shape = len(du), backward.shape

    # Backward: marginal value and saving policy.
    dVa = np.zeros(shape) if Va_terminal is None else Va_terminal
    # dm_endog is the change in cash needed to choose each asset-grid point.
    # The forward pass uses this policy change to update the distribution.
    dm_endog = np.empty((T,)+shape)
    for t in range(T-1, -1, -1):
        # Add any perturbation to continuation expectations.
        continuation = dVa if Va_forcing is None else dVa+Va_forcing[t]
        dVa, dm_endog[t] = backward.step_1st(continuation, du[t])
    dVa0 = dVa

    # Forward: distribution and aggregate assets.
    dF = np.zeros(shape) if initial_F is None else initial_F.copy()
    F_path = np.empty((T+1,)+shape)
    F_path[0] = dF
    dA = np.empty(T)
    for t in range(T):
        dF = forward.step_1st(dF, dm_endog[t], du[t])
        F_path[t+1] = dF
        dA[t] = forward.aggregate_assets(dF)
    return dict(dA=dA, dVa0=dVa0, dF=F_path)


def second_order(backward, forward, du, d2u=None, *,
                 initial_F=None, Va_forcing=None, Va_terminal=None):
    """Return first- and second-order transition derivatives.

    initial_F=(dF0,d2F0) gives the initial CDF derivatives.
    Va_forcing[t] adds to the first derivative of expected continuation marginal value at t.
    Va_terminal=(dVaT,d2VaT) gives marginal-value derivatives at date T.
    """
    T, shape = len(du), backward.shape
    d2u = np.zeros_like(du) if d2u is None else d2u

    # Backward: marginal value and saving policy.
    dVa, d2Va = np.zeros((2,)+shape) if Va_terminal is None else Va_terminal
    dm_endog, d2m_endog = np.empty((2,T)+shape)
    for t in range(T-1, -1, -1):
        # Add any perturbation to continuation expectations.
        continuation = dVa if Va_forcing is None else dVa+Va_forcing[t]
        (dVa,d2Va), (dm_endog[t],d2m_endog[t]) = backward.step_2nd(
            continuation,d2Va,du[t],d2u[t])
    dVa0, d2Va0 = dVa, d2Va

    # Forward: distribution and aggregate assets.
    dF, d2F = np.zeros((2,)+shape) if initial_F is None else initial_F
    F1, F2 = np.empty((2,T+1)+shape)
    F1[0], F2[0] = dF, d2F
    dA, d2A = np.empty((2,T))
    for t in range(T):
        dF,d2F = forward.step_2nd(dF,d2F,dm_endog[t],d2m_endog[t],du[t],d2u[t])
        F1[t+1], F2[t+1] = dF,d2F
        dA[t], d2A[t] = forward.aggregate_assets(dF), forward.aggregate_assets(d2F)
    return dict(dA=dA, d2A=d2A, dVa0=dVa0, d2Va0=d2Va0, dF=F1, d2F=F2)


def third_order(backward, forward, du, d2u=None, d3u=None, *, initial_F=None, Va_terminal=None):
    """Return transition derivatives through third order.

    initial_F=(dF0,d2F0,d3F0) gives the initial CDF derivatives.
    Va_terminal=(dVaT,d2VaT,d3VaT) gives marginal-value derivatives at date T.
    """
    T, shape = len(du), backward.shape
    d2u = np.zeros_like(du) if d2u is None else d2u
    d3u = np.zeros_like(du) if d3u is None else d3u

    # Backward: marginal value and saving policy.
    dVa, d2Va, d3Va = np.zeros((3,)+shape) if Va_terminal is None else Va_terminal
    dm_endog, d2m_endog, d3m_endog = np.empty((3,T)+shape)
    for t in range(T-1, -1, -1):
        (dVa,d2Va,d3Va), (dm_endog[t],d2m_endog[t],d3m_endog[t]) = backward.step_3rd(
            dVa,d2Va,d3Va,du[t],d2u[t],d3u[t])
    dVa0,d2Va0,d3Va0 = dVa,d2Va,d3Va

    # Forward: distribution and aggregate assets.
    dF,d2F,d3F = np.zeros((3,)+shape) if initial_F is None else initial_F
    F1,F2,F3 = np.empty((3,T+1)+shape)
    F1[0],F2[0],F3[0] = dF,d2F,d3F
    dA,d2A,d3A = np.empty((3,T))
    for t in range(T):
        dF,d2F,d3F = forward.step_3rd(
            dF,d2F,d3F,dm_endog[t],d2m_endog[t],d3m_endog[t],du[t],d2u[t],d3u[t])
        F1[t+1],F2[t+1],F3[t+1] = dF,d2F,d3F
        dA[t],d2A[t],d3A[t] = (forward.aggregate_assets(dF),
                              forward.aggregate_assets(d2F),forward.aggregate_assets(d3F))
    return dict(dA=dA,d2A=d2A,d3A=d3A,dVa0=dVa0,d2Va0=d2Va0,d3Va0=d3Va0,
                dF=F1,d2F=F2,d3F=F3)
