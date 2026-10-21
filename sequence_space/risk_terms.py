"""Household derivatives for anticipated risk and its effects on shock transmission."""

import numpy as np
from .paths import first_order, second_order


def stationary_response(backward, forward, du, Va_forcing):
    """Stationary derivatives of Va, F, and assets under constant du and Va_forcing.

    du shifts the inputs (r,Z,Tr). Va_forcing shifts expected continuation marginal value.
    Return dVa, dF, and dA for marginal value, the asset CDF, and aggregate assets.
    """
    # At a steady state, the marginal-value derivative is the same at every date.
    # Solve its fixed point with the expectation forcing added each period.
    dVa = np.zeros(backward.shape)
    for _ in range(10000):
        updated, dm_endog = backward.step_1st(dVa+Va_forcing, du)
        if np.max(np.abs(updated-dVa)) < 1e-10:
            dVa = updated
            break
        dVa = updated
    else:
        raise RuntimeError("Stationary marginal-value response did not converge")

    _, dm_endog = backward.step_1st(dVa+Va_forcing, du)

    # The resulting saving-policy change determines the stationary CDF response.
    dF = np.zeros(backward.shape)
    for _ in range(10000):
        updated = forward.step_1st(dF, dm_endog, du)
        if np.max(np.abs(updated-dF)) < 1e-12:
            dF = updated
            break
        dF = updated
    else:
        raise RuntimeError("Stationary distribution response did not converge")
    return dict(dVa=dVa, dF=dF, dA=forward.aggregate_assets(dF))


def rss_response(backward, forward, du, stationary, Va00, rss_du=None):
    """How starting from the risky steady state changes the response to a shock du.

    Return changes in the asset and date-zero marginal-value responses, before
    multiplying by sigma^2/2. The caller adjusts the shock's income path in GE.
    """
    # Future shocks with variance sigma^2 add sigma^2*Va00/2 to expected marginal
    # value, where Va00 is its second derivative in a future shock. This changes
    # saving and the steady-state distribution. stationary gives the resulting
    # changes in Va and F per unit of sigma^2/2. rss_du gives those in (r,Z,Tr).
    T = len(du)
    rss_du = np.zeros(3) if rss_du is None else rss_du

    # Maintain these steady-state changes throughout the transition. In PE the
    # inputs stay fixed. In GE rss_du includes the steady-state income adjustment.
    # Va00 stays fixed here: changes in it after the shock belong to the TVP effect.
    rss_path = np.broadcast_to(rss_du, du.shape)
    forcing = np.broadcast_to(Va00, (T,)+backward.shape)

    # Households start with the changed distribution and return to the changed
    # marginal value after date T. We need only the first-order change in each.
    terminal_Va = (stationary['dVa'], np.zeros(backward.shape))
    initial_F = (stationary['dF'], np.zeros(backward.shape))

    # Let x scale the shock and q scale the steady-state changes above. Moving
    # both together gives the second derivative A_xx + 2*A_xq + A_qq.
    # Subtract the shock-only and risk-only calculations and divide by two
    # to get A_xq, the change in the shock response per unit of q.
    # Setting q=sigma^2/2 gives the RSS effect.
    impulse = second_order(backward, forward, du)
    risk = second_order(backward, forward, rss_path, initial_F=initial_F,
                        Va_forcing=forcing, Va_terminal=terminal_Va)
    both = second_order(backward, forward, du+rss_path, initial_F=initial_F,
                        Va_forcing=forcing, Va_terminal=terminal_Va)
    return dict(dA=(both['d2A']-impulse['d2A']-risk['d2A'])/2,
                dVa0=(both['d2Va0']-impulse['d2Va0']-risk['d2Va0'])/2)


def tvp_response(backward, forward, Va_xyy, T):
    """Household response to the change in expected marginal value from future risk.

    Va_xyy[l] differentiates marginal value once in today's shock and twice
    in the innovation l dates ahead, as returned by third_histories.
    No GE correction or sigma^2/2 factor is applied here.
    """
    forcing = np.zeros((T,)+backward.shape)
    # Risk at date l enters expectations at l-1. The date-zero shock is already known.
    forcing[:len(Va_xyy)-1] = Va_xyy[1:]
    return first_order(backward, forward, np.zeros((T, 3)), Va_forcing=forcing)
