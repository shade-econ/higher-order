"""Discrete counterpart of the smooth SIM paper calibration, without recalibration."""

import numpy as np

from smooth_sim.model import paper_calibration
from . import sim_steady_state_fast as sim


def paper_steady_state():
    """Return the paper steady state with five iid income states and unchanged parameters."""
    p = paper_calibration()
    n_xi = 5
    # Match log-income variance and normalize mean income to one.
    xi, pi_xi, Pi_xi = sim.discretize_income(0., p["sigma_xi"], n_xi)
    assert np.isclose(pi_xi @ xi, 1.)
    log_xi = np.log(xi)
    assert np.isclose(pi_xi @ (log_xi-pi_xi @ log_xi)**2, p["sigma_xi"]**2)

    # Order states by persistent state, then iid income.
    e = np.kron(p["e"], xi)
    ss = sim.steady_state(
        Pi=np.kron(p["Pi"], Pi_xi), a_grid=p["a_grid"], y=p["Z"]*e+p["Tr"],
        r=p["r"], beta=np.repeat(p["beta"], n_xi, axis=0), eis=p["eis"])
    ss.update(e=e, xi=xi)
    return ss
