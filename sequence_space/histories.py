"""Second- and third-order size and history effects from successive surprise shocks.

As in notebook 2, restart the household calculation when a new shock arrives,
carrying forward the distribution left by earlier shocks.

J_lu is the LU factorization of the steady-state asset-income Jacobian
J[t,s] = dA[t]/dZ[s], holding interest rates and transfers fixed. In GE, solve
J @ dZ = -dA to offset the asset response and clear the market. At second and
third order, these income derivatives also equal output and consumption derivatives.
Without J_lu, return the PE asset derivatives.
"""

import numpy as np
from scipy.linalg import lu_solve

from .paths import first_order, second_order, third_order


def second_histories(backward, forward, u, L, *, J_lu=None):
    """Return second-order size and history effects for two successive shocks.

    Output entry [t,l] is the derivative in shocks t and t+l periods ago.
    u gives the first-order input path (r,Z,Tr) after either shock.
    Return PE asset derivatives, or GE income derivatives when J_lu is supplied.
    """
    T = len(u)
    # Allow T response dates after a second shock arriving up to L-1 dates later.
    # The first shock's input path ends at T, but its distribution keeps evolving.
    single2 = second_order(backward, forward, shifted(u, 0, T+L-1))
    direct2 = np.empty((T, L))
    # Let x arrive l dates before y. Along x=y=nu, A'' = A_xx+2*A_xy+A_yy.
    # Until y arrives, households do not anticipate it. Restart the calculation
    # with the distribution after l periods and the remaining input path from x.
    for l in range(L):
        initial = (single2['dF'][l], single2['d2F'][l])
        joint = second_order(backward, forward, shifted(u, l, T)+u, initial_F=initial)
        # Subtract A_xx and A_yy to isolate A_xy. The second-order income
        # responses to x and y separately would cancel in this subtraction.
        direct2[:, l] = (joint['d2A']-single2['d2A'][l:l+T]-single2['d2A'][:T])/2
    # The one-shot principle gives J @ Z_xy = -A_xy in GE.
    return direct2 if J_lu is None else lu_solve(J_lu, -direct2)


def third_histories(backward, forward, u, L, *, d2Z=None, J_lu=None,
                    workers=1, only_moments=False):
    """Return cubic[t,j,k] for shocks t,t+j,t+k periods ago, and Va_xyy[l].

    u[t] is the first-order input response (r,Z,Tr) to one innovation. In GE,
    d2Z[t,l] is the second-order income response from second_histories.
    cubic contains PE asset derivatives or GE income derivatives with J_lu.

    Va_xyy[l] is marginal value when y arrives, differentiated once in the
    earlier shock x and twice in y, with l periods between them. It is used
    to calculate how x changes the effect of future risk.
    only_moments skips derivatives involving three distinct shock dates.
    """
    # Three shocks x, y, z arrive in that order. g is the gap from x to y,
    # and h the gap from y to z. Measured from z's arrival, the shock dates
    # are -(g+h), -h, 0, so the result belongs in cubic[t,g+h,h].
    T = len(u)
    d2u_pairs = np.zeros((T, L, 3))
    # PE inputs are linear in shock size. GE also needs the second-order
    # income responses, including the size effect at lag zero.
    if d2Z is not None:
        d2u_pairs[:, :, 1] = d2Z

    # As in notebook 2, supply the first- and second-order input responses.
    # Third-order income responses involving only one or two shocks cancel
    # when we isolate A_xyz. Its GE income correction is solved afterward.
    d2u_own = d2u_pairs[:, 0]  # Second-order input response to one innovation.
    single = third_order(backward, forward, shifted(u, 0, T+L-1),
                         shifted(d2u_own, 0, T+L-1))
    # Calculate the two-shock asset responses to subtract from the three-shock
    # response below. Save only assets here to avoid storing L distribution paths.
    d3A_pairs = []
    for l in range(L):
        length = T+L-1-l
        du = shifted(u, l, length)+shifted(u, 0, length)
        # Varying both shocks together gives u_xx + 2*u_xy + u_yy at second order.
        d2u = (shifted(d2u_own, l, length)+shifted(d2u_own, 0, length)
               +2*shifted(d2u_pairs[:, l], 0, length))
        response = third_order(backward, forward, du, d2u,
                               initial_F=inherited_F(single, l))
        d3A_pairs.append(response['d3A'])

    # Moments need g=0 or h=0, when at least two shock dates coincide.
    # For each g, calculate all required h using the same two-shock distribution.
    tasks = []
    for g in range(L):
        hs = (0,) if only_moments and g > 0 else tuple(range(L-g))
        tasks.append((g, hs))
    cubic = np.zeros((T, L, L))
    Va_xyy = np.empty((L,)+backward.shape)
    inputs = (backward, forward, J_lu, u, d2u_pairs, single, d3A_pairs)
    for g, hs, values, Va in third_order_results(tasks, inputs, workers):
        for h, value in zip(hs, values):
            cubic[:, g+h, h] = cubic[:, h, g+h] = value
        Va_xyy[g] = Va
    return cubic, Va_xyy


def third_history_block(g, hs, backward, forward, J_lu, u, d2u_pairs, single, d3A_pairs):
    """Return third derivatives in x,y,z arriving at -(g+h), -h, 0, for fixed g.

    Return one response path per h and Va_xyy at the newer shock's date (h=0).
    Shocks arriving at the same date add together.
    """
    T = len(u)
    d2u_own = d2u_pairs[:, 0]  # Second-order input response to one innovation.
    # Start at y's arrival with the distribution left by x. Follow the response
    # to x and y far enough to use its distribution at each possible arrival of z,
    # and its asset path for T dates afterward.
    length = T+max(hs)
    du = shifted(u, g, length)+shifted(u, 0, length)
    d2u = (shifted(d2u_own, g, length)+shifted(d2u_own, 0, length)
           +2*shifted(d2u_pairs[:, g], 0, length))
    pair = third_order(backward, forward, du, d2u, initial_F=inherited_F(single, g))

    # At z's arrival, restart with the distribution left by x and y.
    # Along x=y=z=nu, the second input derivative is
    # u_xx + u_yy + u_zz + 2*(u_xy + u_xz + u_yz).
    values = []
    for h in hs:
        j = g+h
        du = shifted(u, j, T)+shifted(u, h, T)+u
        d2u = (shifted(d2u_own, j, T)+shifted(d2u_own, h, T)+d2u_own
               +2*(shifted(d2u_pairs[:, g], h, T)+d2u_pairs[:, j]+d2u_pairs[:, h]))
        joint = third_order(backward, forward, du, d2u, initial_F=inherited_F(pair, h))
        # Compare all asset responses over the same dates, starting when z arrives.
        d3A_xy = pair['d3A'][h:h+T]
        d3A_xz, d3A_yz = d3A_pairs[j][:T], d3A_pairs[h][:T]
        d3A_x, d3A_y, d3A_z = single['d3A'][j:j+T], single['d3A'][h:h+T], single['d3A'][:T]
        # D_xyz - D_xy - D_xz - D_yz + D_x + D_y + D_z = 6*A_xyz,
        # where D denotes the third derivative as the included shocks vary with nu.
        direct = (joint['d3A']-d3A_xy-d3A_xz-d3A_yz+d3A_x+d3A_y+d3A_z)/6
        # Solve J @ Z_xyz = -A_xyz in GE.
        value = direct if J_lu is None else lu_solve(J_lu, -direct)
        values.append(value)

        # With y and z coincident, also obtain Va_xyy at their arrival date.
        if h == 0:
            # Marginal value depends on current and future inputs, not the asset distribution.
            old = third_order(backward, forward, shifted(u, g, T), shifted(d2u_own, g, T))
            # joint, pair, old, single vary (x,y) as (nu,2nu), (nu,nu), (nu,0), (0,nu).
            # Their third derivatives with weights 1,-2,1,-6 give 6*Va_xyy.
            direct_Va = (joint['d3Va0']-2*pair['d3Va0']+old['d3Va0']
                         -6*single['d3Va0'])/6
            Va = direct_Va
            if J_lu is not None:
                # GE also changes marginal value through the new income correction.
                dVa = np.zeros(backward.shape)
                for t in range(T-1, -1, -1):
                    dVa, _ = backward.step_1st(dVa, np.array([0., value[t], 0.]))
                Va = Va+dVa
    return values, Va


# Input paths and distributions when the next shock arrives.

def shifted(path, gap, length):
    """Drop gap elapsed dates and pad or truncate the path to length.

    For example, shifted([a,b,c], 1, 4) gives [b,c,0,0].
    """
    result = np.zeros((length,)+path.shape[1:])
    active = min(length, len(path)-gap)
    result[:active] = path[gap:gap+active]
    return result


def inherited_F(response, date):
    """Return the first three derivatives of the distribution entering date."""
    return tuple(response[key][date] for key in ('dF', 'd2F', 'd3F'))


# Standalone GE path solve, also used to check order-by-order market clearing.

def ge_derivatives(backward, forward, debt, J_lu, initial_F=None):
    """Return household and income-path derivatives through third order.

    debt is the first-order debt path, with higher debt derivatives equal to zero.
    initial_F contains the first three derivatives of the initial asset CDF.
    J_lu is defined above using the same horizon as debt.
    """
    if initial_F is None:
        initial_F = np.zeros((3,)+forward.shape)
    u1, u2, u3 = np.zeros((3, len(debt), 3))

    # At first order, income must make asset demand match debt supply.
    direct1 = first_order(backward, forward, u1, initial_F=initial_F[0])
    u1[:, 1] = lu_solve(J_lu, debt-direct1['dA'])

    # At each higher order, offset the asset response to the lower-order inputs.
    direct2 = second_order(backward, forward, u1, initial_F=initial_F[:2])
    u2[:, 1] = lu_solve(J_lu, -direct2['d2A'])

    result = third_order(backward, forward, u1, u2, initial_F=initial_F)
    u3[:, 1] = lu_solve(J_lu, -result['d3A'])

    # Add the household response to the third-order income correction.
    correction = first_order(backward, forward, u3)
    for name in ('A', 'F', 'Va0'):
        result['d3'+name] += correction['d'+name]
    result.update(dZ=u1[:, 1], d2Z=u2[:, 1], d3Z=u3[:, 1])
    return result


# Optional multiprocessing: the same fixed-g calculation as the serial loop.

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context


def third_order_results(tasks, inputs, workers):
    """Run third_history_block for each (g, hs), serially or in separate processes."""
    if workers == 1:
        for g, hs in tasks:
            values, Va = third_history_block(g, hs, *inputs)
            yield g, hs, values, Va
        return

    # Send the large derivative operators and paths once to each process.
    with ProcessPoolExecutor(workers, mp_context=get_context('spawn'),
                             initializer=prepare_worker, initargs=inputs) as pool:
        for (g, hs), (values, Va) in zip(tasks, pool.map(worker_history_block, tasks)):
            yield g, hs, values, Va


def prepare_worker(*args):
    """Send fixed operators and lower-order paths once per process, not per task."""
    global worker_inputs
    worker_inputs = args


def worker_history_block(task):
    """Calculate the third-order effects for one gap g and its associated h values."""
    g, hs = task
    return third_history_block(g, hs, *worker_inputs)
