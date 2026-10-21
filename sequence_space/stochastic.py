"""Simulation, moments, and GIRFs for a series X.

The derivative arrays follow the paper's notation:
X1[j] = X_j, X2[j,l] = X_{j,j+l}, and X3[j,l,lp] = X_{j,j+l,j+lp}.
X3 has shape (T,L,L): j counts periods since the most recent shock, while l and
lp are the lags from that shock to the earlier shocks.
X_risk is the risky-steady-state derivative X_{sigma,sigma}.
X_sigma_sigma is the risk-transmission derivative X_{sigma,sigma,j}.
"""

import numpy as np
from numba import njit
from scipy.signal import oaconvolve


def simulate(epsbar, X1, X2=None, X3=None, X_risk=0., X_sigma_sigma=None, burnin=0):
    """Return the coefficients x in X_t = X_ss + sigma*x[0,t] + sigma^2*x[1,t] + ...

    Output rows give coefficients on successive powers of sigma. Columns are dates,
    after dropping the first burnin periods. epsbar[t] = epsilon_t/sigma.
    Shocks before the simulation starts are zero.
    A burn-in of at least T+L-2 periods removes dependence on this initial history.
    Supplied X2 and X3 must include all size and history effects.
    """
    order = 3 if X3 is not None else (2 if X2 is not None else 1)
    N = len(epsbar)
    x = np.zeros((order, N))

    # Convolution sums X1[j]*epsbar[t-j] over j: each past shock contributes
    # its first-order response to X at date t.
    x[0] = oaconvolve(epsbar, X1)[:N]
    if order == 1:
        return x[:, burnin:]

    # Anticipated risk shifts the level of X and the response to each shock.
    L = X2.shape[1]
    x[1] = X_risk/2
    if order == 3 and X_sigma_sigma is not None:
        x[2] = oaconvolve(epsbar, X_sigma_sigma)[:N]/2

    # Apply the same convolution to products of two shocks,
    # pair[t] = epsbar[t]*epsbar[t-l], and then three shocks at lags 0, l, m.
    for l in range(min(L, N)):
        pair = np.zeros(N)
        pair[l:] = epsbar[l:]*epsbar[:N-l]
        x[1] += oaconvolve(pair, X2[:, l])[:N]/(2 if l == 0 else 1)
        if order == 3:
            for m in range(l+1):
                triple = np.zeros(N)
                triple[m:] = pair[m:]*epsbar[:N-m]
                # We sum each combination of shock dates once. In the full triple
                # sum it appears 1, 3, or 6 times when all dates coincide, two coincide,
                # or all differ. Dividing by 3! gives weights 1/6, 1/2, or 1.
                weight = 1/6 if l == 0 else (1/2 if m == 0 or m == l else 1)
                x[2] += weight*oaconvolve(triple, X3[:, l, m])[:N]
    return x[:, burnin:]


@njit(cache=True)
def simulate_simple(epsbar, X1, X2=None, X3=None, X_risk=0., X_sigma_sigma=None, burnin=0):
    """Version of simulate using direct sums instead of convolution.

    Inputs and returned coefficients follow the same conventions as simulate.
    """
    order = 3 if X3 is not None else (2 if X2 is not None else 1)
    T = len(X1)
    x = np.zeros((order, len(epsbar)))
    # j, k, kp count periods back from t. The smallest index is the most recent shock.
    # We include T dates since that shock and lags below L to the earlier shocks.
    for t in range(len(epsbar)):
        # First-order response and the two anticipated-risk terms.
        for j in range(min(T, t+1)):
            x[0, t] += X1[j]*epsbar[t-j]
        if order >= 2:
            x[1, t] = X_risk/2
        if order == 3 and X_sigma_sigma is not None:
            for j in range(min(T, t+1)):
                x[2, t] += X_sigma_sigma[j]*epsbar[t-j]/2

        # (1/2) sum_{j,k} X_{jk} epsbar[t-j] epsbar[t-k].
        # X2[j_min,l] gives the derivative for shocks j_min and j_min+l periods ago.
        if X2 is not None:
            L = X2.shape[1]
            # Stop at the sample start or the oldest included shock, T+L-2 periods ago.
            n = min(T+L-1, t+1)
            for j in range(n):
                for k in range(n):
                    j_min = min(j, k)
                    l = abs(j-k)
                    if j_min < T and l < L:
                        x[1, t] += X2[j_min, l]*epsbar[t-j]*epsbar[t-k]/2

        # (1/6) sum_{j,k,k'} X_{jkk'} epsbar[t-j] epsbar[t-k] epsbar[t-k'].
        # Look up the derivative with indices in increasing order, but sum every ordering.
        if X3 is not None:
            L = X3.shape[1]
            n = min(T+L-1, t+1)
            for j in range(n):
                for k in range(n):
                    # History effects for shocks this far apart are omitted.
                    if abs(j-k) >= L:
                        continue
                    for kp in range(n):
                        j_min, j_max = min(j, k, kp), max(j, k, kp)
                        j_mid = j+k+kp-j_min-j_max
                        # Both lags are from the most recent shock:
                        # X3[j_min,l,lp] = X_{j_min,j_mid,j_max}.
                        l, lp = j_mid-j_min, j_max-j_min
                        if j_min < T and lp < L:
                            x[2, t] += X3[j_min, l, lp]*epsbar[t-j]*epsbar[t-k]*epsbar[t-kp]/6
    return x[:, burnin:]


def mean_correction(X2_own=None, X_risk=0.):
    """Return mean2 in E[X] = X_ss + sigma^2*mean2, using X2_own[j] = X_{j,j}.

    Assumes independent mean-zero innovations with variance sigma^2.
    """
    return (X_risk + (X2_own.sum() if X2_own is not None else 0.))/2


def covariance_coefficients(X1, X2=None, X3=None, *, X_sigma_sigma=None, lags=(0, 1)):
    """Return Sigma2, Sigma4 in Cov(X_t,X_{t-h}) = sigma^2*Sigma2 + sigma^4*Sigma4.

    Each output array follows the order of lags, with h=0 giving variance.
    Without X3, return only the leading covariance (Sigma4=0). The sigma^4
    correction uses only entries of X3 with at least two shock dates equal.
    Assumes iid mean-zero innovations with variance sigma^2 and Gaussian third and fourth moments.
    """
    T = len(X1)
    Sigma2, Sigma4 = np.zeros((2, len(lags)))
    if X3 is not None:
        L = X2.shape[1]
        # X3_sum stores S_i = sum_j X_ijj, summing over the repeated shock's date.
        X3_sum = np.zeros(T+L-1)
        X3_sum[:T] = X3[:, 0, 0]
        for l in range(1, L):
            X3_sum[:T] += X3[:, l, l]      # Repeated shock l periods before the single shock.
            X3_sum[l:l+T] += X3[:, l, 0]   # Repeated shock l periods after the single shock.

    for k, h in enumerate(lags):
        n = max(T-h, 0)
        Sigma2[k] = X1[h:] @ X1[:n]
        if X3 is not None:
            # For X = X_ss + sigma*x1 + sigma^2*x2 + sigma^3*x3, Sigma4 adds
            # Cov(x2_t, x2_{t-h}) and the two linear-cubic cross terms.
            quadratic = X2[h:, 0] @ X2[:n, 0]/2 + np.sum(X2[h:, 1:]*X2[:n, 1:])
            risk = 0.
            if X_sigma_sigma is not None:
                risk = (X1[h:] @ X_sigma_sigma[:n]+X_sigma_sigma[h:] @ X1[:n])/2
            # The cubic contribution is (1/2)*sum_i (X_{h+i}*S_i + X_i*S_{h+i}).
            # The two dot products shift the first-order and cubic terms in turn.
            m = max(min(T, len(X3_sum)-h), 0)
            cubic = (X1[h:] @ X3_sum[:n]+X3_sum[h:h+m] @ X1[:m])/2
            Sigma4[k] = quadratic + risk + cubic
    return Sigma2, Sigma4


def girf(eps, nu, sigma, X1, X2=None, X3=None, *, X_sigma_sigma=None, horizon=20):
    """Return E_t[X_{t+j}] - E_{t-1}[X_{t+j}] after observing today's shock nu.

    Output columns are horizons j=0,...,horizon-1. Output rows give approximations
    through first, second, and third order, each including the lower-order terms.
    eps contains past innovations in levels, oldest to newest, with eps=sigma*epsbar.
    nu is today's innovation.
    """
    T = len(X1)
    order = 3 if X3 is not None else (2 if X2 is not None else 1)
    if horizon > T:
        raise ValueError("GIRF horizon must not exceed T")
    IRF = np.zeros((order, horizon))
    IRF[0] = X1[:horizon]*nu
    if order == 1:
        return IRF

    L = X2.shape[1]
    # Put today's shock first, then yesterday's, and so on: e[l] is l periods ago.
    e = np.concatenate(([nu], eps[::-1][:L-1]))
    K = len(e)
    # Observing nu replaces the prior expectation E[epsilon_t^2]=sigma^2 by nu^2.
    square_revision = nu**2-sigma**2
    for j in range(horizon):
        # Today's squared shock, then today's shock times each known past shock.
        IRF[1, j] = (IRF[0, j]+X2[j, 0]*square_revision/2
                      +sum(X2[j, l]*e[l] for l in range(1, K))*nu)
    if order == 2:
        return IRF

    for j in range(horizon):
        # At horizon j, X3[i,j-i,0] involves today's shock once and the shock
        # at future date j-i twice. Its expected square is sigma^2.
        future_pairs = sum(X3[i, j-i, 0] for i in range(max(0, j-L+1), j))
        transmission = X_sigma_sigma[j] if X_sigma_sigma is not None else 0.
        risk = (transmission+future_pairs)*sigma**2*nu/2
        own = X3[j, 0, 0]*nu**3/6

        # The change in today's squared shock multiplies each known past shock.
        today_twice = sum(X3[j, l, 0]*e[l] for l in range(1, K))*square_revision/2
        # Today's shock also multiplies each past shock squared.
        past_twice = sum(X3[j, l, l]*e[l]**2 for l in range(1, K))*nu/2
        # Today's shock interacts with two distinct past shocks.
        past_pair = sum(X3[j, l, m]*e[l]*e[m] for l in range(1, K) for m in range(1, l))*nu
        IRF[2, j] = IRF[1, j] + risk + own + today_twice + past_twice + past_pair
    return IRF
