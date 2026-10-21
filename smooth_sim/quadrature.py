"""Reusable Gauss-Legendre rules mapped from [-1, 1] to finite intervals."""

from numpy.polynomial import legendre
from numba import njit

def start(n):
    """Return n nodes and their integration weights on [-1, 1] for reuse."""
    return legendre.leggauss(n)

@njit
def interval(S, a, b):
    """Map the rule S to [a, b], returning integration weights, then nodes."""
    z, wnorm = S
    x = (b-a)/2*(z+1) + a
    w = (b-a)/2*wnorm
    return w, x
