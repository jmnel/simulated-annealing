from typing import Callable

import numpy as np


def armijo_step(phi: Callable,
                d_phi: Callable,
                l0: float,
                alpha: float,
                rho: float):
    """
    Calculates the maximum Armijo step size such that the Goldstein condition is still satisfied.

    Args:
        phi         Function objective value along search direction.
        d_phi:      Derivative of phi with respect to t.
        l0:         Initial base step size.
        alpha:      Armijo parameters.
        rho:        Growth factor.

    Returns:
        float       Armijo max step size.

    """

    k0 = 0
    l = l0 * np.power(rho, k0)
    for k in range(k0, 100):

        l_new = l0 * np.power(rho, k)
        if phi(l_new) > phi(0) + alpha * l_new * d_phi(0):
            return l

        l = l_new

    return l


def gss(phi: Callable, a: float, b: float, tol: float = 1e-12):
    """
    Find minimum of function with Golden-section search.

    Args:
        phi:    Function to minimize.
        a:      Left bracket of search interval.
        b:      Right bracket of search interval.
        tol:    Desired tolerance.

    Returns:
        float   Minimizer of phi.

    """

    INV_PHI = 0.5 * (np.sqrt(5.) - 1.)
    INV_PHI_2 = 0.5 * (3. - np.sqrt(5.))

    a, b = min(a, b), max(a, b)
    h = b - a

    if h <= tol:
        return 0.5 * (a + b)

    n = int(np.ceil(np.log(tol / h) / np.log(INV_PHI)))

    c = a + INV_PHI_2 * h
    d = a + INV_PHI * h
    fc = phi(c)
    fd = phi(d)

    for k in range(n - 1):
        if fc < fd:
            b = d
            d = c
            fd = fc
            h = INV_PHI * h
            c = a + INV_PHI_2 * h
            fc = phi(c)
        else:
            a = c
            c = d
            fc = fd
            h = INV_PHI * h
            d = a + INV_PHI * h
            fd = phi(d)

    if fc < fd:
        return 0.5 * (a + d)

    else:
        return 0.5 * (c + b)
