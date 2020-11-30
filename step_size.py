from typing import Callable

import numpy as np
import scipy.optimize as optim


def armijo_step(f: Callable,
                l0: float,
                jac: Callable,
                alpha: float,
                rho: float):
    """
    Calculates the maximum Armijo step size such that the Goldstein condition is still satisfied.

    Args:
        f                 Function objective value along search direction.
        jac:              Derivative of f with respect to t.
        l0:                 Initial base step size.
        alpha:              Armijo parameters.
        rho:                Growth factor.

    Returns:
        OptimizeResult       Armijo max step size.

    """

    k0 = 0
    l = l0 * np.power(rho, k0)
    f0 = f(0.)
    jac0 = jac(0.)
    for k in range(k0, 100):

        l_new = l0 * np.power(rho, k)
        if f(l_new) > f0 + alpha * l_new * jac0:
            result = optim.OptimizeResult(x=l,
                                          success=True,
                                          status=0,
                                          message='found optimal step size',
                                          nfev=1 + k,
                                          njev=1,
                                          nit=k)
            return result

        l = l_new

    result = optim.OptimizeResult(x=l,
                                  success=False,
                                  status=-1,
                                  message='max iterations exceeded',
                                  nfev=100 + 1,
                                  njev=1,
                                  nit=k)
    return result


def gss(f: Callable, a: float, b: float, tol: float = 1e-12):
    """
    Find minimum of function with Golden-section search.

    Args:
        f:    Function to minimize.
        a:      Left bracket of search interval.
        b:      Right bracket of search interval.
        tol:    Desired tolerance.

    Returns:
        float   Minimizer of f.

    """

    INV_PHI = 0.5 * (np.sqrt(5.) - 1.)
    INV_PHI_2 = 0.5 * (3. - np.sqrt(5.))

    a, b = min(a, b), max(a, b)
    h = b - a

    if h <= tol:
        return optim.OptimizeResult(x=0.5 * (a + b),
                                    success=True,
                                    status=0,
                                    message='found optimal value',
                                    nfev=0,
                                    njev=0,
                                    nit=0)
#        return 0.5 * (a + b)

    n = int(np.ceil(np.log(tol / h) / np.log(INV_PHI)))

    nfev = 0

    c = a + INV_PHI_2 * h
    d = a + INV_PHI * h
    fc = f(c)
    fd = f(d)

    nfev += 2

    for k in range(n - 1):
        if fc < fd:
            b = d
            d = c
            fd = fc
            h = INV_PHI * h
            c = a + INV_PHI_2 * h
            fc = f(c)
            nfev += 1
        else:
            a = c
            c = d
            fc = fd
            h = INV_PHI * h
            d = a + INV_PHI * h
            fd = f(d)
            nfev += 1

    if fc < fd:
        assert (d - a) <= tol
#        x = 0.5 * (a + d)

    else:
        assert (b - c) <= tol
#        x = 0.5 * (c + b)

    return optim.OptimizeResult(x=0.5 * (a + d) if fc < fd else 0.5 * (c + b),
                                success=True,
                                status=0,
                                message='found optimal value',
                                nfev=nfev,
                                njev=0,
                                nit=n)
