from typing import Callable, Dict

import numpy as np
import scipy.optimize as optim

from gradient_descent import grad_descent
from zoo import Zoo


def multistart(f: Callable,
               jac: Callable,
               domain: np.array,
               max_iterations: int = 200,
               tau: float = 1e-4,
               rho: float = 0.8,
               eps: float = 1e-3,
               tol: float = 1e-7,
               ls_method: Callable = optim.minimize,
               ls_kwargs: Dict = dict(),
               polish: bool = True,
               polish_method: Callable = optim.minimize,
               polish_kwargs: Dict = dict()):
    """

    Globally minimizes a function f on region S by repeatly running LS from uniformly sampled
    points in S.

    Args:
        f:                      The function to mininmize
        jac:                    Gradient of f
        domain:                 Search region S on which to minimize f
        max_iterations:         Maximum number of global iterations
        tau:                    Stopping tolerance of LS method
        rho:                    Double-box rule search parameter
        eps:                    Epsilon below which to consider two local minima as the same
        polish:                 Improve final solution by one last LS
        tol:                    Tolerance of polish

    Returns:
        np.array, float, int    Solution x, f(x), and global iterations n

    """

    if 'tol' not in ls_kwargs:
        ls_kwargs['tol'] = tau
    if 'tol' not in polish_kwargs:
        polish_kwargs['tol'] = tol

    dims = domain.shape[1]
    s_min = domain[0]
    s_max = domain[1]

    s_size = s_max - s_min
    s2_min = s_min - 0.5 * (np.power(2., 1. / dims) - 1.) * s_size
    s2_max = s_max + 0.5 * (np.power(2., 1. / dims) - 1.) * s_size
    s2_size = s2_max - s2_min

    m_n = 0

    x_minima = list()
    deltas = list()

    result = optim.OptimizeResult()
    result.nfev = 0
    result.njev = 0

    for n in range(1, max_iterations):

        # Generate points in S2 and discard ones not in S.
        while True:
            x = np.random.uniform(size=dims) * s2_size + s2_min
            in_s = all([s_min[i] <= x[i] and x[i] <= s_max[i] for i in range(dims)])
            m_n += 1
            if in_s:
                break

        # Perform LS with generated initial point in S.
#        x_ls=grad_descent(f, x, grad, tol = tau, max_iterations = 400)
        ls_result = ls_method(f, x0=x, jac=jac, **ls_kwargs)
        x_ls = ls_result.x
        result.nfev += ls_result.nfev
        result.njev += ls_result.njev

        f_ls = f(x_ls)
        result.nfev += 1

        # LS results which escape S are discared.
        if any([x_ls[i] < s_min[i] or x_ls[i] > s_max[i] for i in range(dims)]):
            continue

        delta = n / m_n
        deltas.append(delta)

        sigma_2 = np.var(deltas)

        x_best = None
        f_best = float('inf')

        if f_ls < f_best:
            f_best = f_ls
            x_best = x_ls

            result.x = x_best
            result.fun = f_best
            result.jac = ls_result.jac

        min_is_new = True
        for i, x_min_other in enumerate(x_minima):
            if np.linalg.norm(x_ls - x_min_other) <= eps:
                min_is_new = False
                break

        if min_is_new:
            sigma_2_last = sigma_2
            x_minima.append(x_ls)

        # Otherwise, no new local minimum was found.
        else:

            # Check double-box stop rule.
            if sigma_2 < rho * sigma_2_last:
                break

    if polish:
        polish_result = polish_method(f, x0=x_best, jac=jac, **polish_kwargs)
        result.nfev += polish_result.nfev
        result.njev += polish_result.njev
        result.fun = polish_result.fun
        result.jac = polish_result.jac
        result.x = polish_result.x
#        x=grad_descent(f, x_best, tol = tol, max_iterations = 400)

    return result
