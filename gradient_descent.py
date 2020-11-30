from typing import Callable

import numpy as np
import scipy.optimize as optim

from step_size import armijo_step, gss
from utils import grad_approx


# def _defult_ls(f, x0, jac, ls_kwargs):
#    return armijo_step(f, x0, jac, ls_kwargs)

def _defult_ls(f, x0, jac, ls_kwargs):
    return gss(f, x0, **ls_kwargs)


def grad_descent(f: Callable,
                 x0: np.ndarray,
                 jac: Callable,
                 tol: float = 1e-14,
                 max_iterations=200,
                 ls_method: Callable = _defult_ls,
                 ls_kwargs={'a': 0.0, 'b': 1.0}):

    x = x0

    nfev = 0
    njev = 0

    # Do main algorithm loop.
    for i in range(max_iterations):

        g = jac(x)
        njev += 1

        # Exit if jacient is close to 0. We are likely at a local minimum.
        if np.linalg.norm(g) <= tol:
            return optim.OptimizeResult(x=x,
                                        success=True,
                                        status=0,
                                        message='found optimal value',
                                        fun=f(x),
                                        jac=g,
                                        nfev=nfev,
                                        njev=njev,
                                        nit=i)

            ls_result = ls_method(f=lambda t: f(x - t * g),
                                  jac=lambda t: np.dot(-jac(x - t * g), g),
                                  **ls_kwargs)

            t = ls_result.x
            nfev += ls_result.nfev
            njev += ls_result.njev

            x = x - t * g

    return optim.OptimizeResult(x=x,
                                success=False,
                                status=-1,
                                message='max iteratios exceeded',
                                fun=f(x),
                                jac=g,
                                nfev=nfev,
                                njev=njev,
                                nit=max_iterations)
