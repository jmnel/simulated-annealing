import numpy as np

from step_size import armijo_step, gss
from utils import grad_approx


def grad_descent(f,
                 x0,
                 grad=None,
                 tol=1e-14,
                 max_iterations=10000,
                 ls_method='exact',
                 eps=1e-14,
                 **kwargs):
    """
    Perform unconstrained minimization of f by gradient descent.

    Args:
        f:                  Function to minimize.
        x0:                 Initial position.
        grad:               Gradient of f.
        tol:                Desired tolerance.
        max_iterations:     Maximum number of iterations.
        ls_method:          Line search method; either 'exact' or 'armijo'.

    Returns:
        Dict with solution and various information.

    """

    x = x0

    if grad is None:
        def grad(x): return grad_approx(f, x)

    # Do main algorithm loop.
    for i in range(max_iterations):

        g = grad(x)

        # Exit if gradient is close to 0. We are likely at a local minimum.
        if np.linalg.norm(g) <= tol:
            return x

        # If requested, peform Armijo step size selection.
        if ls_method == 'armijo':
            t = armijo_step(lambda t: f(x - t * g),
                            lambda t: np.dot(-grad(x - t * g), g),
                            kwargs['l0'], kwargs['alpha'], kwargs['rho'])

        # Otherwise, use exact line search.
        elif ls_method == 'exact':
            t = gss(lambda t: f(x - t * g), 0., 1.)

        x = x - t * g

    return x
