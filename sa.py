from typing import Callable, Dict

from scipy.stats import chi2
import numpy as np
import scipy.optimize as optim

import zoo
from gradient_descent import grad_descent


def simulated_annealing(f: Callable,
                        jac: Callable,
                        domain: np.ndarray,
                        l0: int,
                        delta: float,
                        stop_eps: float,
                        chi: float,
                        gamma: float,
                        t: float,
                        init_trials: int = 200,
                        callback: Callable = None,
                        tol: float = 1e-7,
                        polish: bool = True,
                        polish_minimizer: Callable = optim.minimize,
                        polish_kwargs: Dict = dict()):
    """

    Globally minimize f on bounded region using Simulated Annealing.

    Args:
        f:                  Function to minimize.
        jac:                Jacobian of f.
        domain:             Region on which to minimize f.
        l0:                 Basic length of Markov chains.
        delta:              Cooling schedule decrement rate.
        stop_eps:           Stop condition control parameter.
        chi:                Initial acceptance parameter.
        gamma:              Smoothing parameter.
        t:                  Local search affinity.
        init_trials:        Number of initial transitions to initialize schedule.
        callback:           Called on each iteration of global phase.
        tol:                Desire final stopping tolerance of solution.
        polish:             Refine final solution by running some local search method.
        polish_minimizer:   Minimization function to use for polishing.
        polish_kwargs:      Keyword arguments to pass to polish minimizer.

    Returns:
        scipy.optimize.OptimizeResult

    """

    MIN_TRANSITIONS = 10

    dims = domain.shape[1]

    if 'tol' not in polish_kwargs:
        polish_kwargs['tol'] = tol

    def gen_point_a():
        p = np.random.uniform(size=dims) * (domain[1] - domain[0]) + domain[0]
        return p

    def gen_point_b(x0: np.ndarray):

        w = np.random.random()
        if w > t:
            p = np.random.uniform(size=dims) * (domain[1] - domain[0]) + domain[0]

        else:
            dd_result = grad_descent(f, x0, jac, max_iterations=1)

            # Update f and grad f evaluation count stats.
            result.nfev += dd_result.nfev
            result.njev += dd_result.njev

            p = dd_result.x
            p = np.clip(p, domain[0], domain[1])

        return p

    def init_schedule():

        # Generate starting point for initial Markov chain.
        chain = [gen_point_a(), ]

        # Proceed to generate example Markov chain.
        for m in range(init_trials - 1):
            chain.append(gen_point_b(x0=chain[-1]))

        # Evaluate f at each point in chain and take 1-lag difference.
        f_vals = np.array(list(map(f, chain)))
        f_delta = np.diff(f_vals).tolist()

        # Update f evaluation count stats.
        result.nfev += len(chain)

        # Filter for only positive delta values.
        f_delta_plus = np.array([e for e in f_delta if e > 0.])

        # Use chi to calculate cuffoff value of delta.
        cutoff = np.quantile(f_delta_plus, chi)

        # Solve for C0.
        c0 = -cutoff / np.log(chi)

        return c0

    # ----------------------------------

    # Create object to hold results and running stats.
    result = optim.OptimizeResult()
    result.nfev = 0
    result.njev = 0

    # Set Markov chain length.
    l = l0 * domain[0].ndim

    # Initialize cooling schedule.
    c0 = init_schedule()
    c = c0

    # Generate starting point.
    x = gen_point_a()

    x_smooth = x

    n = 0
    while True:

        m_1 = 0
        m_2 = 0

        f_at_c = list()
        f_at_c = [f(x), ]

        current_chain = list()

        # Repeat until some minimum number of transitions are made.
        while len(f_at_c) < MIN_TRANSITIONS:
            for i in range(l):

                # Generate next candidiate in Markov chain.
                y = gen_point_b(x0=x)

                # Evaluate f at x and y; this is wasteful and can be improved.
                f_at_x = f(x)
                f_at_y = f(y)
                delta_f = f_at_y - f_at_x
                result.nfev += 2

                # Accept new point if it is downhill.
                if f_at_y - f_at_x <= 0.:
                    x = y
                    m_1 += 1
                    f_at_c.append(f_at_y)
                    current_chain.append(x)

                # Otherwise, use acceptance criteria.
                elif np.exp(-delta_f / c) > np.random.random():
                    x = y
                    m_2 += 1
                    f_at_c.append(f_at_y)
                    current_chain.append(x)

        if callback is not None:
            callback(iteration=n, x=x, chain=current_chain, c=c)

        f_bar = np.mean(f_at_c)
        sigma = np.std(f_at_c)

        # Prevent sigma from being 0 to avoid divide-by-zero error.
        if sigma == 0.:
            sigma = 1e-14

        # Initialize f_bar_0 and smoothed f_bar if at first iteration.
        if n == 0:
            f_bar_s = f_bar
            f_bar_0 = np.mean(f_at_c)

        # Otherwise, update smoothed f_bar and track previous value for finite difference.
        else:
            f_bar_s_old = f_bar_s
            f_bar_s = (1.0 - gamma) * f_bar_s + gamma * f_bar

        # Decrement C; record prevous C for finite difference.
        c_old = c
        c = c * (1. + (c * np.log(1. + delta)) / (3. * sigma))**-1

        dc = c - c_old

        # Begin checking stop condition after first iteration.
        if n > 0 and dc != 0.0:

            # Calculate differene of smoothed f bar.
            d_fbar_s = f_bar_s - f_bar_s_old

            # Evaluate stop condition.
            stop_term = np.abs((d_fbar_s / dc) * (c / f_bar_0))
            should_stop = stop_term < stop_eps

            # Break out of outer loop when stop condition is reached.
            if should_stop:
                break

        n += 1

    # If requested, improve solution by running a local search starting at solution.
    if polish:
        result_polish = polish_minimizer(f, x0=x, jac=jac, **polish_kwargs)
        result.nfev += result_polish.nfev
        result.njev += result_polish.njev
        result.fun = result_polish.fun
        result.jac = result_polish.jac
        result.x = result_polish.x

    return result
