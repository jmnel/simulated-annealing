from typing import Callable, Dict
from collections import namedtuple

from scipy.stats import chi2
import numpy as np
import scipy.optimize as optim

from gradient_descent import grad_descent
import zoo


def _gen_point_a(domain: np.ndarray):
    p = np.random.uniform(size=2) * (domain[1] - domain[0]) + domain[0]
    return p


def _gen_point_b(f: Callable,
                 x0: np.ndarray,
                 domain: np.ndarray,
                 t: float,
                 jac: Callable,
                 dd_method: Callable,
                 dd_kwargs: Dict):

    w = np.random.random()
    if w > t:
        p = np.random.uniform(size=2) * (domain[1] - domain[0]) + domain[0]

    else:
        p = dd_method(f=f,
                      x0=x0,
                      jac=jac,
                      max_iterations=1,
                      **dd_kwargs)
        p = np.clip(p, domain[0], domain[1])

    return p


def _init_schedule(f: Callable,
                   jac: Callable,
                   domain: np.ndarray,
                   acceptance_ratio: float,
                   m_trials: int,
                   t: float,
                   dd_method: Callable,
                   dd_kwargs: Dict):

    chi = acceptance_ratio

    chain = [_gen_point_a(domain), ]
    for m in range(m_trials - 1):
        p = _gen_point_b(f=f,
                         x0=chain[-1],
                         domain=domain,
                         t=t,
                         jac=jac,
                         dd_method=dd_method,
                         dd_kwargs=dd_kwargs)
        chain.append(p)

    f_vals = np.array(list(map(f, chain)))
    f_delta = np.diff(f_vals).tolist()
    f_delta_plus = np.array([e for e in f_delta if e > 0.])

    print(f'init cnt = {len(f_delta_plus)}')

    cutoff = np.quantile(f_delta_plus, acceptance_ratio)
    c0 = -cutoff / np.log(acceptance_ratio)

    return c0


def simulated_annealing(f: Callable,
                        jac: Callable,
                        domain: np.ndarray,
                        l0: int = 20,
                        delta: float = 1.5,
                        stop_eps: float = 1e-4,
                        chi: float = 0.9,
                        gamma: float = 0.01,
                        t: float = 0.5,
                        callback: Callable = None,
                        dd_method: Callable = grad_descent,
                        dd_kwargs: Dict = dict(),
                        polish: bool = True,
                        polish_minimizer: Callable = optim.minimize,
                        polish_minimizer_kwargs: Dict = {'tol': 1e-7}):

    l = l0 * domain[0].ndim

    # Initialize temperature schedule.
    c0 = _init_schedule(f=f,
                        jac=jac,
                        domain=domain,
                        acceptance_ratio=chi,
                        m_trials=200,
                        t=t,
                        dd_method=dd_method,
                        dd_kwargs=dd_kwargs)
    c = c0

    # Generate starting point.
    x = _gen_point_a(domain)

    f_record = None

    x_smooth = x

    n = 0
    while True:

        #        print(f'n={n}')

        if f_record is None:
            f_record = f(x)

        m_1 = 0
        m_2 = 0

        f_at_c = list()
        f_at_c = [f(x), ]

        current_chain = list()
#        while len(f_at_c) < 1:
#            print(f'inner')
        for i in range(l):
            y = _gen_point_b(domain, t, x, f, jac, method='exact')

            # Accept new point if it is downhill.
            if f(y) - f(x) <= 0.:
                x = y
                m_1 += 1
                f_at_c.append(f(x))
                current_chain.append(x)

            # Otherwise, use acceptance criteria.
            elif np.exp(-(f(y) - f(x)) / c) > np.random.random():
                x = y
                m_2 += 1
                f_at_c.append(f(x))
                current_chain.append(x)

        if callback is not None:
            callback(iteration=n, x=x, chain=current_chain, c=c)

        f_bar = np.mean(f_at_c)
        sigma = np.std(f_at_c)
        if sigma == 0.:
            sigma = 1e-14

        if n > 0:
            f_bar_s_old = f_bar_s
            f_bar_s = (1.0 - gamma) * f_bar_s + gamma * f_bar
        else:
            f_bar_s = f_bar
            f_bar_0 = np.mean(f_at_c)

        c_old = c
        c = c * (1. + (c * np.log(1. + delta)) / (3. * sigma))**-1

        y = f_record
        f_star = 0

        dc = c - c_old

        if n > 0 and dc != 0.0:
            d_fbar_s = f_bar_s - f_bar_s_old

            stop_term = np.abs((d_fbar_s / dc) * (c / f_bar_0))
            should_stop = stop_term < stop_eps

            if should_stop:
                break

        n += 1

    if polish:
        x = jac_descent(f, x, jac, max_iterations=400, ls_method='method')

    return x, n


obj = zoo.Zoo().get('BR').make_explicit()

f, grad = obj.f, obj.grad


res = simulated_annealing(f,
                          grad,
                          domain=np.array(obj.domain),
                          l0=20,
                          delta=1.5,
                          stop_eps=1e-4,
                          chi=0.9,
                          gamma=0.01,
                          t=0.5,
                          dd_method=grad_descent,
                          dd_kwargs=dict(),
                          polish=True,
                          polish_minimizer=optim.minimize,
                          polish_minimizer_kwargs={'tol': 1e-7})
