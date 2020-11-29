from typing import Callable

from scipy.stats import chi2
import numpy as np

from gradient_descent import grad_descent


def _gen_point_a(domain: np.ndarray):
    p = np.random.uniform(size=2) * (domain[1] - domain[0]) + domain[0]
    return p


def _gen_point_b(domain: np.ndarray,
                 descent_affinity: float,
                 x0: np.ndarray,
                 f: Callable,
                 method='exact',
                 **kwargs):

    w = np.random.random()
    if w > descent_affinity:
        p = np.random.uniform(size=2) * (domain[1] - domain[0]) + domain[0]

    else:
        p = grad_descent(f, x0, max_iterations=1, ls_method=method, **kwargs)
        p = np.clip(p, domain[0], domain[1])

    return p


def _init_schedule(f: Callable,
                   domain: np.ndarray,
                   acceptance_ratio: float,
                   m_trials: int,
                   method='alternative_b',
                   **kwargs):

    chi = acceptance_ratio = 0.9

    if method == 'alternative_b':
        chain = [_gen_point_a(domain), ]
        for m in range(m_trials - 1):
            chain.append(_gen_point_b(domain,
                                      kwargs['descent_affinity'],
                                      chain[-1],
                                      f,
                                      method='exact'))

        f_vals = np.array(list(map(f, chain)))
        f_delta = np.diff(f_vals).tolist()
        f_delta_plus = np.array([e for e in f_delta if e > 0.])
        cutoff = np.quantile(f_delta_plus, acceptance_ratio)
        c0 = -cutoff / np.log(acceptance_ratio)

    return c0


def _init_schedule_2(f: Callable,
                     domain: np.ndarray,
                     acceptance_ratio: float,
                     m_trials: int,
                     method='alternative_b',
                     **kwargs):

    chi = 0.9

    x = [_gen_point_a(domain), ]
    for m in range(m_trials - 1):
        x.append(_gen_point_b(domain,
                              kwargs['descent_affinity'],
                              x[-1],
                              f,
                              method='exact'))

    f_vals = np.array(list(map(f, x)))
    f_delta = np.diff(f_vals).tolist()
    f_delta_plus = [fd for fd in f_delta if fd > 0.]
    m_2 = len(f_delta_plus)
    m_1 = m_trials - m_2
    f_delta_plus_avg = np.mean(f_delta_plus)

    m_1, m_2 = m_2, m_1

    c0 = f_delta_plus_avg / (np.log(m_2 / (m_2 * chi + (1. - chi) * m_1)))

    return c0


def simulated_annealing(f: Callable,
                        grad: Callable,
                        domain: np.ndarray,
                        l0: int,
                        delta: float,
                        stop_eps: float,
                        chi: float,
                        gamma: float,
                        descent_affinity: float,
                        callback: Callable = None):

    l = l0 * domain[0].ndim

    # Initialize temperature schedule.
    c0 = _init_schedule_2(f, domain, chi, 100, 'alternative_b', descent_affinity=descent_affinity)
    c = c0

    # Generate starting point.
    x = _gen_point_a(domain)

    f_record = None

    x_smooth = x

    n = 0
    while True:

        if f_record is None:
            f_record = f(x)

        m_1 = 0
        m_2 = 0

        f_at_c = list()

        current_chain = list()
        while len(f_at_c) < 10:
            for i in range(l):
                y = _gen_point_b(domain, descent_affinity, x, f, method='exact')

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

    return x, n
