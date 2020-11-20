from pprint import pprint
from typing import Callable

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

from gradient_descent import grad_descent


# def rosenbrock(x):
#    x1, x2 = x
#    a, b = 1, 100
#    return (a - x1)**2 + b * (x2 - x1**2)**2


# def g_rosenbrock(x):
#    a, b = 1, 100
#    return np.array([-2. * (a - x1) - 2. * b * x1 * (x2 - x1**2),
#                     b * 2. * (x2 - x1**2)])


# def goldstein_price(x):
#    x1, x2 = x
#    return ((1. + (x1 + x2 + 1)**2 *
#             (19. - 14. * x1 + 3. * x1**2 - 14. * x2 + 6. * x1 * x2 + 3. * x2**2))
#            * (30. + (2. * x1 - 3. * x2)**2
#                * (18. - 32. * x1 + 12. * x1**2 + 48. * x2 - 36. * x1 * x2 + 27. * x2**2)))


def gen_point_a(dom: np.ndarray):
    p = np.random.uniform(size=2) * (dom[1] - dom[0]) + dom[0]
    return p


def gen_point_b(dom: np.ndarray,
                t: float,
                x0: np.ndarray,
                f: Callable,
                method='exact',
                **kwargs):

    w = np.random.random()
    if w > t:
        p = np.random.uniform(size=2) * (dom[1] - dom[0]) + dom[0]
    else:
        p = grad_descent(f, x0, max_iterations=1, ls_method=method, **kwargs)

    return p


def init_schedule(f: Callable,
                  dom: np.ndarray,
                  acceptance_ratio: float,
                  m_trials: int,
                  method='alternative_b',
                  **kwargs):

    chi = acceptance_ratio = 0.9

    if method == 'alternative_b':
        chain = [gen_point_a(dom), ]
        for m in range(m_trials - 1):
            chain.append(gen_point_b(dom,
                                     kwargs['t'],
                                     chain[-1],
                                     f,
                                     method='exact'))
#            print(f'chain {chain[-1]}')

        f_vals = np.array(list(map(f, chain)))
        f_delta = np.diff(f_vals).tolist()
        f_delta_plus = np.array([e for e in f_delta if e > 0.])
        cutoff = np.quantile(f_delta_plus, acceptance_ratio)
        c0 = -cutoff / np.log(acceptance_ratio)


#    f_delta = [f(gen_point_a(dom)) - f(gen_point_a(dom)) for _ in range(m_trials)]
#    f_delta_plus = [e for e in f_delta if e > 0]
#    f_delta_plus = np.array(f_delta_plus)

#    cutoff = np.quantile(f_delta_plus, acceptance_ratio)
#    c0 = -cutoff / np.log(acceptance_ratio)

    return c0


hist = list()


def fmin(f, dom,
         l0,
         delta,
         eps,
         chi,
         smoothing,
         t):

    l = l0 * dom[0].ndim

    c0 = init_schedule(f, dom, chi, 100, 'alternative_b', t=t)
    c = c0

    x = gen_point_a(dom)

    n = 0
    while True:

        hist_inner = [x, ]

        f_at_c = [f(x), ]

        m_1 = 0
        m_2 = 0

        for i in range(L):
            #            y = gen_point_a(dom)
            y = gen_point_b(dom, T, x, f, method='exact')

            # Accept new point if it is downhill.
            if f(y) - f(x) <= 0.:
                hist_inner.append(y)
                x = y
                m_1 += 1

            # Otherwise, use acceptance criteria.
            elif np.exp(-(f(y) - f(x)) / c) > np.random.random():
                hist_inner.append(y)
                x = y
                m_2 += 1

            else:
                pass

            f_at_c.append(f(x))

#        print(f'{n} -> m1={m_1}, m2={m_2}')
#        print(f'c={c}')

        if len(hist_inner) > 0:
            hist.append(hist_inner)

        f_bar = np.mean(f_at_c)

        if n == 0:
            f_bar_0 = np.mean(f_at_c)

        if n > 0:
            f_bar_s_old = f_bar_s
            f_bar_s = (1.0 - SMOOTHING) * f_bar_s + SMOOTHING * f_bar
        else:
            f_bar_s = f_bar

        if len(f_at_c) > 0:
            sigma = np.std(f_at_c)
#            if sigma == 0.:
#                sigma = 1e-28
            c_old = c
#            print(f'{n} -> c={c}, sigma={sigma}, len of chain: {len(f_at_c)}')
            c = c / (1. + (c * np.log(1. + DELTA)) / (3. * sigma))

        dc = c - c_old

        if n > 0:
            d_fbar_s = f_bar_s - f_bar_s_old

            stop_term = np.abs((c * d_fbar_s) / (dc * f_bar_0))
            should_stop = stop_term < EPS

            if n % 1000 == 0 or should_stop:
                n_samples = 200
#                xlim = [-5, 5]
#                ylim = [-5, 7]
                xlim = [-2, 3]
                ylim = [-2, 3]
                x_plt, y_plt = np.linspace(*xlim, n_samples), np.linspace(*ylim, n_samples)
                mx, my = np.meshgrid(x_plt, y_plt)
#                z = f([mx, my])
#                z = np.power(f([mx, my]), 0.01)
#                z = np.log(f([mx, my]))
                z = np.power(f([mx, my]), 0.1)

                fig, ax = plt.subplots(1, 1)
                ax.contourf(x_plt, y_plt, z)

                ax.scatter([x[0], ], [x[1], ], c='red')

                plt.show()
                print('{} -> stop term={:.4f}, f_bar_s={:.4f}, c={:.4f}'.format(n, stop_term, f_bar_s, c))
                print(f'sigma={sigma}')

            if should_stop:
                print(f'{n} -> stop condition')
                break

        n += 1

    return x


#DOM_DIM = 2
#L0 = 10
#L = DOM_DIM * L0
#DELTA = 0.1
#EPS = 1e-4
#CHI = 0.9
#SMOOTHING = 0.01
#T = 0.75

#DOM = np.array([[-2, -1], [2, 3]])
#f = rosenbrock

#f = goldstein_price
#dom = np.array([[-2, -2], [2, 2]])

# res = fmin(f,
#           dom=dom,
#           l0=L0,
#           delta=DELTA,
#           eps=EPS,
#           chi=CHI,
#           smoothing=SMOOTHING,
#           t=T)
#print(f'solution x={res}')
