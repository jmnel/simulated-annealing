from pprint import pprint
from typing import Callable

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import chi2
import numpy as np

from gradient_descent import grad_descent


def _gen_point_a(dom: np.ndarray):
    p = np.random.uniform(size=2) * (dom[1] - dom[0]) + dom[0]
    return p


def _gen_point_b(dom: np.ndarray,
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
        p = np.clip(p, dom[0], dom[1])

    return p


def _init_schedule(f: Callable,
                   dom: np.ndarray,
                   acceptance_ratio: float,
                   m_trials: int,
                   method='alternative_b',
                   **kwargs):

    chi = acceptance_ratio = 0.9

    if method == 'alternative_b':
        chain = [_gen_point_a(dom), ]
        for m in range(m_trials - 1):
            chain.append(_gen_point_b(dom,
                                      kwargs['t'],
                                      chain[-1],
                                      f,
                                      method='exact'))

        f_vals = np.array(list(map(f, chain)))
        f_delta = np.diff(f_vals).tolist()
        f_delta_plus = np.array([e for e in f_delta if e > 0.])
        cutoff = np.quantile(f_delta_plus, acceptance_ratio)
        c0 = -cutoff / np.log(acceptance_ratio)

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

    c0 = _init_schedule(f, dom, chi, 100, 'alternative_b', t=t)
    c = c0
    print(f'c0={c0}')

    x = _gen_point_a(dom)
    f_record = None

    hist_new = list()

    n = 0
    while True:

        if f_record is None:
            f_record = f(x)

        hist_inner = [x, ]

        f_at_c = [f(x), ]

        m_1 = 0
        m_2 = 0

        for i in range(L):
            #            y = _gen_point_a(dom)
            y = _gen_point_b(dom, T, x, f, method='exact')

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

            hist_new.append(x)
            if len(hist_new) > 1000:
                hist_new = hist_new[-1000:]

            f_at_c.append(f(x))

            if f_at_c[-1] < f_record:
                f_record = f_at_c[-1]

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
            c_old = c
            try:
                c = c / (1. + (c * np.log(1. + DELTA)) / (3. * sigma))

        y = f_record
        f_star = 0

        dc = c - c_old

        if n > 0:
            d_fbar_s = f_bar_s - f_bar_s_old

            stop_term = np.abs((c * d_fbar_s) / (dc * f_bar_0))

            should_stop = stop_term < EPS

#            should_stop = c < 1e-24

            if n % 10 == 0 or should_stop:
                print(f'{n} -> c={c}')
                n_samples = 200
#                xlim = [-5, 5]
#                ylim = [-5, 7]
#                xlim = [-2, 3]
#                ylim = [-2, 3]
#                xlim = [-5, 10]
#                ylim = [-0, 15]
                xlim = [-3, 3]
                ylim = [-3, 3]
                x_plt, y_plt = np.linspace(*xlim, n_samples), np.linspace(*ylim, n_samples)
                mx, my = np.meshgrid(x_plt, y_plt)
                z = np.power(f([mx, my]), 0.1)

                fig, ax = plt.subplots(1, 1)
                ax.contourf(x_plt, y_plt, z, levels=50)

                ax.scatter([x[0], ], [x[1], ], c='red')

                ax.plot(*tuple(zip(*hist_new)), color='magenta')

                plt.show()

            if should_stop:
                break

        n += 1

    return x


from zoo import Zoo

obj = Zoo().get('goldstein_price').make_explicit()
#obj = Zoo().get('branin').make_explicit()
#obj = Zoo().get('rosenbrock').make_explicit()
f, grad = obj.f, obj.grad
dom, plt_dom = obj.domain, obj.domain_plot
dom = np.array(dom)

# pprint(dom)
# exit()

DOM_DIM = 2
L0 = 100
L = DOM_DIM * L0
DELTA = 0.01
EPS = 1e-17
CHI = 0.9
#SMOOTHING = 0.01
SMOOTHING = 0.05
#T = 0.0
T = 0.75

# print(obj.xmin)
# print(f(obj.xmin[0]))
# exit()

res = fmin(f,
           dom=dom,
           l0=L0,
           delta=DELTA,
           eps=EPS,
           chi=CHI,
           smoothing=SMOOTHING,
           t=T)

print(f'solution x={res}')
