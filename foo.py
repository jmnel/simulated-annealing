from pprint import pprint
from typing import Callable

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np


def rosenbrock(x):
    x1, x2 = x
    a, b = 1, 100
    return (a - x1)**2 + b * (x2 - x1**2)**2


def g_rosenbrock(x):
    a, b = 1, 100
    return np.array([-2. * (a - x1) - 2. * b * x1 * (x2 - x1**2),
                     b * 2. * (x2 - x1**2)])


def gen_point(dom):
    return np.random.random(2) * (dom[:, 1] - dom[:, 0]) + dom[:, 0]


def gen_point_a(dom: np.ndarray):
    #    p = np.random.uniform(size=2) * dom[:, 1] - dom[:, 0] + dom[:, 0]
    p = np.random.uniform(size=2) * (dom[1] - dom[0]) + dom[0]
    return p


def init_schedule(f: Callable,
                  dom: np.ndarray,
                  acceptance_ratio: float,
                  m_trials: int):

    #    f_plus = list()
    chi = acceptance_ratio = 0.9

    f_delta = [f(gen_point_a(dom)) - f(gen_point_a(dom)) for _ in range(m_trials)]
    f_delta_plus = [e for e in f_delta if e > 0]
    f_delta_plus = np.array(f_delta_plus)

    cutoff = np.quantile(f_delta_plus, acceptance_ratio)
    c0 = -cutoff / np.log(acceptance_ratio)

    return c0


hist = list()

L0 = 10.
DELTA = 0.1
EPS = 1e-4
CHI = 0.9


def fmin(f, dom):

    c0 = init_schedule(f, dom, 0.9, 100)
    L = 20
    c = c0

    for n in range(10000):

        x = gen_point_a(dom)
        hist_inner = [x, ]

        q = list()
        for i in range(L):
            y = gen_point_a(dom)
            q.append(f(y))
            if f(y) - f(x) <= 0.:
                hist_inner.append(y)
                x = y
            elif np.exp(-f(y) - f(x) / c) > np.random.random():
                hist_inner.append(y)
                x = y

        sigma = np.std(q)
        c = c / (1. + (c * np.log(1. + DELTA)) / (3. * sigma))

        if len(hist_inner) > 0:
            hist.append(hist_inner)

    return x


dom = np.array([[-2, -1], [2, 3]])
f = rosenbrock

res = fmin(f, dom)
print(f'x={res}')
n_samples = 200
xlim = [-2, 2]
ylim = [-1, 3]
x, y = np.linspace(*xlim, n_samples), np.linspace(*ylim, n_samples)
mx, my = np.meshgrid(x, y)
z = np.power(rosenbrock([mx, my]), 0.1)

fig, ax = plt.subplots(1, 1)
ax.contourf(x, y, z)

h_x = [h[-1][0] for h in hist]
h_y = [h[-1][1] for h in hist]
h_x = h_x[-1:]
h_y = h_y[-1:]
ax.scatter(h_x, h_y, c='red')

plt.show()

# init_schedule(f, dom, acceptance_ratio=0.9, m_trials=100000)
# sa(f)
