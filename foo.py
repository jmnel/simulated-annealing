from pprint import pprint
from typing import Callable

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np


def rosenbrock(x, y):
    a, b = 1, 100
    return (a - x)**2 + b * (y - x**2)**2


# z = f_rosenbrock(mx, my)
# plt.contourf(x, y, z, levels=100, cmap='viridis')


def gen_point(dom):
    #    pprint(dom)
    return np.random.random(2) * (dom[:, 1] - dom[:, 0]) + dom[:, 0]
#    return np.random.random(2)

# def local_search(f,


# def f(x, y):
#    return -x**2 + -y**2

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

    f_delta = [f(*gen_point_a(dom)) - f(*gen_point_a(dom)) for _ in range(m_trials)]
    f_delta_plus = [e for e in f_delta if e > 0]
    f_delta_plus = np.array(f_delta_plus)

#    c = 2000
#    plt.hist(np.exp(-f_delta_plus / c), bins=200)
#    plt.show()
#    m_2 = len(f_delta_plus)

    cutoff = np.quantile(f_delta_plus, acceptance_ratio)
    c0 = -cutoff / np.log(acceptance_ratio)

    return c0


def update_schedule(


dom=np.array([[-2, -1], [2, 3]])
f=rosenbrock
# n_samples = 200
# xlim = [-2, 2]
# ylim = [-1, 3]
# x, y = np.linspace(*xlim, n_samples), np.linspace(*ylim, n_samples)
# mx, my = np.meshgrid(x, y)
# z = np.power(rosenbrock(mx, my), 0.1)

init_schedule(f, dom, acceptance_ratio=0.9, m_trials=100000)
# sa(f)
