from pprint import pprint

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.optimize as optim

from zoo import Zoo
import sa
from sa import simulated_annealing
from multistart import multistart


#objectives = [Zoo().get('BR').make_explicit(), ]

obj = Zoo().get('BR').make_explicit()

TOL = 1e-7

NUM_RUNS = 20
TOL_SAME = 1e-5

SA_L0 = 20
SA_DELTA = 1.1
SA_STOP_EPS = 1e-6
SA_CHI = 0.9
SA_GAMMA = 1e-2
SA_T = 0.5

results = dict()


def f(x):
    global f_calls
    f_calls += 1
    return obj.f(x)


def grad(x):
    global grad_calls
    grad_calls += 1
    return obj.grad(x)


for idx in range(NUM_RUNS):

    f_calls = 0
    grad_calls = 0
    result = simulated_annealing(f,
                                 grad,
                                 np.array(obj.domain),
                                 l0=SA_L0,
                                 delta=SA_DELTA,
                                 stop_eps=SA_STOP_EPS,
                                 chi=SA_CHI,
                                 gamma=SA_GAMMA,
                                 t=SA_T,
                                 polish=True,
                                 tol=TOL)

    row = '{} & ( {:.5f}, {:.5f} ) & {:.5f} & {} & {} \\\\'
    row = row.format(idx + 1, *result.x, result.fun, result.nfev, result.njev)
    print(row)
