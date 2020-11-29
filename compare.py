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


objectives = [Zoo().get('BR').make_explicit(), ]

TOL = 1e-7

NUM_RUNS = 20
TOL_SAME = 1e-5

MS_MAX_ITER = 10000
MS_TAU = 1e-5
MS_RHO = 0.8
MS_EPS = 1e-4

SA_L0 = 20
SA_DELTA = 1.1
SA_STOP_EPS = 1e-4
SA_CHI = 0.9
SA_GAMMA = 1e-2
SA_DESC_AFFINITY = 0.5

results = dict()

for obj_idx, obj in enumerate(objectives):

    print(f'Benchmark {obj_idx+1} of {len(objectives)}: {obj.name_short}-{obj.name}')

    def f(x):
        global f_calls
        f_calls += 1
        return obj.f(x)

    def grad(x):
        global grad_calls
        grad_calls += 1
        return obj.grad(x)

    # Do Basin-hopping benchmark.
    print(f'  Method BH (Basin-hopping)')
    x_solutions = list()
    for idx_run in range(NUM_RUNS):

        f_calls = 0
        grad_calls = 0
        x0 = sa._gen_point_a(np.array(obj.domain))
        res = optim.basinhopping(f, x0, minimizer_kwargs={'tol': TOL})
        x = res.x
        x_solutions.append(x)
        print('  Run {} of {}: x={}, #f evals={}, #grad evals={}'.format(
            idx_run + 1, NUM_RUNS, x, f_calls, grad_calls))
    print()

    # Do Differential Evolution benchmark.
    print(f'  Method DE (Differential Evolution)')
    bounds = optim.Bounds(lb=obj.domain[0], ub=obj.domain[1])
    x_solutions = list()
    for idx_run in range(NUM_RUNS):

        f_calls = 0
        grad_calls = 0
        res = optim.differential_evolution(f, bounds=bounds, tol=TOL)
        x = res.x
        x_solutions.append(x)
        print('  Run {} of {}: x={}, #f evals={}, #grad evals={}'.format(
            idx_run + 1, NUM_RUNS, x, f_calls, grad_calls))
    print()

    # Do Multi-start benchmark.
    print(f'  Method MS (Multi-start)')
    x_solutions = list()
    for idx_run in range(NUM_RUNS):

        f_calls = 0
        grad_calls = 0
        x, f_best, n = multistart(f,
                                  grad,
                                  np.array(obj.domain),
                                  MS_MAX_ITER,
                                  tol=TOL,
                                  rho=MS_RHO,
                                  eps=MS_EPS)

#        x = res.x
        x_solutions.append(x)
        print('  Run {} of {}: x={}, #f evals={}, #grad evals={}'.format(
            idx_run + 1, NUM_RUNS, x, f_calls, grad_calls))
    print()

    # Do Simulated Annealing benchmark.
    print(f'  Method SA (Simulated Annealing)')
    x_solutions = list()
    for idx_run in range(NUM_RUNS):

        f_calls = 0
        grad_calls = 0
        x, n = simulated_annealing(f,
                                   grad,
                                   np.array(obj.domain),
                                   l0=SA_L0,
                                   delta=SA_DELTA,
                                   stop_eps=SA_STOP_EPS,
                                   chi=SA_CHI,
                                   gamma=SA_GAMMA,
                                   descent_affinity=SA_DESC_AFFINITY,
                                   polish=True,
                                   tol=TOL)

#        x = res.x
        x_solutions.append(x)
        print('  Run {} of {}: x={}, #f evals={}, #grad evals={}'.format(
            idx_run + 1, NUM_RUNS, x, f_calls, grad_calls))
    print()
