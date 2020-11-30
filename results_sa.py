from pprint import pprint

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import numpy as np

from zoo import Zoo
from sa import simulated_annealing

TOL = 1e-7

NUM_RUNS = 5
TOL_SAME = 1e-6

SA_L0 = 20
SA_DELTA = 1.1
SA_STOP_EPS = 1e-4
SA_CHI = 0.9
SA_GAMMA = 1e-2
SA_T = 0.5


objectives = [
    #    Zoo().get('RB').make_explicit(),
    #    Zoo().get('GP').make_explicit(),
    #    Zoo().get('BR').make_explicit(),
    #    Zoo().get('H3').make_explicit(),
    Zoo().get('H6').make_explicit(),
    #    Zoo().get('S', m=5).make_explicit(),
    #    Zoo().get('S', m=7).make_explicit(),
    #    Zoo().get('S', m=10).make_explicit(),
]


for idx, obj in enumerate(objectives):

    print(f'benchmark {idx+1} of {len(objectives)}: {obj.name}')

    f, grad = obj.f, obj.grad

    s = obj.domain
    dims = obj.dims

    xmin_true = obj.xmin

    num_sol = len(xmin_true)

    ks = list()

    for jdx in range(NUM_RUNS):

        sol_found = list()
#        print(f'run -> {jdx}')
        k = 0
#        traj_hist = list()

#        def cb(iteration, x, chain, c):
#            traj.append(x)

        while len(sol_found) < num_sol:

            traj = list()
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
                                         tol=TOL,
                                         callback=None)

#            print(f'x={result.x}, f={result.fun}')

            is_min = False
            for j, xmin_i in enumerate(xmin_true):
                dist = np.linalg.norm(xmin_i - result.x)
                print(f'd1 ={dist}')
                if np.linalg.norm(xmin_i - result.x) <= TOL_SAME:
                    is_min = True
                    break

            is_new = True
            for j, x_found_i in enumerate(sol_found):
                if np.linalg.norm(x_found_i - result.x) <= TOL_SAME:
                    is_new = False
                    break

            if is_new and is_min:
                sol_found.append(result.x)
                print(f'{jdx} {k} -> found new {result.x}')
#                traj_hist.append(traj)

#            if not is_min:
#                print(result.x)

            k += 1

        ks.append(k)

    print(f'avg num runs: {np.mean(ks)}')
