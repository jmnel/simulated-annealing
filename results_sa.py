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

NUM_RUNS = 50
TOL_SAME = 1e-4

SA_L0 = 20
SA_DELTA = 1.1
SA_STOP_EPS = 1e-4
SA_CHI = 0.9
SA_GAMMA = 1e-2
SA_T = 0.5


objectives = [
    Zoo().get('RB').make_explicit(),
    Zoo().get('GP').make_explicit(),
    Zoo().get('BR').make_explicit(),
    Zoo().get('H3').make_explicit(),
    Zoo().get('H6').make_explicit(),
    #        Zoo().get('S', m=5).make_explicit(),
    #    Zoo().get('S', m=7).make_explicit(),
    #    Zoo().get('S', m=10).make_explicit(),
]

rows = list()
minima_results = dict()

for idx, obj in enumerate(objectives):

    print(f'benchmark {idx+1} of {len(objectives)}: {obj.name}')

    f, grad = obj.f, obj.grad

    s = obj.domain
    s_min, s_max = s[0], s[1]
    dims = obj.dims

    xmin_true = obj.xmin

    num_sol = len(xmin_true)

    ks = list()

    local_minima = list()

    nfev = 0
    njev = 0

    for jdx in range(NUM_RUNS):

        sol_found = list()
        k = 0

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

            nfev += result.nfev
            njev += result.njev

            if any([result.x[i] < s_min[i] or result.x[i] > s_max[i] for i in range(dims)]):
                k += 1
                continue

            is_min = False
            for j, xmin_i in enumerate(xmin_true):
                dist = np.linalg.norm(xmin_i - result.x)
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

            if not is_min:
                is_new_local = True
                for j, xmin_local in enumerate(local_minima):
                    if np.linalg.norm(xmin_local - result.x) <= TOL_SAME:
                        is_new_local = False
                        break
                if is_new_local:
                    local_minima.append(result.x)

            k += 1

        ks.append(k)
    nfev_avg = nfev / float(NUM_RUNS)
    njev_avg = njev / float(NUM_RUNS)

    print(f'avg num runs to find all x*: {np.mean(ks)}')
    print(f'avg nfev: {np.round(nfev_avg,1)}, avg njev: {np.round(njev_avg,1)}')
    print(f'solutions x* are:')
    for jdx, sol in enumerate(sol_found):
        print(f'  {jdx+1} -> {sol}')

    print()

    row = '{} & {} & {} & {} & {} & {}'
    rows.append(row.format(jdx + 1, obj.name_short, np.mean(ks), len(sol_found), nfev_avg, njev_avg))

    f_at_glob = [f(x) for x in sol_found]
    print(f'f at glob: {f_at_glob}')
    f_at_loc = [f(x) for x in local_minima]
    minima_results[obj.name_short] = [sol_found, f_at_glob, local_minima, f_at_loc]
    print(f'added {obj.name_short}')

print()
print('-' * 40)

for row in rows:
    print(row)

print('-' * 40)
print()

# print(minima_results.keys())
# exit()

for name, values in minima_results.items():

    print(f'{name} & g & ', end='')
    for i, x in enumerate(values[0]):
        if i > 0:
            print('& & ', end='')

        x_str = ', '.join(['{}', ] * len(x))
        x_str = '( ' + x_str + ' )'
        x_str = x_str.format(* [np.round(x_i, 5) for x_i in x])
        print(f'{x_str} & {np.round(values[1][i], 5)} \\\\')

    for i, x in enumerate(values[2]):
        if i == 0:
            print('& l &', end='')
        else:
            print('& & ', end='')

        x_str = ', '.join(['{}', ] * len(x))
        x_str = '( ' + x_str + ' )'
        x_str = x_str.format(* [np.round(x_i, 5) for x_i in x])
        print(f'{x_str} & {np.round(values[3][i], 5)} \\\\')
    print('\\hline')
