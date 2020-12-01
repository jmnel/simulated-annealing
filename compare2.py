from pprint import pprint
from time import perf_counter
import json

import matplotlib
# matplotlib.use('module://matplotlib-backend-kitty')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.optimize as optim

from zoo import Zoo
import sa
from sa import simulated_annealing
from multistart import multistart

TOL = 1e-7
TOL_SAME = 1e-4

NUM_RUNS = 20
TOL_SAME = 1e-5

MS_MAX_ITER = 100
MS_TAU = 1e-4
MS_RHO = 0.8
MS_EPS = 1e-4

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
]


def runner_bh(f, x0, obj, jac):
    return optim.basinhopping(f, x0, minimizer_kwargs={'tol': TOL})


def runner_de(f, x0, obj, jac):
    bounds = optim.Bounds(lb=obj.domain[0], ub=obj.domain[1])
    results = optim.differential_evolution(f, bounds=bounds, tol=TOL)
    results.njev = 0
    return results


def runner_ms(f, x0, obj, jac):
    result = multistart(f,
                        jac,
                        np.array(obj.domain),
                        MS_MAX_ITER,
                        tau=MS_TAU,
                        rho=MS_RHO,
                        eps=MS_EPS,
                        polish=True,
                        tol=TOL)
    return result


def runner_sa(f, x0, obj, jac):
    result = simulated_annealing(f,
                                 jac,
                                 np.array(obj.domain),
                                 l0=SA_L0,
                                 delta=SA_DELTA,
                                 stop_eps=SA_STOP_EPS,
                                 chi=SA_CHI,
                                 gamma=SA_GAMMA,
                                 t=SA_T,
                                 polish=True,
                                 polish_kwargs={'tol': TOL})
    return result


methods = [
    {'name': 'BH',
     'name_long': 'Basin-hopping',
     'runner': runner_bh},
    {'name': 'DE',
     'name_long': 'Differential Evolution',
     'runner': runner_de},
    {'name': 'MS',
        'name_long': 'Multi-start',
        'runner': runner_ms},
    {'name': 'SA',
        'name_long': 'Simulated Annealing',
        'runner': runner_sa}
]

results = dict()

for idx, obj in enumerate(objectives):

    print(f'Benchmark {idx+1} of {len(objectives)}: {obj.name_short} ({obj.name})')
    f, jac = obj.f, obj.grad

    results[obj.name_short] = {
        'name': obj.name,
        'w': len(obj.xmin),
        'methods': dict()}

    for method in methods:
        name = method['name']
        name_long = method['name_long']
        runner = method['runner']

        print(f'  Method {name} ({name_long})')

        num_runs_list = list()
        nfev_list = list()
        njev_list = list()
        times_list = list()

        for jdx in range(NUM_RUNS):

            s = np.array(obj.domain)

            x_solutions = list()

            num_runs = 0
            nfev = 0
            njev = 0
            t_delta = 0.
            while len(x_solutions) < len(obj.xmin):
                x0 = np.random.uniform(size=obj.dims) * (s[1] - s[0]) + s[0]
                t_start = perf_counter()

                result = runner(f, x0, obj, jac)

                t_delta += perf_counter() - t_start

                nfev += result.nfev
                njev += result.njev

                is_min = False
                for i in range(len(obj.xmin)):
                    if np.linalg.norm(result.x - obj.xmin[i]) <= TOL_SAME:
                        is_min = True
                        break

                is_new = True
                for i in range(len(x_solutions)):
                    if np.linalg.norm(result.x - x_solutions[i]) <= TOL_SAME:
                        is_new = False
                        break

                if is_new and is_min:
                    x_solutions.append(result.x)
                num_runs += 1

            num_runs_list.append(num_runs)
            nfev_list.append(nfev)
            njev_list.append(njev)
            times_list.append(t_delta)

            print('  Run {} of {}: runs={}, nfev={}, njev={}, time={}'.format(
                jdx + 1, NUM_RUNS, num_runs, nfev, njev, np.round(t_delta, 4)))
        print('  ' + '-' * 40)
        print('  avg_runs={}, avg_nfev={}, avg_njev={}, avg_time={}'.format(
            np.mean(num_runs_list), np.mean(nfev_list), np.mean(njev_list), np.round(np.mean(times_list), 4)))
        metric_1 = np.mean(num_runs_list) / len(obj.xmin)
        metric_2 = np.mean(nfev_list + njev_list) / len(obj.xmin)
        print('  metric_1={}, metric_2={}'.format(np.round(metric_1, 2), np.round(metric_2, 2)))
        print()

        results[obj.name_short]['methods'][name] = {
            'name_long': name_long,
            'num_runs': num_runs_list,
            'nfev': nfev_list,
            'njev': njev_list,
            'times': times_list,
            'metric_1': metric_1,
            'metric_2': metric_2}

pprint(results)

with open('compare.json', 'wt') as fi:
    json.dump(results, fi)


fig, ax = plt.subplots(1, 1)

for bench_name, vals in results.items():
    w = vals['w']
    for idx, (method, props) in enumerate(vals['methods'].items()):
        evals = (np.array(props['nfev']) + np.array(props['njev'])) / w
        times = np.array(props['times']) / w

        if bench_name == 'RB':
            ax.scatter(times, evals, c=f'C{idx}', s=8, label=f'{method} ({props["name_long"]})')
        else:
            ax.scatter(times, evals, c=f'C{idx}', s=8)

ax.legend()
ax.set_xlabel(r'$\Delta t$ - Time (s)')
ax.set_ylabel(r'$n(f_{ij}) + n(\nabla_{ij}) / w$ - Function evals. per minima')
# plt.show()

fig.savefig('figures/compare1-1.pdf', dpi=200)

fig, ax = plt.subplots(1, 1)
ax.set_xscale('log')
ax.set_yscale('log')

for bench_name, vals in results.items():
    w = vals['w']
    for idx, (method, props) in enumerate(vals['methods'].items()):
        evals = (np.array(props['nfev']) + np.array(props['njev'])) / w
        times = np.array(props['times']) / w

        if bench_name == 'RB':
            ax.scatter(times, evals, c=f'C{idx}', s=8, label=f'{method} ({props["name_long"]})')
        else:
            ax.scatter(times, evals, c=f'C{idx}', s=8)

ax.legend()
ax.set_xlabel(r'$\Delta t$ - Time (s)')
ax.set_ylabel(r'$n(f_{ij}) + n(\nabla_{ij}) / w$ - Function evals. per minima')
# plt.show()

fig.savefig('figures/compare1-2.pdf', dpi=200)

for bench_name, vals in results.items():
    print(f'{bench_name} ', end='')
    w = vals['w']
    for idx, (method, props) in enumerate(vals['methods'].items()):
        avg_nfev = np.round(np.mean(props['nfev']), 1)
        avg_njev = np.round(np.mean(props['njev']), 1)
        avg_time = np.round(np.mean(props['times']) * 1e3, 0)
        metric_1 = np.round(props['metric_1'], 2)
        metric_2 = np.round(props['metric_2'], 0)

        if idx == 0:
            print(' & ', end='')
        else:
            print(' & ', end='')

        row = '{} & {} & {} & {:.0f} & {} & {} \\\\'.format(
            method, avg_nfev, avg_njev, avg_time, metric_1, metric_2)
        print(row)
    print('\\hline')
