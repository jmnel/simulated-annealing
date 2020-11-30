from pprint import pprint
from time import perf_counter

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.optimize as optim
import optuna

from zoo import Zoo
from sa2 import simulated_annealing

OPTUNA_NUM_TRIALS = 2000

objectives = [Zoo().get('BR').make_explicit(), ]

TOL = 1e-5

NUM_RUNS = 10

for idx, obj in enumerate(objectives):

    f, grad = obj.f, obj.grad
    s = np.array(obj.domain)

    def tune_obj(trial: optuna.Trial):

        l0 = trial.suggest_int(name='l0', low=2, high=100, step=1)
        delta = trial.suggest_uniform('delta', low=1.01, high=2.)
        eps_stop = trial.suggest_loguniform(name='eps', low=1e-7, high=1e-3)
        chi = trial.suggest_uniform(name='chi', low=0.5, high=0.99)
        gamma = trial.suggest_loguniform(name='gamma', low=1e-3, high=0.9)
        t = trial.suggest_uniform(name='t', low=0.1, high=0.90)

#        print(f'l0={l0}, delta={delta}, eps={eps_stop}, chi={chi}, gamma={gamma}, t={t}')

        t_elapsed = 0.
        x_list = list()
        for jdx in range(NUM_RUNS):

            t_start = perf_counter()

            x, _ = simulated_annealing(f=f,
                                       grad=grad,
                                       domain=s,
                                       l0=l0,
                                       delta=delta,
                                       stop_eps=eps_stop,
                                       chi=chi,
                                       gamma=gamma,
                                       descent_affinity=t,
                                       polish=False,
                                       )

            t_elapsed += perf_counter() - t_start

            x_list.append(x)
#            print(f'run {jdx}')

        x_minima = [[obj.xmin[i], 0] for i in range(len(obj.xmin))]
        num_wrong = 0

        dist = 0.
#        matched = [False, ] * len(x_minima)
        for x_i in x_list:

            found = False
            for i in range(len(x_minima)):
                if np.linalg.norm(x_i - x_minima[i][0]) < TOL:
                    x_minima[i][1] += 1
                    dist += np.linalg.norm(x_i - x_minima[i][0])
                    found = True
#                    matched[i] = True
                    break

            if not found:
                num_wrong += 1

#        score_1 = np.max([x_minima[i][1] for i in range(len(x_minima))]) \
#            - np.min([x_minima[i][1] for i in range(len(x_minima))])
        score_2 = t_elapsed

        print(f'sc1={num_wrong}')
        print(f'sc2={score_2}')
        print(f'dist={dist}')
        print()

        return 1e2 * dist + num_wrong + score_2

    study = optuna.create_study(study_name='test',
                                storage=f'sqlite:///optuna-test.db',
                                load_if_exists=True)

    study.optimize(tune_obj, n_trials=OPTUNA_NUM_TRIALS)
