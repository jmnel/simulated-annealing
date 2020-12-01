from pprint import pprint

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


#objectives = [Zoo().get('BR').make_explicit(), ]

obj = Zoo().get('BR').make_explicit()

TOL = 1e-7

NUM_RUNS = 20
TOL_SAME = 1e-5

SA_L0 = 20
SA_DELTA = 1.1
SA_STOP_EPS = 1e-7
SA_CHI = 0.9
SA_GAMMA = 1e-2
SA_T = 0.001


def callback(iteration, x, chain, c):
    traj.extend(chain)
    print(c)


traj = list()
result = simulated_annealing(obj.f,
                             obj.grad,
                             np.array(obj.domain),
                             l0=SA_L0,
                             delta=SA_DELTA,
                             stop_eps=SA_STOP_EPS,
                             chi=SA_CHI,
                             gamma=SA_GAMMA,
                             t=SA_T,
                             polish=True,
                             polish_kwargs={'tol': TOL},
                             callback=callback)

fig, ax = plt.subplots()
plt_dom = np.array(obj.domain_plot)
xlim = plt_dom[:, 0]
ylim = plt_dom[:, 1]
n_samples = 200
x_plt, y_plt = np.linspace(*xlim, n_samples), np.linspace(*ylim, n_samples)
mx, my = np.meshgrid(x_plt, y_plt)
z = np.power(obj.f([mx, my]), 0.1)

fig, ax = plt.subplots(1, 1)
ax.contourf(x_plt, y_plt, z, levels=50, cmap='viridis')
ax.scatter(*tuple(zip(*traj)), s=3, c='white', label=r'Trajectory $\mathbf{x}^{(i)}$')

ax.scatter([result.x[0], ], [result.x[1], ], c='red', label=r'Solution $\mathbf{x}^\star$')
ax.legend()
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

fig.savefig('figures/fig-traj.pdf', dpi=200)
# plt.show()
