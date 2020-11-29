import numpy as np
import matplotlib
# matplotlib.use('module://matplotlib-backend-kitty')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from zoo import Zoo
from sa import simulated_annealing

objectives = [Zoo().get('branin').make_explicit(),
              Zoo().get('goldstein_price').make_explicit()]

NUM_RUNS = 4
GAMMA = 0.01

for obj in objectives:

    f, grad = obj.f, obj.grad
    domain, plt_domain = np.array(obj.domain), np.array(obj.domain_plot)

    DOM_DIM = 2
    L0 = 10
    L = DOM_DIM * L0
    DELTA = 0.1
    EPS = 1e-4
    CHI = 0.9
    SMOOTHING = 0.01
    T = 0.1

    def callback(iteration, x, chain, c):
        x_hist.extend(chain)
        c_hist.append(c)

    xlim = plt_domain[:, 0]
    ylim = plt_domain[:, 1]
    n_samples = 200
    x_plt, y_plt = np.linspace(*xlim, n_samples), np.linspace(*ylim, n_samples)
    mx, my = np.meshgrid(x_plt, y_plt)
    z = np.power(f([mx, my]), 0.1)

    fig, ax = plt.subplots(1, 1)
    ax.contourf(x_plt, y_plt, z, levels=50, cmap='viridis')

    colors = plt.cm.twilight(np.linspace(0, 1, NUM_RUNS + 1))

    temps = list()

    for idx_run in range(NUM_RUNS + 1):
        c_hist = list()
        x_hist = list()

        res = simulated_annealing(f,
                                  grad,
                                  domain=domain,
                                  l0=L0,
                                  delta=DELTA,
                                  stop_eps=EPS,
                                  chi=CHI,
                                  smoothing=SMOOTHING,
                                  descent_affinity=T,
                                  callback=callback)

        x_smooth = res[0]
        x_smooth_hist = [x_smooth, ]

        temps.append(c_hist)

        for x_i in reversed(x_hist):
            x_smooth = GAMMA * x_i + (1. - GAMMA) * x_smooth
            x_smooth_hist.append(x_smooth)

        ax.scatter([res[0][0], ], [res[0][1], ], c=np.array(colors[idx_run]).reshape((1, 4)))
        ax.plot(*tuple(zip(*x_smooth_hist)), color=colors[idx_run], linewidth=0.6)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')

    fig.savefig(f'figures/fig51-{obj.name}.pdf', dpi=200)

    fig, ax = plt.subplots(1, 1)

    for c_hist in temps:
        ax.plot(np.arange(len(c_hist)), c_hist)

    ax.set_xlabel(r'Iteration $n$')
    ax.set_ylabel(r'Temperature $c^{(n)}$')
    fig.savefig(f'figures/fig52-{obj.name}.pdf', dpi=200)
