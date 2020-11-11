from pprint import pprint

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np


def f_rosenbrock(x, y):
    a, b = 1, 100
    return (a - x)**2 + b * (y - x**2)**2


n_samples = 1000
xlim = [-2, 2]
ylim = [-1, 3]
x, y = np.linspace(*xlim, n_samples), np.linspace(*ylim, n_samples)
mx, my = np.meshgrid(x, y)
z = np.power(f_rosenbrock(mx, my), 0.1)
# z = f_rosenbrock(mx, my)
plt.contourf(x, y, z, levels=100, cmap='viridis')


def gen_point(dom):
    #    pprint(dom)
    return np.random.random(2) * (dom[:, 1] - dom[:, 0]) + dom[:, 0]
#    return np.random.random(2)


def sa(f):
    dom = np.array([[-2, 2], [-1, 3]])
    m = 1000
    init_trials = [gen_point(dom) for _ in range(m + 1)]
    f_m = np.diff([f(*p) for p in init_trials])

    m_1 = sum(e <= 0 for e in f_m)
    m_2 = m - m_1
#    print(m_1)
    acceptance_ratio = 0.9

#    print([e for e in f_m if e > 0])
    den = m_2 * acceptance_ratio + (1. - acceptance_ratio) * m_1
    fbar = np.mean([v for v in f_m if v > 0])
    print(f'fbar: {fbar}')
    print(f'den: {den}')
    print(f'm_1: {m_1}, m_2: {m_2}')
    q = np.log(m_2 / den)
    print(f'q: {q}')
#    c_0 = np.mean([e for e in f_m if e > 0]) * (1 / (np.log(m_2 / (m_2 * xhi + (1 - xhi) * m_1))))
#    print(c_0)

#    pprint(f_m)
#    f_m = np.diff(list(map(f, init_trials)))

#    pprint(f_m)

#    for idx, p in enumerate(init_trials):
#        print(f'{idx} -> {p}')
#    pass


sa(f_rosenbrock)


# plt.show()
