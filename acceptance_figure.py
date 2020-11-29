import matplotlib
# matplotlib.use('module://matplotlib-backend-kitty')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from zoo import Zoo
from sa import _gen_point_a

obj = Zoo().get('rosenbrock').make_explicit()
dom = np.array(obj.domain)
f = obj.f
m = 50000

x_samples = [_gen_point_a(dom) for _ in range(m + 1)]
delta = np.diff([f(x) for x in x_samples])
delta = np.array([d for d in delta if d > 0.])

#x = np.linspace(0, 1000, 300)
c = [30, 50, 1000]

fig, ax = plt.subplots(3, 1)

for idx, c_i in enumerate(c):

    p = np.exp(-delta / c_i)
    ax[idx].hist(p, bins=200, color=f'C{idx}', label=f'$c={c_i}$')
    ax[idx].set_xlabel(r'Acceptance probability $\exp( -\Delta f / c)$')
    ax[idx].set_ylabel(r'Num. transitions')
    ax[idx].legend()

plt.tight_layout()

fig.savefig('figures/fig33.pdf', dpi=200)
# plt.show()
