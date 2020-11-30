from pprint import pprint

import sympy as sym
import numpy as np

from .benchmark import Benchmark


class Hartmann3(Benchmark):

    def __init__(self):
        super().__init__()

        self.name = 'Hartmann\'s family n=3'
        self.name_short = 'H3'

        m, n = 4, 3
        x = sym.IndexedBase('x')
        self.x = [x[i] for i in range(0, n)]
        self.dims = n

        assert len(self.x) == self.dims

        i, j = sym.Idx('i'), sym.Idx('j')
        a = sym.IndexedBase('a')
        c = sym.IndexedBase('c')
        p = sym.IndexedBase('p')

        a_ij = np.array([[3, 10, 30],
                         [0.1, 10, 35],
                         [3, 10, 30],
                         [0.1, 10, 35]])
        c_i = np.array([1, 1.2, 3., 3.2])
        p_ij = np.array([[0.3689, 0.1170, 0.2673],
                         [0.4699, 0.4387, 0.7470],
                         [0.1091, 0.8732, 0.5547],
                         [0.038150, 0.5743, 0.8828]])

        c_params = {f'c[{i}]': [c[i], c_i[i]] for i in range(m)}
        a_params = {f'a[{i},{j}]': [a[i, j], a_ij[i, j]] for i in range(m) for j in range(n)}
        p_params = {f'p[{i},{j}]': [p[i, j], p_ij[i, j]] for i in range(m) for j in range(n)}

        self.params = {**c_params, **a_params, **p_params}

#        print(f'i range is: {[ i for i in range(m)]}')
#        print(f'j range is: {[ j for j in range(n)]}')

        self.expr = -sym.Sum(c[i] * sym.exp(-sym.Sum(a[i, j] * (x[j] - p[i, j])**2,
                                                     (j, 0, n - 1))), (i, 0, m - 1))
#        self.xmin = [[p_ij[i, j] for j in range(n)] for i in range(m)]
        self.xmin = [[0.11461433, 0.55564885, 0.85254695], ]
        self.domain = [np.zeros(self.dims), np.ones(self.dims)]
        self.domain_plot = None


class Hartmann6(Benchmark):

    def __init__(self):
        super().__init__()

        self.name = 'Hartmann\'s family n=3'
        self.name_short = 'H6'

        m, n = 4, 6
        x = sym.IndexedBase('x')
        self.x = [x[i] for i in range(0, n)]
        self.dims = n

        i, j = sym.Idx('i'), sym.Idx('j')
        a = sym.IndexedBase('a')
        c = sym.IndexedBase('c')
        p = sym.IndexedBase('p')

        a_ij = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                         [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                         [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                         [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
        c_i = np.array([1.0, 1.2, 3.0, 3.2])
        p_ij = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                         [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                         [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                         [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

        c_params = {f'c[{i}]': [c[i], c_i[i]] for i in range(m)}
        a_params = {f'a[{i},{j}]': [a[i, j], a_ij[i, j]] for i in range(m) for j in range(n)}

        p_params = {f'p[{i},{j}]': [p[i, j], p_ij[i, j]] for i in range(m) for j in range(n)}

        self.params = {**c_params, **a_params, **p_params}

        self.expr = -sym.Sum(c[i] * sym.exp(-sym.Sum(a[i, j] * (x[j] - p[i, j])**2,
                                                     (j, 0, n - 1))), (i, 0, m - 1))
#        self.xmin = [[p_ij[i, j] for j in range(n)] for i in range(m)]
        self.xmin = [[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573], ]
        self.domain = [np.zeros(self.dims), np.ones(self.dims)]
        self.domain_plot = None
