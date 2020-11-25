from pprint import pprint

import sympy as sym
import numpy as np

from benchmark import Benchmark


class Hartmann3(Benchmark):

    def __init__(self):
        super().__init__()

        self.name = "hartmann_3"

        m, n = 4, 3
        x = sym.IndexedBase('x')
        self.x = [x[i] for i in range(1, m + 1)]

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

        a_subs = {a[i, j]: a_ij[i - 1, j - 1] for i in range(1, m + 1) for j in range(1, n + 1)}
        c_subs = {c[i]: c_i[i - 1] for i in range(1, m + 1)}
        p_subs = {p[i, j]: p_ij[i - 1, j - 1] for i in range(1, m + 1) for j in range(1, n + 1)}
        subs_list = {**a_subs, **c_subs, **p_subs}

        self.expr = -sym.Sum(c[i] * sym.exp(-sym.Sum(a[i, j] * (x[i] - p[i, j])**2,
                                                     (j, 1, n))), (i, 1, m))
        self.xmin = [[p_ij[i, j] for j in range(n)] for i in range(m)]
        self.domain = [np.zeros(n), np.ones(n)]
        self.domain_plot = None


class Hartmann6(Benchmark):

    def __init__(self):
        super().__init__()

        self.name = "hartmann_3"

        m, n = 4, 6
        x = sym.IndexedBase('x')
        self.x = [x[i] for i in range(1, m + 1)]

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

        a_subs = {a[i, j]: a_ij[i - 1, j - 1] for i in range(1, m + 1) for j in range(1, n + 1)}
        c_subs = {c[i]: c_i[i - 1] for i in range(1, m + 1)}
        p_subs = {p[i, j]: p_ij[i - 1, j - 1] for i in range(1, m + 1) for j in range(1, n + 1)}
        subs_list = {**a_subs, **c_subs, **p_subs}

        self.expr = -sym.Sum(c[i] * sym.exp(-sym.Sum(a[i, j] * (x[i] - p[i, j])**2,
                                                     (j, 1, n))), (i, 1, m))
        self.xmin = [[p_ij[i, j] for j in range(n)] for i in range(m)]
        self.domain = [np.zeros(n), np.ones(n)]
        self.domain_plot = None
