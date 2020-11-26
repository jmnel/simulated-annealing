from pprint import pprint

import sympy as sym

sym.init_printing(use_latex=True)
import numpy as np

from benchmark import Benchmark


class Shekel(Benchmark):

    def __init__(self, m: int):
        super().__init__()

        if m not in {5, 7, 10}:
            raise ValueError('m must be one of 5, 7, or 10')

        self.name = "hartmann3"

        n = 4
        x = sym.IndexedBase('x')
        self.x = [x[i] for i in range(0, n)]

        var_x = sym.Matrix(self.x)

        i, j = sym.Idx('i'), sym.Idx('j')
        a = sym.IndexedBase('a')
        c = sym.IndexedBase('c')

        a_ij = np.array([[4., 4., 4., 4.],
                         [1., 1., 1., 1.],
                         [8., 8., 8., 8.],
                         [6., 6., 6., 6.],
                         [3., 7., 3., 7.],
                         [2., 9., 2., 9.],
                         [5., 5., 3., 3.],
                         [8., 1., 8., 1.],
                         [6., 2., 6., 2.],
                         [7., 3.6, 7., 3.6]])
        c_i = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

        a_ij = a_ij[:m]
        c_i = c_i[:m]

        c_params = {f'c[{i}]': [c[i], c_i[i]] for i in range(m)}
        a_params = {f'a[{i},{j}]': [a[i, j], a_ij[i, j]] for i in range(m) for j in range(n)}

        self.params = {**c_params, **a_params}

        self.expr = -sym.Sum(1.0 / (sym.Sum((x[j] - a[i, j])**2, (j, 0, n - 1)) + c[i]), (i, 0, n - 1))

        self.xmin = [[a_ij[i, j] for j in range(n)] for i in range(m)]
        self.domain = [np.zeros(n), np.ones(n)]
        self.domain_plot = None


#q = Shekel(m=5)
