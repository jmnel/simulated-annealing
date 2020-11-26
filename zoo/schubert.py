from pprint import pprint

import sympy as sym

sym.init_printing(use_latex=True)
import numpy as np

from benchmark import Benchmark


class Schubert(Benchmark):

    def __init__(self, case: str):
        super().__init__()

        if case not in {'p3', 'p8', 'p16', 'p22'}:
            raise ValueError('case must be one of p3, p8, p16, or p22')

        self.name = f"schubert {case}"

        def u(x_i, a, k, m):
            #            a, k, m = symbols('a k m')

            #            x = IndexedBase('x')
            #            i = Idx('i')

            return sym.Piecewise(
                (k * (x_i - a)**m, sym.Gt(x_i, a)),
                (0, sym.And(sym.Ge(x[i], -a), sym.Le(x[i], a))),
                (k * (-x_i - a)**m, sym.Lt(x_i, -a))
            )

        a, k, m = sym.symbols('a k m')

        if case == 'p3':
            n = 2
            x = sym.IndexedBase('x')
            self.x = [x[i] for i in range(0, n)]

            i = sym.Idx('i')

            term1 = sym.Sum(i * sym.cos((i + 1) * x[0] + 1), (i, 0, 4))
            term2 = sym.Sum(i * sym.cos((i + 1) * x[1] + 1), (i, 0, 4))

            self.expr = term1 * term2 + u(x[0], a, k, m)

#            print(repr(u(x[i], 0.2, k, m)))

            print(repr(self.expr))

#        var_x = sym.Matrix(self.x)

#        i, j = sym.Idx('i'), sym.Idx('j')
#        a = sym.IndexedBase('a')
#        c = sym.IndexedBase('c')

#        a_ij = np.array([[4., 4., 4., 4.],
#                         [1., 1., 1., 1.],
#                         [8., 8., 8., 8.],
#                         [6., 6., 6., 6.],
#                         [3., 7., 3., 7.],
#                         [2., 9., 2., 9.],
#                         [5., 5., 3., 3.],
#                         [8., 1., 8., 1.],
#                         [6., 2., 6., 2.],
#                         [7., 3.6, 7., 3.6]])
#        c_i = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

#        a_ij = a_ij[:m]
#        c_i = c_i[:m]

#        c_params = {f'c[{i}]': [c[i], c_i[i]] for i in range(m)}
#        a_params = {f'a[{i},{j}]': [a[i, j], a_ij[i, j]] for i in range(m) for j in range(n)}

#        self.params = {**c_params, **a_params}

#        self.expr = -sym.Sum(1.0 / (sym.Sum((x[j] - a[i, j])**2, (j, 0, n - 1)) + c[i]), (i, 0, n - 1))

#        self.xmin = [[a_ij[i, j] for j in range(n)] for i in range(m)]
#        self.domain = [np.zeros(n), np.ones(n)]
#        self.domain_plot = None


# q = Shekel(m=5)

q = Schubert(case='p3')

# print(q)
