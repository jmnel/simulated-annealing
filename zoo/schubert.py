from pprint import pprint

import sympy as sym

sym.init_printing(use_latex=True)
import numpy as np

from .benchmark import Benchmark


class Schubert(Benchmark):

    def __init__(self, case: str):
        super().__init__()

        if case not in {'p3', 'p8', 'p16', 'p22'}:
            raise ValueError('case must be one of p3, p8, p16, or p22')

        self.name = f"schubert {case}"

        def u(x_i, a, k, m):

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

            self.expr = term1 * term2 + u(x[0], a, k, m) + u(x[1], a, k, m)
            self.params = {'a': [a, 10.],
                           'k': [k, 100.],
                           'm': [m, 2]}

            self.xmin = None
            self.domain = [-10. * np.ones(n), 10. * np.ones(n)]
            self.domain_plot = self.domain

        elif case == 'p8':
            n = 3
            x = sym.IndexedBase('x')
            self.x = [x[i] for i in range(0, n)]

            y = sym.IndexedBase('y')

            i = sym.Idx('i')

            k_1, k_2 = sym.symbols('k_1 k_2')

            pprint(y)

            self.expr = (sym.pi / n) * (
                k_1 * sym.sin(sym.pi * y[0])**2
                + sym.Sum((y[i] - k_2)**2
                          * (1. + k_1 * sym.sin(sym.pi * y[i + 1])**2), (i, 0, n - 2))
                + (y[n - 1] - k_2)**2) \
                + sym.Sum(u(x[i], a, k, m), (i, 0, n - 1))

            y_subs = {y[i]: 1. + 0.25 * (x[i] + 1.) for i in range(n)}
            self.expr = self.expr.doit().subs(y_subs)

            self.params = {'a': [a, 10.],
                           'k': [k, 100.],
                           'm': [m, 4],
                           'k_1': [k_1, 10.],
                           'k_2': [k_2, 1.]}

            self.xmin = [[1., 1., 1.], ]
            self.domain = [-10. * np.ones(n), 10. * np.ones(n)]
            self.domain_plot = None
