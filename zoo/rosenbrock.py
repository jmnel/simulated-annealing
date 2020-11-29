import sympy as sym

from .benchmark import Benchmark


class Rosenbrock(Benchmark):

    def __init__(self):
        super().__init__()

        self.name = 'Rosenbrock'
        self.name_short = 'RB'
        x = sym.IndexedBase('x')
        i = sym.Idx('i')
        x1, x2 = x[1], x[2]
        self.x = [x1, x2]

        a, b = sym.symbols('a b')
        self.params = {'a': [a, 1.],
                       'b': [b, sym.pi * 4]
                       }

        self.expr = (a - x1)**2 + b * (x2 - x1**2)**2
        self.xmin = [[a, a**2], ]

        self.domain = [[-2., -1.], [2., 2.]]
        self.domain_plot = [[-2., -2.], [3., 3.]]
