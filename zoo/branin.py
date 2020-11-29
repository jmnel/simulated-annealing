import sympy as sym

from .benchmark import Benchmark


class Branin(Benchmark):

    def __init__(self):
        super().__init__()

        self.name = 'Branin'
        self.name_short = 'BR'

        x = sym.IndexedBase('x')
        i = sym.Idx('i')
        x1, x2 = x[1], x[2]
        self.x = [x1, x2]

        a, b, c, d, e, f = sym.symbols('a b c d e f')
        self.params = {'a': [a, 1],
                       'b': [b, 5.1 / (4 * sym.pi**2)],
                       'c': [c, 5 / sym.pi],
                       'd': [d, 6],
                       'e': [e, 10],
                       'f': [f, 1 / (8 * sym.pi)]}

        self.expr = a * (x2 - b * x1**2 + c * x1 - d)**2 + e * (1 - f) * sym.cos(x1) + e
        self.xmin = [[-sym.pi, 12.275],
                     [sym.pi, 2.275],
                     [3 * sym.pi, 2.475]]

        self.domain = [[-5., -0.], [10., 15.]]
        self.domain_plot = [[-5., -0.], [10., 15.]]
