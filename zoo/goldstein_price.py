import sympy as sym

from .benchmark import Benchmark


class GoldsteinPrice(Benchmark):

    def __init__(self):
        super().__init__()

        self.name = 'Goldstein-Price'
        self.name_short = 'GP'
        x1, x2 = sym.symbols('x_1 x_2')
        self.x = [x1, x2]

        self.params = dict()

        term1 = (1 + (x1 + x2 + 1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2))
        term2 = (30 + (2 * x1 - 3 * x2)**2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2))

        self.expr = term1 * term2
        self.xmin = [[0., -1.], ]
        self.domain = [[-2., -2.], [2., 2.]]
        self.domain_plot = [[-2., -2.], [2., 2.]]
