import copy
from pprint import pprint
from numbers import Real
from typing import Sequence

import numpy as np
import sympy as sym
from sympy.parsing.latex import parse_latex


class Benchmark():

    def __init__(self):

        self.name = None
        self.x = None
        self.params = list()
        self.expr = None
        self.domain = np.array([[0., 0.], [1., 1.]])
        self.domain_plot = np.array([[0., 0.], [1., 1.]])
        self.xmin = list()

        self._f = None
        self._grad = None

    def make_explicit(self, **kwargs):

        bench = copy.deepcopy(self)

        for key, value in kwargs.items():
            if key in bench.params:
                bench.params[key][1] = value

        subs_list = {symbol: value for _, (symbol, value) in bench.params.items()}
        f = bench.expr \
            .doit() \
            .subs(subs_list)

        bench._f = sym.lambdify(bench.x, f)
        bench._grad = [sym.lambdify(bench.x, sym.diff(f, x_i)) for x_i in bench.x]

        assert isinstance(bench.xmin, Sequence)
        assert len(bench.xmin) > 0
        for idx, xmin_i in enumerate(bench.xmin):

            assert isinstance(xmin_i, Sequence)
            assert len(xmin_i) > 0

            for jdx, xmin_ij in enumerate(xmin_i):

                if isinstance(xmin_ij, Real):
                    bench.xmin[idx][jdx] = float(xmin_ij)

                elif isinstance(xmin_ij, sym.Expr):
                    bench.xmin[idx][jdx] = xmin_ij.subs(subs_list).n()

                else:
                    raise ValueError('xmin element component must either be Real or sympy.Expr')

        return bench

    def f(self, x):
        if self._f is None:
            raise ValueError('benchmark must be made explicit first before evaluation')

        return self._f(*x)

    def grad(self, x):
        if self._grad is None:
            raise ValueError('benchmark must be made explicit first before evaluation')

        return np.array([grad_i(*x) for grad_i in self._grad])
