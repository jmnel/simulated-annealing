from typing import List, Dict, Callable, Union, Sequence
from numbers import Real
from collections import namedtuple
import json
from pprint import pprint
from pathlib import Path
import copy

import sympy
from sympy.parsing.latex import parse_latex
import numpy as np

sympy.init_printing(use_latex=True)


class Benchmark():
    """
    Class representing a single benchmark test function.

    A LaTeX expression is used to symbolically construct the function. The known minimum/
    minima can either be:
        1) a point reprensted by a Numpy array,
        2) a str with LaTeX expression of minimum/minima,
        3) or a sequence of one of the above options, in the case of multiple minima.

    When the caller retrieves the function from the zoo, arguments are subbed in for the
    function, gradient, and the minimum/minima when appropriate.

    """

    def __init__(self,
                 name: str,
                 latex: str,
                 domain: np.array,
                 plot_domain: np.array,
                 args: Dict,
                 xmin: Union[np.array, str, Sequence[Union[np.array, str]]]):

        self.name = name
        self.latex = latex
        self.expr = parse_latex(latex)
        self.domain = domain
        self.plot_domain = plot_domain
        self.args = args
        self.xmin = xmin

    def __str__(self):
        s = 'benchmark {}\n'.format(self.name)
        s += '  expr: {}\n'.format(self.expr)
        s += '  args: {}'.format(self.args)

        return s


class Zoo():
    """
    The zoo provides a unified API to access, numerically evaluate,
    and lookup info for various optimization benchmark functions.

    Lambdas are provided to evaluate functions. Gradients are automatically computed.

    """

    def __init__(self):

        # Load function library from JSON file.
        config_path = Path(__file__).absolute().parent / 'zoo.json'
        with open(config_path, 'rt') as config_f:
            config = json.load(config_f)

        self.benchmarks = dict()
        for name, props in config.items():
            self.benchmarks[name] = Benchmark(
                name,
                props['latex'],
                props['domain'],
                props['plot_domain'],
                props['args'],
                props['xmin'])

    def get(self, name: str, **kwargs) -> Benchmark:
        """
        Get a benchmark function using its name.

        Args:
            name:       Name of function to get.

        """

        if name not in self.benchmarks:
            raise ValueError(f'benchmark "{name}" is not available')

        bench = copy.deepcopy(self.benchmarks[name])

        # Override function parameters with those supplied by caller.
        for key, value in kwargs.items():
            if key in bench.args:
                bench.args[key] = value

        # Parse arguments if they are LaTeX; special treatment is given to π.
        pi_symbol = sympy.symbols('pi')
        for key, value in bench.args.items():
            if isinstance(value, str):
                bench.args[key] = parse_latex(value).subs(pi_symbol, np.pi)

        # Parse function expression and sub in arguments.
        f = parse_latex(bench.latex)
        f = f.subs(bench.args)

        # Get remaining symbols and sort.
        x = list(f.free_symbols)
        x = [(x_i.name.replace('{', '')
              .replace('}', '')
              .replace('_', ''), x_i) for x_i in x]
        x = [(int(x_i[0][1:]), *x_i) for x_i in x]
        x = sorted(x)

        # Convert remaining symbols to function variables.
        x = [sympy.symbols(x_i[2].name) for x_i in x]

        bench.dim = len(x)

        # Construct function lambda.
        f_inner = sympy.lambdify(x, f)
        bench.f = lambda x: f_inner(*x)

        # Construct function gradient lambda.
        g = [sympy.lambdify(x, sympy.diff(f, x_i)) for x_i in x]
        bench.grad = lambda x: np.array([g_i(*x) for g_i in g])

        # Make minima/minimum numerically explicit by subbing in arguments.

        assert isinstance(bench.xmin, Sequence)

        for min_idx, min_i in enumerate(bench.xmin):

            # Case 1: single minimum is provided as a LaTeX expression.
            if isinstance(min_i, str):
                bench.xmin[min_idx] = parse_latex(min_i) \
                    .subs(bench.args) \
                    .subs(pi_symbol, np.pi)

            # Case 2: a sequence of minima is provided.
            elif isinstance(min_i, Sequence):
                for min_jdx, min_ij in enumerate(min_i):

                    # Case 2.1: Component provided is a LaTeX expression.
                    if isinstance(min_ij, str):
                        bench.xmin[min_idx][min_jdx] = parse_latex(min_ij) \
                            .subs(bench.args) \
                            .subs(pi_symbol, np.pi)

                    # Case 2.2: Component proivded is a real number.
                    else:
                        assert isinstance(min_ij, Real)
                        bench.xmin[min_idx][min_jdx] = float(min_ij)

            # Case 3: this component of the minimum must be a real number.
            else:
                assert isinstance(min_i, Real)
                bench.xmin[min_idx] = float(min_i)

        return bench


zoo = Zoo()

gp = zoo.get('branin')

pprint(gp.xmin)

# pprint(gp.args)

#res = gp.f([0.0, -1.0])
#g = gp.grad([0.0, -1.0])

#print(res, g)

#rosen = zoo.get('rosenbrock')

#print(f'res={rosen.f([12, 12])}')
#print(f'grad={rosen.grad([1, 1])}')
#print(f'xmin = {rosen.xmin}')
