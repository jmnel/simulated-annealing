from __future__ import annotations
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

from .rosenbrock import Rosenbrock
from .goldstein_price import GoldsteinPrice
from .branin import Branin
from .hartmann import Hartmann3, Hartmann6
from .shekel import Shekel


class Zoo():
    """
    The zoo provides a unified API to access, numerically evaluate,
    and lookup info for various optimization benchmark functions.

    Lambdas are provided to evaluate functions. Gradients are automatically computed.

    """

    def __init__(self):

        self.benchmarks = {'RB': Rosenbrock,
                           'GP': GoldsteinPrice,
                           'BR': Branin,
                           'H3': Hartmann3,
                           'H6': Hartmann6,
                           'S': Shekel}
#        self.benchmarks = {'rosenbrock': bench for bench in benches}

    def get(self, name: str, **kwargs) -> Benchmark:
        """
        Get a benchmark function by name.

        Pass function parameter overrides to **kwargs.

        Arg:
            name:           Name of benchmark.

        Returns:
            Benchmark:      Explicit benchmark function.

        """

        if name not in self.benchmarks:
            raise ValueError(f'benchmark "{name}" is not available')

        return self.benchmarks[name](**kwargs).make_explicit(**kwargs)

    def list_benchmarks(self) -> List[str]:
        return list(self.benchmarks.keys())

    def __getitem__(self, item):
        return self.benchmarks[item]

    def __len__(self):
        return len(self.benchmarks)

    def keys(self):
        return self.benchmarks.keys()

    def itesm(self):
        return self.benchmarks.items()
