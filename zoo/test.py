from rosenbrock import Rosenbrock
from hartmann3 import Hartmann3

#bench = Rosenbrock()
bench = Hartmann3()

bench_exp = bench.make_explicit()

print(bench_exp.expr)
