import sympy

# from sympy.parsing.sympy_parser import(parse_expr, standard_transformations,
#                                       implicit_multiplication_application, convert_xor)

from sympy.parsing.latex import parse_latex

#transforms = (standard_transformations + (implicit_multiplication_application, ))

sympy.init_printing(use_unicode=True)

ltx = r'\sum_{i=1}^m x+y'

expr = parse_latex(ltx)

#expr = parse_expr('$\sum x$', transformations=transforms)

print(expr)
