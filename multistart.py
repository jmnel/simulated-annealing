
import random as rnd
import numpy as np

def multi_start(n, a, b, K, f, tol):
    i = 1
    x_stars = [] # will hold all x_stars
    f_xstars = [] # will hold all values of f(x_star), where x_star is in x_stars

    while(i <= K):
        x = np.random.uniform(a, b, n)
        x_star = grad_descent(f, x, grad=None, tol=1e-14, max_iterations=10000, ls_method='exact', eps=1e-14, method='steepest_descent')
        x_stars.append(x_star)
        f_xstars.append(f(x_star))
        i += 1

    return x_stars[np.argmin(f_xstars)]