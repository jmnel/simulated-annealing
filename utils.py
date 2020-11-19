import numpy as np


def grad_approx(f, x, tau=1e-14):
    x1, x2 = x
    return np.array([
        (f([x1 + 0.5 * tau, x2]) - f([x1 - 0.5 * tau, x2])) / tau,
        (f([x1, x2 + 0.5 * tau]) - f([x1, x2 - 0.5 * tau])) / tau])
