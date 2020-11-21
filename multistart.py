
# Multistart method is two-step:
# 1. Solution construction
# 2. Solution improvement
def multi_start(K, tol):
    # initialize i and the bests
    i = 1
    old_best = 0
    new_best = np.inf
    # assume stopping condition is when |new_best - old_best| < tol and
    #                               when i reaches the maximum iteration number K
    while(i <= K and abs(new_best - old_best) < tol):
        # Step 1: Solution consstruction
        x = fmin(f, dom, l0, delta, eps, chi, smoothing, t)
        # Step 2: Solution improvement
        improved_x = grad_descent(f, x, grad=None, tol=1e-14, max_iterations=10000, ls_method='exact', eps=1e-14)
        if (improved_x < new_best):
            old_best = new_best
            new_best = improved_x
        i += 1
    return new_best
        