# Simulated Annealing

Global stochastic optimization method

## Description

A numerical analysis and implementation of *Global Optimization and Simulated Annealing* by A. Dekkers and E. Aarts.

![](figures/fig-traj.svg?raw=true)

## Getting Started

### Dependencies

* Matplotlib
* NumPy
* SciPy

### API

Use our implementation of Simulated Annealing with the following API.

```
sa.simulated_annealing(f, jac, domain, l0, delta, stop_eps, chi, gamma, t, init_trials = 200,
                       callback = None, tol = 1e-7, polish = True, polish_minimizer = optim.minimize, 
                       polish_kwargs = dict())
```

#### Parameters
* **f : Callable**
        Function to minimize
* **jac : Callable**
        Jacobian of f.
* **domain : ndarray**
        Region S on which to minimize f.
* **l0 : int**
        Basic length of Markov chains.
* **delta : float**
        Cooling schedule decrement rate.
* **stop_eps : float**
        Stop condition control parameter.
* **chi : float**
        Initial acceptance: between 0 and 1.
* **gamma : float**
        Smoothing parameter between 0 and 1.
* **t : float**
        Descent direction affinity between 0 and 1.
* **init_trials : int**
        Initial transitions to initialize schedule.
* **callback : Callable**
        Function called on each iteration.
* **tol : float**
        Stopping tolerance of final solution.
* **polish : bool**
        Refine final solution using LS.
* **polish_minimizer : Callable**
        Method to use to refine solution.
* **polish_kwrags : dict**
        Keyword arguments to pass to polish minimizer.

#### Returns
* **polish_kwrags : dict**
        Contains solution x and running statistics.
        
                            

## Code organisation

* `zoo.Zoo` provides an easy-to-use interface to use the various benchmark functions.
* `compare2.py` performs the main comparisons between the different methods.
* `compare.json` contains exhaustive detail run logs from `compare2.py`.
* `multistart.py` implements the Multi-start optimization algorithm using the double-box trick stopping rule.
* The remaining `.py` files are used for generating misc. figures for the report.


## Authors

* [Gian Alix @techGIAN](https://github.com/techGIAN)
* Celina Landolfi
* [Jacques Nel @jmnel](https://github.com/jmnel)
