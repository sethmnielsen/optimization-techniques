#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import jax
import jax.numpy as jnp
from jax.numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import pyoptsparse as pyop
from pyoptsparse.pyOpt_solution import Solution
from pyoptsparse.pyOpt_optimization import Optimization
from pyoptsparse.pyOpt_optimizer import Optimizer
import seaborn as sns
sns.set_style('whitegrid')


from truss import truss
import derivatives as deriv

'''
abs(sigma) < sigma_yield  # okay, but deriv at zero is undefined for abs
sigma**2 < sigma_yield**2    # bad

---- good case -----
1) sigma < sigma_yield
2) sigma > -sigma_yield

1) g =  sigma - sigma_yield
2) g = -sigma - sigma_yield
--------------------

----next------
grad_g = partial(gi)/partial(xj)   ; transpose --> partial(gj)/partial(xi)


----next------
for finite differencing, remember to RESET vector between each loop

----next------
normmalize/scaling
instead of g = sigma-sigma_yield, do g = sigma/sigma_yield - 1
... obj = mass/1000

'''

'''
A nonlinear, constrained optimization problem for finding the optimal cross-sectional area
for a 10-bar truss to minimize the overall mass of the structure.
Objective:
  mass -> 0
Design Variables:
  Cross-sectional area of each bar
Constraints:
  s >= -s_y : each bar must not yield in compression
  s <=  s_y : each bar must not yield in tension
  area >= 0.1 sq-in (bound constraint)

*may need to consider scaling the objective and constraints
'''

class truss_solver():
    def __init__(self):
        self.func_evals = 0
        self.mass_hist = []
        self.stress_hist = []

        self.n = 10
        # self.areas = np.ones(self.n) * 0.1
        self.areas = np.ones(self.n)

        self.stress_yield = np.array([25e3]*self.n)
        self.stress_yield[8] = 75e3

        self.iterations = 0
        self.iters_limit = 1e4

    def run(self):
        sol = self.solve_problem()
        # self.plot_final_results(sol)

    def solve_problem(self):
        method = 'FD'
        self.m, self.s = truss(self.areas)
        # while self.iterations < self.iters_limit:
        while self.iterations < 1:
            # dm will be vector, ds will be matrix
            dm, ds = deriv.get_derivatives(method, self.areas, self.m, self.s)

            self.iterations += 1


        print('...done!')

        print(f'method: {method}')
        print(f'max of dm/dA: {np.max(dm)}')
        print(f'max of ds/dA: {np.max(ds)}')

        return np.zeros(1)

    def plot_final_results(self, sol):
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False)

        stress_data = np.array(self.stress_hist)
        inds = np.array([0,1,2,3,4,5,6,7,9])
        stress9 = stress_data[:,8]
        stress_others = stress_data[:,inds]
        others_labels = ['1','2','3','4','5','6','7','8','10']
        xdata = np.arange(len(self.mass_hist))
        # axes[2].set_xticks(self.num_pts)
        # axes[2].set_xticks(np.arange(0, self.num_pts[-1], 8), minor=True)

        axes[0].plot(xdata, self.mass_hist)
        axes[0].set_title("Truss Problem Convergence")
        axes[0].set_ylabel("Mass (lbs)")
        axes[1].plot(xdata, stress9, label='Bar 9')
        axes[1].plot(xdata, stress_others)
        axes[1].set_ylabel("Stress on bar (lbs)")
        axes[1].legend(loc='upper right')

        axes[2].plot(xdata[300:], stress9[300:], label='Bar 9')
        axes[2].plot(xdata[300:], stress_others[300:])
        axes[2].set_xlabel('Function calls')
        axes[2].set_ylabel('Stress on bar (lbs)')
        axes[2].legend(loc='upper right')
        plt.show()


if __name__ == '__main__':
    op = truss_solver()
    op.run()
    print('Optimization finished')
