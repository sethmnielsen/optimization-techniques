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


import derivatives as deriv

#region
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
normalize/scaling
instead of g = sigma-sigma_yield, do g = sigma/sigma_yield - 1
obj = mass/1000

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
#endregion

class truss_solver:
    def __init__(self, method, areas=None):
        self.method = method
        self.func_calls = 0
        self.mass_hist = []
        self.stress_hist = []
        self.eval_hist = []
        deriv.truss.func_calls_fd = 0
        deriv.truss.func_calls_adj = 0

        self.n = 10
        # self.areas = np.ones(self.n) * 0.1
        self.areas = np.ones(self.n)
        
        if areas is not None:
            self.areas = areas
        else:
            self.areas = np.ones(self.n)

        self.iterations = 0
        self.iters_limit = 1e4

    def objfunc(self, data):
        x = data['areas']
        funcs = {}
        mass, stress = deriv.truss.truss(x)
        
        # mass, stress, dm, ds = deriv.finite_diff(x)
        funcs['mass'] = mass
        funcs['stress_arr'] = stress

        if self.method == 'FD':
            self.eval_hist.append(deriv.truss.func_calls_fd)
        else:
            self.eval_hist.append(deriv.truss.func_calls_adj)
        self.mass_hist.append(mass)
        self.stress_hist.append(np.mean(stress))
        return funcs, False

    def init_problem(self):
        # Optimization problem
        self.opt_prob: pyop.Optimization = pyop.Optimization('trusty', self.objfunc)
        # Design variables
        self.opt_prob.addVarGroup('areas', nVars=10, type='c', value=self.areas, \
                                  lower=0.1, upper=None)
        

        # Constraints
        # yield_compression, yield_tension, area_min
        stress_yield = np.array([25e3]*10)
        stress_yield[8] = 75e3
        self.opt_prob.addConGroup('stress_arr', nCon=10, lower=-stress_yield, upper=stress_yield)

        # Assign the key value for the objective function
        self.opt_prob.addObj('mass')

        self.optimizer = pyop.SNOPT()
        self.optimizer.setOption('iPrint',0)
        path = '/home/seth/school/optimization/output/'
        self.optimizer.setOption('Print file', path+f'SNOPT_print.out')
        self.optimizer.setOption('Summary file', path+f'SNOPT_summary.out')

    def solve_problem(self):
        if self.method == 'my FD':
            sens_func = deriv.finite_diff
        elif self.method == 'adjoint':
            sens_func = deriv.adjoint

        sol: Solution = self.optimizer(self.opt_prob, sens=sens_func)

        np.set_printoptions(precision=8, linewidth=200, floatmode='fixed', suppress=False)
        print('\n...done!')
        print(f'\nmethod: {self.method}\n')

        # print(f'max of dm/dA: {np.max(dm)}')
        # print(f'max of ds/dA: {np.max(ds)}')
        print(f'Final mass: {sol.fStar}')
        print(f'Stresses:   {sol.constraints["stress_arr"].value}')
        print(f'Areas:      {sol.xStar["areas"]}')
        print(f'Number of function calls (pyOp count): {sol.userObjCalls}')
        print(f'Number of function calls (my own count): {self.eval_hist[-1]}')

        return sol


    def run(self):
        self.init_problem()
        sol = self.solve_problem()
        # self.plot_final_results(sol)
        return sol

def plot_final_results(op1, op2):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    # finite_diff + adjoint
    # x-axis: num func calls
    # y-axis norm of Lagrangian (max constraint?)

    xdata1 = op1.eval_hist
    xdata2 = op2.eval_hist
    
    axes[0].set_title("Truss Problem Convergence")
    axes[1].set_xlabel("Function calls")
    axes[0].set_ylabel("Mass (lbs)")
    axes[0].plot(xdata1, op1.mass_hist, label='FD')
    axes[0].plot(xdata2, op2.mass_hist, label='adj')
    axes[1].set_ylabel('Stress on bar (lbs)')
    axes[1].plot(xdata1, op1.stress_hist, label='FD')
    axes[1].plot(xdata2, op2.stress_hist, label='adj')
    axes[1].legend(loc='upper right')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.show()


if __name__ == '__main__':

    op_fd = truss_solver('my FD')
    sol_fd = op_fd.run()
    print('\nOptimization finished\n')

    op_adj = truss_solver('adjoint')
    sol_adj = op_adj.run()
    print('\nOptimization finished\n')

    rng = np.random.default_rng()
    k = 20
    evals1 = np.zeros(k)
    evals2 = np.zeros(k)
    for i in range(k):
        areas = rng.uniform(0.1,2.0, size=10)
        op_fd0 = truss_solver('my FD', areas)
        op_ad0 = truss_solver('adjoint', areas)
        sol1 = op_fd0.run()
        sol2 = op_ad0.run()

        evals1[i] = op_fd0.eval_hist[-1]
        evals2[i] = op_ad0.eval_hist[-1]

    print(f'\nAverge number of function calls required to converge:')
    print(f'\tFD: {np.mean(evals1)}')
    print(f'\tadjoint: {np.mean(evals2)}')
    
    plot_final_results(op_fd, op_adj)    