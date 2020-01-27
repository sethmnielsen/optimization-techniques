import numpy as np
import pyoptsparse as pyop
from pyoptsparse.pyOpt_solution import Solution
from pyoptsparse.pyOpt_optimization import Optimization
from pyoptsparse.pyOpt_optimizer import Optimizer
from pyoptsparse.pyOpt_history import History
import matplotlib.pyplot as plt
import time
import traceback

import seaborn as sns
sns.set_style('whitegrid')

# Solve the Brachistochrone Problem
# Find the min time between 2 pts for a particle subject to gravity only

# 1. Plot the optimal shape
# 2. Report the travel time between the two points
# 3. Study the effect of increased problem dimensionality

class Minimize:
    def __init__(self):
        self.num_pts = np.array([4, 8, 16, 32]) # number of pts including start and end
        # self.num_pts = np.array([4])
        self.warm_start = True
        self.ms = 1.0

        # Figure for x, y points 
        self.fig_pts = plt.figure()
        self.ax_pts = self.fig_pts.add_subplot(111)

        # Plotting histories
        self.time_hist = []
        self.wall_time_hist = []
        self.func_evals = []
        self.num_iterations = []

    def init_problem(self, n):
        print(f"Solving Brachistochrone problem with n = {n}...")
        
        self.n = n
        self.x_arr = np.linspace(0, 1, n)  # fixed
        if self.warm_start:
            y_inner = np.linspace(1-1/n, 1/n, n-2)  # initial guess of (inner) values
        else:
            y_inner = np.zeros(n-2)

        # Optimal/final y points
        self.y_arr = np.zeros(n)
        self.y_arr[0] = 1.0

        # Optimization problem
        opt_prob: pyop.Optimization = pyop.Optimization('brachistochrone', self.objfunc)
        # Design variables
        opt_prob.addVarGroup('y', nVars=n-2, type='c', value=y_inner, \
                                  lower=None, upper=None)

        # Assign the key value for the objective function
        opt_prob.addObj('obj')

        # Optimizer
        optimizer = pyop.SNOPT()
        optimizer.setOption('iPrint',0)

        return opt_prob, optimizer
    
    def run(self):
        for n in self.num_pts:
            opt_prob = None
            optimizer = None
            opt_prob, optimizer = self.init_problem(n)
            self.solve_problem(opt_prob, optimizer)
            self.plot_this_solution(n)

        self.plot_final_results()

    def solve_problem(self, opt_prob: pyop.Optimization, optimizer: pyop.SNOPT):
        sol: Solution = optimizer(opt_prob, storeHistory=f"opt_hist{self.n}.hst")
        
        print("...done!")
        
        sol.fStar *= np.sqrt(2/9.81)
        print(f'sol.fStar: {sol.fStar}')
        print(f'self.test: {self.test}')
        print(f'self.test2: {self.test2}')
        
        self.y_arr[1:-1] = sol.xStar['y']
        self.time_hist.append(sol.fStar)
        self.wall_time_hist.append(sol.optTime)
        self.func_evals.append(sol.userObjCalls)
        print("sol.optTime:", sol.optTime)
        print("Calls to objective function:", sol.userObjCalls)
        print(f"Printing solution for n = {self.n}:", sol)
    
    def objfunc(self, dvars):
        g = 9.81
        h = 1.0
        mu = 0.3

        funcs = {}
        y = np.zeros(self.n)
        y[1:-1] = dvars['y']
        y[0] = h 

        time_sum = 0
        time_sum2 = 0
        for i in range(self.n-1):
            # Loop over x and y arrays
            xi = self.x_arr[i]
            xip = self.x_arr[i+1]
            yi = y[i]
            yip = y[i+1]
            dx = xip-xi
            dy = yip-yi
            
            # Gravity not needed - will multiply it for final result 
            a = np.sqrt(2.0/g)
            b = np.sqrt(dx**2+dy**2) 
            c = np.sqrt(h-yip-mu*xip) + np.sqrt(h-yi-mu*xi)

            time_sum += b/c
            time_sum2 += a*b/c

        funcs['obj'] = time_sum
        fail = False
        self.test = time_sum * a
        self.test2 = time_sum2
        return funcs, fail
    
    def plot_this_solution(self, n):
        props = {
            'label': n,
            'marker': 'o', 
            'markersize': 12*self.ms
        }
        self.ms *= 0.75
        self.ax_pts.plot(self.x_arr, self.y_arr, **props)
    
    def plot_final_results(self):
        self.ax_pts.set_title('Brachistochrone Optimization Problem')
        self.ax_pts.set_xlabel('x')
        self.ax_pts.set_ylabel('y')
        self.ax_pts.legend(title='No. Points')

        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
        # plt.gcf().subplots_adjust(left=0.15)
        
        axes[0].plot(self.num_pts, self.time_hist)
        axes[0].set_title("Dimensionality")
        axes[0].set_ylabel("travel time (s)")
        
        axes[1].plot(self.num_pts, self.wall_time_hist)
        axes[1].set_ylabel("wall\ntime (s)")
        
        axes[2].plot(self.num_pts, self.func_evals)
        axes[2].set_ylabel("function\nevaluations")
        axes[2].set_xlabel('number of points')
        
        plt.show()

if __name__ == '__main__':
    op = Minimize()
    op.run()
    print("Optimization successful")
    # try:
        # op.run()
        # print("Optimization successful")
    # except Exception as e:
        # print("Exception occured, printing traceback...\n")
        # print(traceback.format_exc())
        # print(e)
