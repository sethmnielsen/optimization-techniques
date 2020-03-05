import numpy as np
import pyoptsparse as pyop
from pyoptsparse.pyOpt_solution import Solution
from pyoptsparse.pyOpt_optimization import Optimization
from pyoptsparse.pyOpt_optimizer import Optimizer
from pyoptsparse.pyOpt_history import History
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import seaborn as sns
sns.set_style('white')

class ConstrainedOptimizer:
    def __init__(self):
        self.marksz = 1.0

        # Figure for x, y points
        self.fig_pts = plt.figure()
        self.ax_pts = self.fig_pts.add_subplot(111)

        # Plotting histories
        self.time_hist = []
        self.wall_time_hist = []
        self.func_evals = []
        self.num_iterations = []

    def init_problem(self):
        self.px = np.zeros(6) + 1e-5
        self.py = np.zeros(6) + 1e-5

        self.n = 100
        self.t = np.linspace(0, 20, 100)
        self.t_exp = self.t**np.arange(6).reshape(6,1) # exponents for trajectory equation
        self.tvel_exp = self.t**np.array([0,0,1,2,3,4]) # exponenets for xdot, ydot
        self.tjerk_exp = self.t**np.array([0,0,0,0,1,2]).reshape(6,1) # exponents for each jerk term
        
        self.cvel = np.arange(6) # Constant mulipliers for each velocity term
        self.cjerk = np.array([0, 0, 0, 6, 24, 60]) # Constant multipliers for each jerk term
        
        xf = 10
        yf = 5
        self.x_arr = np.zeros(self.n) + 1e-5
        self.y_arr = np.zeros(self.n) + 1e-5
        self.x_arr[0] = 10
        self.y_arr[0] = 5 

        self.L = 2.5 # length of car
        gam_max = np.pi/4
        vmax_square = 10**2 # vmax = 10 m/s

        # Optimization problem
        self.opt_prob: pyop.Optimization = pyop.Optimization('differential_flat', self.objfunc)

        # Design variables
        self.opt_prob.addVarGroup('px', nVars=6, type='c', value=self.px, \
                                  lower=None, upper=None)

        self.opt_prob.addVarGroup('py', nVars=6, type='c', value=self.py, \
                                  lower=None, upper=None)

        #### CONSTRAINTS ####
        # start and finish constraints
        self.opt_prob.addConGroup('initial pos', nCon=2, lower=1e-5, upper=1e-5)
        self.opt_prob.addConGroup('initial vel', nCon=2, lower=[1e-5,2], upper=[1e-5,2])
        self.opt_prob.addConGroup('final pos', nCon=2, lower=[xf,yf], upper=[xf,yf])
        self.opt_prob.addConGroup('final vel', nCon=2, lower=[0,1], upper=[0,1])
        
        # constraints over entire trajectory
        self.opt_prob.addConGroup('vmax', nCon=self.n, lower=0, upper=vmax_square)
        self.opt_prob.addConGroup('gam_max', nCon=self.n, lower=-gam_max, upper=gam_max)
        # self.opt_prob.addConGroup('gam_max_plus', nCon=self.n, lower=0, upper=vmax_square)
        # self.opt_prob.addConGroup('gam_max_minus', nCon=self.n, lower=0, upper=vmax_square)
        
        # Assign the key value for the objective function
        self.opt_prob.addObj('obj-min-jerk')

        # Optimizer
        optimizer = pyop.SNOPT()
        # optimizer.setOption('iPrint',0)
        # path = '/home/seth/school/optimization/output/'
        # optimizer.setOption('Print file', path+f'SNOPT_print-{n}.out')
        # optimizer.setOption('Summary file', path+f'SNOPT_summary-{n}.out')

        return self.opt_prob, optimizer
        
        
    def trajectory(self, x, d):
        xd = self.px * self.t_exp
        yd = self.py * self.t_exp
        return xd, yd
    
    def inequality_constraints(self):
        vx = xdot / np.cos(thd)
        vy = ydot / np.sin(thd)
        
        v = vx**2
        thdot = ()
        
        
        
        return None

    def run(self):
        self.opt_prob = None
        self.optimizer = None
        self.opt_prob, self.optimizer = self.init_problem()
        self.solve_problem(self.opt_prob, self.optimizer)
        # self.plot_this_solution(n)

        # self.plot_final_results()

    def solve_problem(self, opt_prob: pyop.Optimization, optimizer: pyop.SNOPT):
        sol: Solution = optimizer(opt_prob, sens='CS', sensMode='pgc', storeHistory=f"output/diff_flat.hst")
        # sol: Solution = optimizer(opt_prob, sens='CS', sensMode=None, storeHistory=f"output/diff_flat.hst")

        print("...done!")

        print(f'sol.fStar:  {sol.fStar}')

        # self.y_arr[1:-1] = sol.xStar['y']
        self.time_hist.append(sol.fStar)
        self.wall_time_hist.append(sol.optTime)
        self.func_evals.append(sol.userObjCalls)
        print("sol.optTime:", sol.optTime)
        print("Calls to objective function:", sol.userObjCalls)
        # print("Iterations:", sol.)
        print(f"Printing solution:\n", sol)


    def objfunc(self, data):
        funcs = {}
        px = data['px'].reshape(6,1)
        py = data['py'].reshape(6,1)
        
        pxtjerk = px * self.tjerk_exp
        pytjerk = py * self.tjerk_exp
        x_jerk = np.sum(self.cjerk @ pxtjerk)
        y_jerk = np.sum(self.cjerk @ pytjerk)
        
        jerk_total = x_jerk**2 + y_jerk**2

        self.inequality_constraints(px, py)
        
        funcs['obj'] = jerk_total
        fail = False
        return funcs, fail

    def plot_this_solution(self, n):
        props = {
            'label': n,
            'marker': 'o',
            'markersize': 12*self.marksz
        }
        self.marksz *= 0.75
        self.ax_pts.plot(self.x_arr, self.y_arr, **props)

#region
    def plot_final_results(self):
        pass
    #     props = {
    #         'ls': '--',
    #         'color': 'k'
    #     }
    #     self.ax_pts.plot([0,1], [1,0], **props)
    #     self.ax_pts.set_title('Brachistochrone Optimization Problem')
    #     self.ax_pts.set_xlabel('x')
    #     self.ax_pts.set_ylabel('y')
    #     self.ax_pts.legend(title='No. Points')
    #     self.ax_pts.xaxis.set_major_locator(MultipleLocator(0.2))
    #     self.ax_pts.xaxis.set_minor_locator(AutoMinorLocator(5))
    #     self.ax_pts.yaxis.set_major_locator(MultipleLocator(0.2))
    #     self.ax_pts.yaxis.set_minor_locator(AutoMinorLocator(5))
    #     self.ax_pts.grid(which='major')
    #     self.ax_pts.grid(which='minor', alpha=0.2)

    #     fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    #     plt.gcf().subplots_adjust(left=0.15)

    #     axes[2].set_xlabel('number of points')
    #     axes[2].set_xticks(self.num_pts)
    #     axes[2].set_xticks(np.arange(0, self.num_pts[-1], 8), minor=True)

    #     axes[0].plot(self.num_pts, self.time_hist, color='blue')
    #     axes[0].set_title("Dimensionality")
    #     axes[0].set_ylabel("travel time (s)")
    #     axes[0].yaxis.set_major_locator(MultipleLocator(0.005))
    #     # axes[0].yaxis.set_minor_locator(AutoMinorLocator(5))
    #     axes[0].grid(which='major')
    #     axes[0].grid(which='minor', alpha=0.2)
    #     axes[0].set_ylim([0.62, 0.66])

    #     axes[1].plot(self.num_pts, self.wall_time_hist, color='orange')
    #     axes[1].set_ylabel("wall time (s)\n")
    #     axes[1].yaxis.set_major_locator(MultipleLocator(20))
    #     # axes[1].yaxis.set_minor_locator(AutoMinorLocator(5))
    #     axes[1].grid(which='major')
    #     axes[1].grid(which='minor', alpha=0.2)
    #     axes[1].set_ylim([0, None])

    #     axes[2].plot(self.num_pts, self.func_evals, color='green')
    #     axes[2].set_ylabel("function evaluations")
    #     axes[2].grid(which='major')
    #     axes[2].grid(which='minor', alpha=0.2)
    #     axes[2].set_xlim([self.num_pts[0], self.num_pts[-1]])

    #     plt.show()
#endregion

if __name__ == '__main__':
    op = ConstrainedOptimizer()
    op.run()
    print("Optimization successful")
