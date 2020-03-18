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
        # Plotting histories
        self.time_hist = []
        self.wall_time_hist = []
        self.func_evals = []
        self.num_iterations = []

    def init_problem(self):
        px = np.zeros(6) + 1e-4
        py = np.zeros(6) + 1e-4

        self.n = 400
        self.t = np.linspace(0, 15, 100)
        self.t_exp = self.t**np.arange(6).reshape(6,1) # exponents for trajectory equation
        self.tvel_exp = self.t**np.array([0,0,1,2,3,4]).reshape(6,1) # exponents for veloc terms
        self.tacc_exp = self.t**np.array([0,0,0,1,2,3]).reshape(6,1) # exponents for accel terms
        self.tjerk_exp = self.t**np.array([0,0,0,0,1,2]).reshape(6,1) # exponents for each jerk term
        
        self.cvel = np.arange(6) # Constant mulipliers for each velocity term
        self.cacc = np.array([0, 0, 2, 6, 12, 20]) # Constant mulipliers for each acceleration term
        self.cjerk = np.array([0, 0, 0, 6, 24, 60]) # Constant multipliers for each jerk term
        
        xf = 10
        yf = 5
        self.x_arr = np.zeros(self.n) + 1e-5
        self.y_arr = np.zeros(self.n) + 1e-5
        self.x_arr[0] = 10
        self.y_arr[0] = 5 

        self.L = 1.5 # length of car
        gam_max = np.pi/4
        vmax_square = 10**2 # vmax = 10 m/s

        # Optimization problem
        self.opt_prob: pyop.Optimization = pyop.Optimization('differential_flat', self.objfunc)

        # Design variables
        self.opt_prob.addVarGroup('px', nVars=6, type='c', value=px, \
                                  lower=None, upper=None)

        self.opt_prob.addVarGroup('py', nVars=6, type='c', value=py, \
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
        
        
    def trajectory(self, px, py):
        x = px @ self.t_exp
        y = py @ self.t_exp
        return x, y
    
    def velocity(self, px, py):
        xdot = (self.cvel * px) @ self.tvel_exp
        ydot = (self.cvel * py) @ self.tvel_exp
        return xdot, ydot
    
    def acceleration(self, px, py):
        xddot = (self.cacc * px) @ self.tacc_exp
        yddot = (self.cacc * py) @ self.tacc_exp
        return xddot, yddot
        
    def inequality_constraints(self, px, py):
        xdot, ydot = self.velocity(px,py)
        xacc, yacc = self.acceleration(px,py)
        
        v = np.sqrt(xdot**2 + ydot**2)
        a = np.sqrt(xacc**2 + yacc**2)
        arg1 = v/1 + np.tan(np.pi/4)
        arg2 = (xdot*yacc - ydot*xacc) / v**3
        gmax = arg1 - arg2
        gmax2 = arg1 + arg2
        
        # th = np.arctan2(ydot, xdot)
        # v = xdot / np.cos(th)

        # thdot = yacc*xdot - ydot*xacc
        # gam = np.arctan2(thdot*self.L, v)
        
        return v, gam, xdot, ydot 

    def objfunc(self, data):
        funcs = {}
        px = data['px']
        py = data['py']
        
        x_jerk = (self.cjerk * px) @ self.tjerk_exp
        y_jerk = (self.cjerk * py) @ self.tjerk_exp
        
        jerk_total = np.sum(x_jerk**2 + y_jerk**2)

        x, y = self.trajectory(px, py)
        v, gam, xdot, ydot = self.inequality_constraints(px, py)
        
        # Bundle up constraints
        funcs['initial pos'] = [x[0], y[0]]
        funcs['initial vel'] = [xdot[0], ydot[0]] 
        funcs['final pos'] = [x[-1], y[-1]]
        funcs['final vel'] = [xdot[-1], ydot[-1]]
        
        funcs['vmax'] = v
        funcs['gam_max'] = gam
        
        funcs['obj-min-jerk'] = jerk_total
        fail = False
        return funcs, fail

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
        
        px = sol.xStar['px']
        py = sol.xStar['py']
        x, y = self.trajectory(px, py)

        print("...done!")

        print(f'sol.fStar:  {sol.fStar}')
        print(f'sol.xStar[\'px\']: {px}')
        print(f'sol.xStar[\'py\']: {py}')

        self.time_hist.append(sol.fStar)
        self.wall_time_hist.append(sol.optTime)
        self.func_evals.append(sol.userObjCalls)
        print("sol.optTime:", sol.optTime)
        print("Calls to objective function:", sol.userObjCalls)
        # print("Iterations:", sol.)
        # print(f"Printing solution:\n", sol)
        
        self.plot_final_results(x, y)


#region
    def plot_final_results(self, x, y):
        props = {
            'ls': '--',
            'color': 'k'
        }
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)

        ax.set_title(r"$x_d$ vs $y_d$")
        ax.plot(x, y, linewidth=2, color='r')
        
        ax.set_xlabel(r"$x_d$")
        ax.set_ylabel(r"$y_d$")
        # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.show()
#endregion

if __name__ == '__main__':
    op = ConstrainedOptimizer()
    op.run()
    print("Optimization successful")
