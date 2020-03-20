import numpy as np
from numpy import ndarray
import pyoptsparse as pyop
from pyoptsparse.pyOpt_solution import Solution
from pyoptsparse.pyOpt_optimization import Optimization
from pyoptsparse.pyOpt_optimizer import Optimizer
from pyoptsparse.pyOpt_history import History
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import List, Dict, Any

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

        self.n = 100
        self.t = np.linspace(0, 15, self.n)
        self.tpos_exp = self.t**np.arange(6).reshape(6,1) # exponents for trajectory equation
        self.tvel_exp = self.t**np.array([0,0,1,2,3,4]).reshape(6,1) # exponents for veloc terms
        self.tacc_exp = self.t**np.array([0,0,0,1,2,3]).reshape(6,1) # exponents for accel terms
        self.tjerk_exp = self.t**np.array([0,0,0,0,1,2]).reshape(6,1) # exponents for each jerk term
        
        self.cvel = np.arange(6) # Constant mulipliers for each velocity term
        self.cacc = np.array([0, 0, 2, 6, 12, 20]) # Constant mulipliers for each acceleration term
        self.cjerk = np.array([0, 0, 0, 6, 24, 60]) # Constant multipliers for each jerk term
        

        # Optimization problem
        opt_prob: pyop.Optimization = pyop.Optimization('differential_flat', self.objfunc)

        # Design variables
        opt_prob.addVarGroup('px', nVars=6, type='c', value=px, lower=None, upper=None)
        opt_prob.addVarGroup('py', nVars=6, type='c', value=py, lower=None, upper=None)

        x0, y0, xf, yf = [0., 0., 10., 0.]
        vx0, vy0, vxf, vyf = [0., 2., 0., 1.]

        self.L = 1.5 # length of car
        gam_max = np.pi/4
        self.vmax_square = 10**2 # vmax = 10 m/s
        self.amax_square = 2**2  # amax = 2 m/s**2

        #### CONSTRAINTS ####
        # start and finish constraints
        opt_prob.addConGroup('initial pos', nCon=2, lower=[x0,y0], upper=[x0,y0])
        opt_prob.addConGroup('initial vel', nCon=2, lower=[vx0,vy0], upper=[vx0,vy0])
        opt_prob.addConGroup('final pos', nCon=2, lower=[xf,yf], upper=[xf,yf])
        opt_prob.addConGroup('final vel', nCon=2, lower=[vxf,vyf], upper=[vxf,vyf])
        
        # constraints over entire trajectory
        opt_prob.addConGroup('v', nCon=self.n, lower=0, upper=self.vmax_square)
        opt_prob.addConGroup('a', nCon=self.n, lower=0, upper=self.amax_square)
        opt_prob.addConGroup('gam_max', nCon=self.n, lower=-gam_max, upper=gam_max)
        # opt_prob.addConGroup('gam_plus', nCon=self.n, lower=0, upper=gam_max)
        # opt_prob.addConGroup('gam_minus', nCon=self.n, lower=0, upper=-gam_max)
        
        # Assign the key value for the objective function
        opt_prob.addObj('obj-min-jerk')

        # Optimizer
        optimizer = pyop.SNOPT()
        optimizer.setOption('iPrint',0)
        optimizer.setOption('iSumm', 0)
        # path = '/home/seth/school/optimization/output/'
        # optimizer.setOption('Print file', path+f'SNOPT_print-{n}.out')
        # optimizer.setOption('Summary file', path+f'SNOPT_summary-{n}.out')

        return opt_prob, optimizer
        
    def position(self, px:ndarray, py:ndarray, idx=...):
        x = px @ self.tpos_exp[:,idx]
        y = py @ self.tpos_exp[:,idx]
        return x, y
    
    def velocity(self, px:ndarray, py:ndarray, idx:Any=...):
        xdot = (self.cvel * px) @ self.tvel_exp[:,idx]
        ydot = (self.cvel * py) @ self.tvel_exp[:,idx]
        return xdot, ydot
    
    def acceleration(self, px:ndarray, py:ndarray):
        xddot = (self.cacc * px) @ self.tacc_exp
        yddot = (self.cacc * py) @ self.tacc_exp
        return xddot, yddot
        
    def equality_constraints(self, funcs:Dict, px:ndarray, py:ndarray, vx, vy):
        x0, y0 = self.position(px, py, 0)
        xf, yf = self.position(px, py, -1)
        
        # Bundle up constraints
        funcs['initial pos'] = [x0, y0]
        funcs['initial vel'] = [vx[0], vy[0]] 
        funcs['final pos'] = [xf, yf]
        funcs['final vel'] = [vx[-1], vy[-1]]
        
        return funcs
        
    def inequality_constraints(self, funcs:Dict, px:ndarray, py:ndarray) -> Dict:
        vx, vy = self.velocity(px,py)
        ax, ay = self.acceleration(px,py)
        
        funcs = self.equality_constraints(funcs, px, py, vx, vy)
        
        v = np.sqrt(vx**2 + vy**2)
        a = np.sqrt(ax**2 + ay**2)
        gam_limit = v/self.L * np.tan(np.pi/4)
        current_gam = (vx*ay - vy*ax) / v**3
        gam_minus = gam_limit - current_gam
        gam_plus = gam_limit + current_gam
        
        funcs['v'] = v
        funcs['a'] = a
        # funcs['gam_plus'] = gam_plus 
        # funcs['gam_minus'] = gam_minus
        funcs['gam_max'] = current_gam
        
        # th = np.arctan2(vy, vx)
        # v = vx / np.cos(th)

        # thdot = ay*vx - vy*ax
        # gam = np.arctan2(thdot*self.L, v)
        
        return funcs

    def objfunc(self, data):
        funcs = {}
        px:ndarray = data['px']
        py:ndarray = data['py']
        
        x_jerk = (self.cjerk * px) @ self.tjerk_exp
        y_jerk = (self.cjerk * py) @ self.tjerk_exp
        
        jerk_total = np.sum(x_jerk**2 + y_jerk**2)

        funcs = self.inequality_constraints(px, py, funcs)
        
        funcs['obj-min-jerk'] = jerk_total
        fail = False
        return funcs, fail

    def run(self):
        opt_prob, optimizer = self.init_problem()
        self.solve_problem(opt_prob, optimizer)


    def solve_problem(self, opt_prob: pyop.Optimization, optimizer: pyop.SNOPT):
        # sol: Solution = optimizer(opt_prob, sens='CS', sensMode='pgc', storeHistory=f"output/diff_flat.hst")
        sol: Solution = optimizer(opt_prob, sens='CS', sensMode=None, storeHistory=None)
        
        px = sol.xStar['px']
        py = sol.xStar['py']
        x, y = self.position(px, py)
        vx, vy = self.velocity(px, py)
        ax, ay = self.acceleration(px, py)
        v = np.sqrt(vx**2 + vy**2)
        theta = np.rad2deg(np.arctan2(vy, vx))
        gamma = np.rad2deg(np.arctan2(self.L * (vx * ay - vy * ax), v**3)) 

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
        
        # self.plot_final_results(x, y, v, theta, gamma)


#region
    def plot_final_results(self, x, y, v, theta, gamma):
        props = {
            'ls': '--',
            'color': 'k'
        }
           
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, squeeze=False)
        fig: Figure
        ax: List[Axes]
        ax = ax[0]
        ax[0].set_title("Trajectory")
        ax[0].plot(x, y, linewidth=2, color='r')
        ax[0].set_xlabel(r"$x_d$")
        ax[0].set_ylabel(r"$y_d$")
        
        fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True)
        fig2: Figure
        ax2: List[Axes]
        ax2[0].set_title("States over time")
        ax2[2].set_xlabel(r"$t$")
        ax2[0].plot(self.t, x, linewidth=2, color='r')
        ax2[1].plot(self.t, y, linewidth=2, color='r')
        ax2[2].plot(self.t, theta, linewidth=2, color='r')
        ax2[0].set_ylabel(r"$x_d$")
        ax2[1].set_ylabel(r"$y_d$")
        ax2[2].set_ylabel(r"$\theta_d$")
        
        
        fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=True)
        fig2: Figure
        ax2: List[Axes]
        ax2[0].set_title("Inputs over time")
        ax2[1].set_xlabel(r"$t$")
        ax2[0].plot(self.t, v, linewidth=2, color='r')
        ax2[1].plot(self.t, gamma, linewidth=2, color='r')
        ax2[0].set_ylabel(r"$v$")
        ax2[1].set_ylabel(r"$\gamma$")
        
        plt.show()
        
#endregion

if __name__ == '__main__':
    op = ConstrainedOptimizer()
    op.run()
    print("Optimization successful")
