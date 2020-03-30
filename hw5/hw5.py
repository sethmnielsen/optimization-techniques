## %%
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
from typing import Tuple, List, Dict, Any

import seaborn as sns
sns.set_style('white')

class QuadTrajectoryGenerator:
    def __init__(self):
        # Plotting histories
        p0 = np.zeros(6) + 1e-4

        self.n = 100
        self.t = np.linspace(0, 15, self.n)
        self.tpos_exp = self.t**np.arange(6).reshape(6,1) # exponents for trajectory equation
        self.tvel_exp = self.t**np.array([0,0,1,2,3,4]).reshape(6,1) # exponents for veloc terms
        self.tacc_exp = self.t**np.array([0,0,0,1,2,3]).reshape(6,1) # exponents for accel terms
        self.tjerk_exp = self.t**np.array([0,0,0,0,1,2]).reshape(6,1) # exponents for each jerk term
        
        self.cvel = np.arange(6, dtype=np.float64) # Constant mulipliers for each velocity term
        self.cacc = np.array([0, 0, 2, 6, 12, 20], dtype=np.float64) # Constant mulipliers for each acceleration term
        self.cjerk = np.array([0, 0, 0, 6, 24, 60], dtype=np.float64) # Constant multipliers for each jerk term

        # Optimization problem
        opt_prob: pyop.Optimization = pyop.Optimization('differential_flat_quad', self.objfunc)

        # Design variables
        opt_prob.addVarGroup('px', nVars=6, type='c', value=p0, lower=None, upper=None)
        opt_prob.addVarGroup('py', nVars=6, type='c', value=p0, lower=None, upper=None)
        opt_prob.addVarGroup('pz', nVars=6, type='c', value=p0, lower=None, upper=None)
        opt_prob.addVarGroup('ps', nVars=6, type='c', value=p0, lower=None, upper=None)

        # ENU
        #      [ x,   y,   z, psi ]
        pos0 = np.array([0.,  0.,  0.,  0. ])
        posf = np.array([0., 10., 10.,  0. ])
        vel0 = np.array([0.,  2.,  2.,  0. ])
        velf = np.array([0.,  2.,  0.,  0. ])
        # x0, y0, z0, psi0 = [0.,  0.,  0., 0.]
        # xf, yf, zf, psif = [0., 10., 10., 0.]
        # vx0, vy0, vz0, w0 = [0., 2., 2., 0.]
        # vxf, vyf, vzf, wf = [0., 2., 0., 0.]

        self.L = 1.5 # length of car
        gam_max = np.pi/4
        self.vmax_square = 10**2 # vmax = 10 m/s
        self.amax_square = 2**2  # amax = 2 m/s**2

        #### CONSTRAINTS ####
        # start and finish constraints
        opt_prob.addConGroup('initial pos', nCon=4, lower=pos0, upper=pos0)
        opt_prob.addConGroup('initial vel', nCon=4, lower=vel0, upper=vel0)
        opt_prob.addConGroup('final pos',   nCon=4, lower=posf, upper=posf)
        opt_prob.addConGroup('final vel',   nCon=4, lower=velf, upper=velf)
        
        # constraints over entire trajectory
        # opt_prob.addConGroup('v', nCon=self.n, lower=0, upper=self.vmax_square)
        # opt_prob.addConGroup('a', nCon=self.n, lower=0, upper=self.amax_square)
        # opt_prob.addConGroup('gam_max', nCon=self.n, lower=-gam_max, upper=gam_max)
        # opt_prob.addConGroup('gam_plus', nCon=self.n, lower=0, upper=None)
        # opt_prob.addConGroup('gam_minus', nCon=self.n, lower=0, upper=None)
        
        # Assign the key value for the objective function
        opt_prob.addObj('obj-min-jerk')

        # Optimizer
        optimizer = pyop.SNOPT()
        # optimizer.setOption('iPrint',0)
        # optimizer.setOption('iSumm', 0)
        path = '/home/seth/school/optimization/output/'
        optimizer.setOption('Print file', path+f'SNOPT-hw5.out')
        optimizer.setOption('Summary file', path+f'SNOPT-hw5-summary.out')

        self.opt_prob: pyop.Optimization = opt_prob 
        self.optimizer: pyop.SNOPT = optimizer
        
    def objfunc(self, data:Dict):
        px: ndarray = data['px'] 
        py: ndarray = data['py']
        pz: ndarray = data['pz']
        ps: ndarray = data['ps']
        
        p = np.array([px, py, pz, ps])

        funcs: Dict = {}
        funcs = self.equality_constraints(funcs, p)
        jerk_total = self.jerk(p)
        funcs['obj-min-jerk'] = jerk_total
        fail = False
        return funcs, fail
    
    def equality_constraints(self, funcs:Dict, p) -> Dict:
        v = np.array(self.velocity(p))
        
        # Bundle up constraints
        funcs['initial pos'] = np.array([self.position(p,0)])
        funcs['initial vel'] = np.array([v[0,0], v[1,0], v[2,0], v[3,0]])
        funcs['final pos']   = np.array([self.position(p, -1)])
        funcs['final vel']   = np.array([v[0,-1], v[1,-1], v[2,-1], v[3,-1]])
        
        return funcs
    
    def derivs(self, p, c, t_exp, idx=...):
        x = (c*p[0]) @ t_exp[:,idx]
        y = (c*p[1]) @ t_exp[:,idx]
        z = (c*p[2]) @ t_exp[:,idx]
        s = (c*p[3]) @ t_exp[:,idx]
        return x, y, z, s

        
    def position(self, p:ndarray, idx=...):
        return self.derivs(p, 1, self.tpos_exp, idx)
    
    def velocity(self, p:ndarray, idx:Any=...):
        return self.derivs(p, self.cvel, self.tvel_exp, idx)
    
    def acceleration(self, p:ndarray):
        return self.derivs(p, self.cacc, self.tacc_exp)
    
    def jerk(self, p:ndarray) -> float:
        jerk = np.array(self.derivs(p, self.cjerk, self.tjerk_exp))
        jerk_total = np.sum(jerk**2)
        return jerk_total
        
    def run(self):
        # sol: Solution = self.optimizer(self.opt_prob, sens='CS', sensMode=None, storeHistory=None)
        sol: Solution = self.optimizer(self.opt_prob, sens='CS', sensMode=None, storeHistory=f'/home/seth/school/optimization/output/hw5.hst')
        
        px = sol.xStar['px']
        py = sol.xStar['py']
        pz = sol.xStar['pz']
        ps = sol.xStar['ps']
        p = np.array([px, py, pz, ps])
        pos = self.position(p)
        v = np.array(self.velocity(p))
        a = np.array(self.acceleration(p))
        vn = np.sqrt(np.sum(v**2))
        theta = np.rad2deg(np.arctan2(v[1], v[0]))
        gamma = np.rad2deg(np.arctan2(self.L * (v[0] * a[1] - v[1] * a[0]), vn**3)) 

        print("\n\n...done!\n\n")
        # print("PRINTING SOLUTION\n\n")
        # print(sol)

        print(f'sol.fStar:  {sol.fStar}')
        print(f'sol.xStar[\'px\']: {p[0]}')
        print(f'sol.xStar[\'py\']: {p[1]}')
        print(f'sol.xStar[\'pz\']: {p[2]}')
        print(f'sol.xStar[\'ps\']: {p[3]}')
        print(f'userObjCalls: {sol.userObjCalls}')

        self.plot_final_results(p, theta)


    def plot_final_results(self, p, theta):
        props = {
            'ls': '--',
            'color': 'k'
        }
        
        x, y, z, psi = self.position(p)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig: Figure
        ax: Axes
        ax.set_title("Trajectory")
        ax.plot(x, y, z, linewidth=2, color='r')
        ax.set_xlabel(r"$x_d$")
        ax.set_ylabel(r"$y_d$")
        # ax.set_zlabel(r"$z_d$")
        
        # fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True)
        # fig2: Figure
        # ax2: List[Axes]
        # ax2[0].set_title("States over time")
        # ax2[2].set_xlabel(r"$t$")
        # ax2[0].plot(self.t, x, linewidth=2, color='r')
        # ax2[1].plot(self.t, y, linewidth=2, color='r')
        # ax2[2].plot(self.t, theta, linewidth=2, color='r')
        # ax2[0].set_ylabel(r"$x_d$")
        # ax2[1].set_ylabel(r"$y_d$")
        # ax2[2].set_ylabel(r"$\theta_d$")
        
        # fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=True)
        # fig2: Figure
        # ax2: List[Axes]
        # ax2[0].set_title("Inputs over time")
        # ax2[1].set_xlabel(r"$t$")
        # ax2[0].plot(self.t, v, linewidth=2, color='r')
        # ax2[1].plot(self.t, gamma, linewidth=2, color='r')
        # ax2[0].set_ylabel(r"$v$")
        # ax2[1].set_ylabel(r"$\gamma$")
        
        plt.show()
        
    def gamma(self, vn, v, a):
        gam_limit = vn/self.L * np.tan(np.pi/4)
        current_gam = (v[0]*a[1] - v[1]*a[0]) / vn**3
        gam_minus = gam_limit - current_gam
        gam_plus = gam_limit + current_gam
        return current_gam, gam_minus, gam_plus
    
    def inequality_constraints_old(self, funcs:Dict, p:ndarray) -> Dict:
        v = np.array(self.velocity(p))
        a = np.array(self.acceleration(p))
        
        funcs = self.equality_constraints(funcs, p, v)
        
        vn = np.sqrt(np.sum(v**2))
        an = np.sqrt(np.sum(a**2))
        
        funcs['v'] = vn
        funcs['a'] = an
        current_gam, gam_minus, gam_plus = self.gamma(vn, v, a)
        funcs['gam_plus'] = gam_plus 
        funcs['gam_minus'] = gam_minus
        funcs['gam_max'] = current_gam
        
        th = np.arctan2(vy, vx)
        v = vx / np.cos(th)
        thdot = ay*vx - vy*ax
        gam = np.arctan2(thdot*self.L, v)
        return funcs


if __name__ == '__main__':
    tg = QuadTrajectoryGenerator()
    tg.run()
    print("Optimization successful")


# %%
