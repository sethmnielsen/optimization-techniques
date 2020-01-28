#!/usr/bin/env python3

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
from scipy.optimize import NonlinearConstraint
import pyoptsparse as pyop
from pyoptsparse.pyOpt_solution import Solution
from pyoptsparse.pyOpt_optimization import Optimization
from pyoptsparse.pyOpt_optimizer import Optimizer
from truss import truss
import seaborn as sns
sns.set_style('whitegrid')

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

        self.start = np.ones(10) * 0.1

    def init_problem(self):
        # Optimal/final DVs
        self.areas = np.zeros(10)

        # Optimization problem
        opt_prob: pyop.Optimization = pyop.Optimization('trusty', self.objfunc)
        # Design variables
        opt_prob.addVarGroup('areas', nVars=10, type='c', value=self.start, \
                                  lower=0.1, upper=None)
                            
        # Constraints
        # yield_compression, yield_tension, area_min
        stress_yield = np.array([25e3]*10)
        stress_yield[8] = 75e3 
        opt_prob.addConGroup('stress_arr', nCon=10, lower=-stress_yield, upper=stress_yield)

        # Assign the key value for the objective function
        opt_prob.addObj('mass')

        
        optimizer = pyop.SNOPT()
        optimizer.setOption('iPrint',0)
        path = '/home/seth/school/optimization/output/'
        optimizer.setOption('Print file', path+f'SNOPT_print.out')
        optimizer.setOption('Summary file', path+f'SNOPT_summary.out')

        return opt_prob, optimizer
    
    def run(self):
        opt_prob, optimizer = self.init_problem()
        sol = self.solve_problem(opt_prob, optimizer)
        self.plot_final_results(sol)

    def solve_problem(self, opt_prob: pyop.Optimization, optimizer: pyop.SNOPT):
        sol: Solution = optimizer(opt_prob)

        print("...done!")

        print(f'Final mass: {sol.fStar}')
        print(f'Stresses:   {sol.constraints["stress_arr"].value}')
        print(f'Areas:      {sol.xStar["areas"]}')
        print(f'Number of function calls: {sol.userObjCalls}')

        return sol

    
    def objfunc(self, data):
        x = data['areas']
        funcs = {}
        mass, stress = truss(x)
        funcs['mass'] = mass
        funcs['stress_arr'] = stress

        self.mass_hist.append(mass)
        self.stress_hist.append(stress)
        return funcs

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
        # axes[0].yaxis.set_major_locator(MultipleLocator(0.005))
        # # axes[0].yaxis.set_minor_locator(AutoMinorLocator(5))
        # axes[0].grid(which='major')
        # axes[0].grid(which='minor', alpha=0.2)
        # axes[0].set_ylim([0.62, 0.66])
        
        # axes[1].plot(self.num_pts, self.wall_time_hist, color='orange')
        # axes[1].set_ylabel("wall time (s)\n")
        # axes[1].yaxis.set_major_locator(MultipleLocator(20))
        # # axes[1].yaxis.set_minor_locator(AutoMinorLocator(5))
        # axes[1].grid(which='major')
        # axes[1].grid(which='minor', alpha=0.2)
        # axes[1].set_ylim([0, None])
        
        # axes[2].plot(self.num_pts, self.func_evals, color='green')
        # axes[2].set_ylabel("function evaluations")
        # axes[2].grid(which='major')
        # axes[2].grid(which='minor', alpha=0.2)
        # axes[2].set_xlim([self.num_pts[0], self.num_pts[-1]])
        
        plt.show()

        
if __name__ == '__main__':
    op = truss_solver()
    op.run()
    print('Optimization finished')