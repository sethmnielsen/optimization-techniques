import numpy as np
import pyoptsparse as pyop
import matplotlib.pyplot as plt
import time

# Solve the Brachistochrone Problem
# Find the min time between 2 pts for a particle subject to gravity only

# 1. Plot the optimal shape
# 2. Report the travel time between the two points
# 3. Study the effect of increased problem dimensionality

class Minimize:
    def __init__(self):
        # Create optimization problem
        self.mu = 0.3

        self.num_pts = np.array([5, 10, 20, 50, 100]) # number of pts including start and end
        self.warm_start = True

        x0, xk, y0, yk = 0, 1, 1, 0
        p0 = np.array([0.,1.])
        pk = np.array([1.,0.])

        # Optimization problem
        self.opt_prob: pyop.Optimization = pyop.Optimization('brachistochrone', self.objfunc)
        # Design variables
        self.opt_prob.addVarGroup('y', nVars=self.num_pts, type='c', value=self.yvals, \
                                  lower=None, upper=None)
        # Constraints ...needed?

        # Assign the key value for the objective function
        self.opt_prob.addObj('obj')
        print(self.opt_prob)

        # optimizer
        self.opt = pyop.SNOPT()
        self.opt.setOption('iPrint',-1)
    
    def run(self):
        for n in self.num_pts:
            self.solve_problem(n)
            
    def solve_problem(self, n):
        self.xvals = np.linspace(0, 1, n)  # fixed
        self.yvals = np.linspace(1-1/n, 1/n, n-2)  # initial guess

    
    def objfunc(self, g=None):
        funcs = {}
        funcs['obj'] = 
        all_y = np.zeros(num)
        all_y[0] = start_pt[1]
        all_y[1:-1] = y
        all_y[-1] = end_pt[1]

        sum = 0
        for i in range(0,num-1):

            xi = all_x[i]
            yi = all_y[i]
            xip1 = all_x[i+1]
            yip1 = all_y[i+1]
            dx = xip1-xi
            dy = yip1-yi
            if g == 0:
                sum += np.sqrt(dx**2+dy**2)/(np.sqrt(H-yip1-mu*xip1)+np.sqrt(H-yi-mu*xi))
            else:
                sum += np.sqrt(2/g) * np.sqrt(dx**2+dy**2)/(np.sqrt(H-yip1-mu*xip1)+np.sqrt(H-yi-mu*xi))

        return sum
    
    def sum_time(self):
        pass 